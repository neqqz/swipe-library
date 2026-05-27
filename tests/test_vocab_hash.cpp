/**
 * Unit tests for vocab_hash: wyhash, multiply-shift, and VocabTable.
 * Does not require ExecuTorch build.
 *
 * Pass --bench to enable throughput benchmarks.
 */

#include "swipe_decoder/vocab_hash.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

// ============================================================================
// Test framework: deferred execution so main() can parse flags first
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static bool run_benchmarks = false;

struct TestFailed {};

using TestFunc = void(*)();
struct TestEntry { const char* name; TestFunc func; };
static std::vector<TestEntry>& test_registry() {
    static std::vector<TestEntry> r;
    return r;
}

#define TEST(name) \
    static void test_##name(); \
    static struct Register_##name { \
        Register_##name() { test_registry().push_back({#name, test_##name}); } \
    } register_##name; \
    static void test_##name()

#define ASSERT(cond) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond); throw TestFailed{}; } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { printf("FAIL\n    %s:%d: %s (%llu) != %s (%llu)\n", \
        __FILE__, __LINE__, #a, (unsigned long long)_a, #b, (unsigned long long)_b); throw TestFailed{}; } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    if ((a) != (b)) { printf("FAIL\n    %s:%d: \"%s\" != \"%s\"\n", \
        __FILE__, __LINE__, std::string(a).c_str(), std::string(b).c_str()); throw TestFailed{}; } \
} while(0)

using namespace swipe_decoder;

// ============================================================================
// wyhash correctness: golden values from Python
// ============================================================================

TEST(wyhash_empty_string) {
    uint64_t h = wyhash("", 0);
    ASSERT_EQ(h, 1471106076447886454ULL);
}

TEST(wyhash_1_to_3_bytes) {
    ASSERT_EQ(wyhash("a", 1),   1336536464068998331ULL);
    ASSERT_EQ(wyhash("ab", 2),  15343311779450632941ULL);
    ASSERT_EQ(wyhash("abc", 3), 6964467373594007166ULL);
}

TEST(wyhash_4_to_16_bytes) {
    ASSERT_EQ(wyhash("abcd", 4),  6924378447136111592ULL);
    ASSERT_EQ(wyhash("hello", 5), 66478811968527032ULL);
    ASSERT_EQ(wyhash("the", 3),   5933384505577477024ULL);
    ASSERT_EQ(wyhash("don't", 5), 15644294451650341770ULL);
    ASSERT_EQ(wyhash("abcdefgh", 8), 17179726566512205478ULL);

    // 15 chars (boundary)
    ASSERT_EQ(wyhash("abcdefghijklmno", 15), 13773973106979503683ULL);
    // 16 chars (boundary)
    ASSERT_EQ(wyhash("abcdefghijklmnop", 16), 2149303937770850849ULL);
}

TEST(wyhash_17_to_48_bytes) {
    // 17 chars - enters the 17..48 branch
    ASSERT_EQ(wyhash("abcdefghijklmnopq", 17), 11444799058722442184ULL);
}

TEST(wyhash_over_48_bytes) {
    // 50-char string - enters the >48 triple-accumulator branch
    const char* s50 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX";
    ASSERT_EQ(strlen(s50), 50u);
    ASSERT_EQ(wyhash(s50, 50), 1187001438105522303ULL);
}

TEST(wyhash_with_seed) {
    // seed=0x12345 (used by VocabTable::find)
    ASSERT_EQ(wyhash("hello", 5, 0x12345), 1160519760136090597ULL);
    ASSERT_EQ(wyhash("the", 3, 0x12345),   16506902160635417505ULL);
    ASSERT_EQ(wyhash("don't", 5, 0x12345), 12926329045876433123ULL);
}

TEST(wymix_portable_matches_native) {
    // Test portable 32-bit _wymix against known-good values from __uint128_t path.
    // Covers: zero, small, large, max, and high-carry edge cases.
    struct { uint64_t a, b, expected; } cases[] = {
        {0, 0, 0},
        {1, 1, 1},
        {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE00000001ULL ^ 0},
        {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
         _wymix(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL)},
        {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL,
         _wymix(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL)},
        {0x8000000000000000ULL, 0x8000000000000000ULL,
         _wymix(0x8000000000000000ULL, 0x8000000000000000ULL)},
        {0x00000001FFFFFFFFULL, 0xFFFFFFFF00000001ULL,
         _wymix(0x00000001FFFFFFFFULL, 0xFFFFFFFF00000001ULL)},
    };
    for (auto& c : cases) {
        uint64_t got = _wymix_portable(c.a, c.b);
        if (got != c.expected) {
            printf("FAIL: _wymix_portable(0x%llx, 0x%llx) = 0x%llx, expected 0x%llx\n",
                   (unsigned long long)c.a, (unsigned long long)c.b,
                   (unsigned long long)got, (unsigned long long)c.expected);
            throw TestFailed{};
        }
    }
}

// ============================================================================
// multiply_shift + compute_hash_indices_wy
// ============================================================================

TEST(compute_hash_indices_32768) {
    uint32_t out[2];

    compute_hash_indices_wy("hello", 5, 32768, out);
    ASSERT_EQ(out[0], 22259u);
    ASSERT_EQ(out[1], 20925u);

    compute_hash_indices_wy("the", 3, 32768, out);
    ASSERT_EQ(out[0], 31603u);
    ASSERT_EQ(out[1], 14789u);

    compute_hash_indices_wy("don't", 5, 32768, out);
    ASSERT_EQ(out[0], 31366u);
    ASSERT_EQ(out[1], 22265u);

    compute_hash_indices_wy("xylophone", 9, 32768, out);
    ASSERT_EQ(out[0], 1546u);
    ASSERT_EQ(out[1], 30130u);

    compute_hash_indices_wy("a", 1, 32768, out);
    ASSERT_EQ(out[0], 14655u);
    ASSERT_EQ(out[1], 4630u);
}

TEST(compute_hash_indices_256) {
    uint32_t out[2];

    compute_hash_indices_wy("hello", 5, 256, out);
    ASSERT_EQ(out[0], 173u);
    ASSERT_EQ(out[1], 163u);

    compute_hash_indices_wy("the", 3, 256, out);
    ASSERT_EQ(out[0], 246u);
    ASSERT_EQ(out[1], 115u);
}

TEST(bench_wyhash) {
    if (!run_benchmarks) return;

    // Hash 1M strings of various lengths
    std::vector<std::string> strings;
    size_t total_bytes = 0;
    const int N = 1000000;
    strings.reserve(N);
    for (int i = 0; i < N; ++i) {
        int len = 3 + (i % 20);  // 3..22 chars
        std::string s(len, 'a');
        for (int j = 0; j < len; ++j) s[j] = 'a' + (j + i) % 26;
        total_bytes += s.size();
        strings.push_back(std::move(s));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    volatile uint64_t sink = 0;
    for (auto& s : strings) {
        sink = wyhash(s.data(), s.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb = total_bytes / (1024.0 * 1024.0);
    printf("1M strings (%.1f MB) in %.1f ms (%.0f MB/s)\n",
           mb, ms, mb / (ms / 1000.0));
}

// ============================================================================
// Main: parse flags, then run registered tests
// ============================================================================

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--bench") == 0) run_benchmarks = true;

    printf("Running vocab_hash tests...\n\n");

    for (auto& t : test_registry()) {
        tests_run++;
        printf("  %-50s ", t.name);
        try {
            t.func();
            tests_passed++;
            printf("PASS\n");
        } catch (TestFailed&) {
            // already printed
        } catch (...) {
            printf("FAIL (exception)\n");
        }
    }

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
