cmake_minimum_required(VERSION 3.17)
project(swipe_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ETI_IS_TOP_LEVEL YES)
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(ETI_IS_TOP_LEVEL NO)
endif()


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Sanitizer support (ASan + UBSan)
option(SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
if(SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

# Android vs host build
if(ANDROID)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
else()
    option(USE_MARCH_NATIVE "Use -march=native for CPU-specific optimization" ON)
    if(USE_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    endif()
endif()

# ExecuTorch setup
include(cmake/executorch.cmake)

# ===========================================================================
# Core static library: libswipe_decoder
# ===========================================================================

set(CORE_SOURCES
    src/resampler.cpp
    src/beam_search.cpp
    src/trie.cpp
    src/unicode_case.cpp
)

if(HAVE_EXECUTORCH)
    list(APPEND CORE_SOURCES
        src/encoder.cpp
        src/decoder.cpp
        src/engine.cpp
        src/context_lm.cpp
        src/model_metadata.cpp
    )
endif()

add_library(swipe_decoder STATIC ${CORE_SOURCES})

target_include_directories(swipe_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(HAVE_EXECUTORCH)
    target_compile_definitions(swipe_decoder PUBLIC HAVE_EXECUTORCH)
    # Include dirs only (not link, whole-archive must be on final binary)
    target_include_directories(swipe_decoder PRIVATE ${ET_INCLUDE_DIRS})
    target_compile_definitions(swipe_decoder PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
    # ET is built with EXECUTORCH_OPTIMIZE_SIZE=ON (-fno-rtti). Match it so that
    # TUs referencing ET polymorphic types (e.g. std::unique_ptr<Module>) don't
    # emit typeinfo references that aren't in libexecutorch.a.
    target_compile_options(swipe_decoder PUBLIC -fno-rtti)
endif()

# ===========================================================================
# JNI shared library (Android only)
# ===========================================================================

if(ANDROID AND HAVE_EXECUTORCH)
    add_library(swipe_jni SHARED jni/swipe_jni.cpp)
    target_link_libraries(swipe_jni PRIVATE swipe_decoder)
    target_link_executorch(swipe_jni)
endif()

# ===========================================================================
# End-to-end benchmark (requires ExecuTorch)
# ===========================================================================

if(HAVE_EXECUTORCH AND ETI_IS_TOP_LEVEL)
    add_executable(benchmark_e2e benchmark/benchmark_e2e.cpp)
    target_link_libraries(benchmark_e2e PRIVATE swipe_decoder)
    target_link_executorch(benchmark_e2e)

    add_executable(eval_accuracy targets/eval_accuracy.cpp)
    target_link_libraries(eval_accuracy PRIVATE swipe_decoder)
    target_link_executorch(eval_accuracy)

    add_executable(infer_jsonl targets/infer_jsonl.cpp)
    target_link_libraries(infer_jsonl PRIVATE swipe_decoder)
    target_link_executorch(infer_jsonl)
    target_include_directories(infer_jsonl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/executorch/third-party/json/include)
endif()

# ===========================================================================
# Native tests
# ===========================================================================

if(ETI_IS_TOP_LEVEL AND NOT ANDROID)
    add_executable(test_beam_search tests/test_beam_search.cpp)
    target_link_libraries(test_beam_search PRIVATE swipe_decoder)

    add_executable(test_beam_search_pruning tests/test_beam_search_pruning.cpp)
    target_link_libraries(test_beam_search_pruning PRIVATE swipe_decoder)

    add_executable(test_vocab_hash tests/test_vocab_hash.cpp)
    target_include_directories(test_vocab_hash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    if(HAVE_EXECUTORCH)
        add_executable(test_engine tests/test_engine.cpp)
        target_link_libraries(test_engine PRIVATE swipe_decoder)
        target_link_executorch(test_engine)

        add_executable(test_lm_vocab tests/test_lm_vocab.cpp)
        target_link_libraries(test_lm_vocab PRIVATE swipe_decoder)
        target_link_executorch(test_lm_vocab)

        # Contract test: every shipped metadata.json must parse cleanly.
        # Gates releases against silently-broken universal-swipe exports.
        add_executable(test_metadata tests/test_metadata.cpp src/model_metadata.cpp)
        target_include_directories(test_metadata PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_compile_definitions(test_metadata PRIVATE
            SWIPE_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ExecuTorch support: ${HAVE_EXECUTORCH}")
if(ANDROID)
    message(STATUS "Android ABI: ${ANDROID_ABI}")
endif()


# ===========================================================================
# Misc
# ===========================================================================

# Source - https://stackoverflow.com/a/47801116
# Posted by Martin R., modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-16, License - CC BY-SA 3.0
function(make_includable input_file output_file)
    file(READ ${input_file} content)
    set(delim "for_c++_include")
    set(content "R\"${delim}(\n${content})${delim}\"")
    file(WRITE ${output_file} "${content}")
endfunction(make_includable)

make_includable(models/scoring.json include/generated/scoring.json)
