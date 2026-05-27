package org.futo.ml.inference

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.AfterClass
import org.junit.Assert.*
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented tests that run on a connected Android device.
 *
 * Requires `make push` (or `make push-demo`) to stage assets on-device:
 *   encoder model_fp32.pte + metadata.json, vocabs/en.combined, layouts/en_qwerty.json,
 *   and optionally context_lm.pte for LM-enabled tests.
 */
@RunWith(AndroidJUnit4::class)
class SwipeDecoderTest {

    private lateinit var decoder: SwipeDecoder

    companion object {
        private const val BASE_DIR = "/data/local/tmp/ml-inference"
        private const val VOCAB_PATH = "$BASE_DIR/en.combined"
        private const val LAYOUT_PATH = "$BASE_DIR/en_qwerty.json"
        private const val LM_MODEL_PATH = "$BASE_DIR/hungry_jellyfish/context_lm.pte"
        private const val LM_VOCAB_PATH = "$BASE_DIR/hungry_jellyfish/vocab.txt"
        private const val DECODER_PATH = "$BASE_DIR/magic_macaw/model_fp32.pte"
        private const val ENCODER_PATH = "$BASE_DIR/model_fp32.pte"

        // en_qwerty letters + coordinates (must match models/layouts/en_qwerty.json).
        private const val EN_LETTERS = "abcdefghijklmnopqrstuvwxyz"
        private val EN_CX = floatArrayOf(
            0.1f, 0.6f, 0.4f, 0.3f, 0.25f, 0.4f, 0.5f, 0.6f, 0.75f, 0.7f,
            0.8f, 0.9f, 0.8f, 0.7f, 0.85f, 0.95f, 0.05f, 0.35f, 0.2f, 0.45f,
            0.65f, 0.5f, 0.15f, 0.3f, 0.55f, 0.2f
        )
        private val EN_CY = floatArrayOf(
            0.5f, 0.833f, 0.833f, 0.5f, 0.167f, 0.5f, 0.5f, 0.5f, 0.167f, 0.5f,
            0.5f, 0.5f, 0.833f, 0.833f, 0.167f, 0.167f, 0.167f, 0.167f, 0.5f, 0.167f,
            0.167f, 0.833f, 0.167f, 0.833f, 0.167f, 0.833f
        )

        // Shared encoder-only decoder reused across non-destructive tests.
        // The constructor brackets the load with big-cluster affinity internally.
        private lateinit var SHARED: SwipeDecoder

        @BeforeClass
        @JvmStatic
        fun initShared() {
            SHARED = SwipeDecoder(
                encoderPath = ENCODER_PATH,
                vocabPath = VOCAB_PATH,
                layoutPath = LAYOUT_PATH,
                threads = 1,
                beamWidth = 100
            )
        }

        @AfterClass
        @JvmStatic
        fun closeShared() {
            SHARED.close()
        }
    }

    private fun bezierSwipe(n: Int = 80): Triple<FloatArray, FloatArray, FloatArray> {
        val x = FloatArray(n) { i -> 0.2f + 0.6f * i.toFloat() / (n - 1) }
        val y = FloatArray(n) { i -> 0.5f + 0.1f * Math.sin(i.toDouble() * Math.PI / (n - 1)).toFloat() }
        val t = FloatArray(n) { i -> i.toFloat() * 8f }
        return Triple(x, y, t)
    }

    /**
     * Lightweight per-test reset
     */
    @Before
    fun setUp() {
        decoder = SHARED
        decoder.clearContext()
    }

    @After
    fun tearDown() {
        // Shared decoder is closed in @AfterClass; nothing to do per-test.
    }

    /** Restore SHARED to encoder-only state. Call from tests that load a decoder. */
    private fun unloadSharedDecoder() {
        if (SHARED.hasDecoder()) {
            SHARED.setMode(decoderPath = "")
        }
    }

    @Test
    fun testInitialization() {
        // If we got here, init succeeded (constructor throws on failure)
        assertTrue(true)
    }

    @Test
    fun testRecognizeSyntheticPath() {
        // Generate a simple synthetic swipe path
        val n = 80
        val x = FloatArray(n) { i -> 0.2f + 0.6f * i.toFloat() / (n - 1) }
        val y = FloatArray(n) { i -> 0.5f + 0.1f * Math.sin(i.toDouble() * Math.PI / (n - 1)).toFloat() }
        val t = FloatArray(n) { i -> i.toFloat() * 8f }

        val results = decoder.recognize(x, y, t, topK = 4)

        assertFalse("Should return at least one result", results.isEmpty())
        assertTrue("Should return at most 4 results", results.size <= 4)

        // Each result should be a non-empty word
        for (result in results) {
            assertTrue("Word should be non-empty", result.word.isNotEmpty())
            assertTrue("Word should be lowercase a-z",
                result.word.all { it in 'a'..'z' })
        }

        // Results should be sorted by score descending
        for (i in 0 until results.size - 1) {
            assertTrue("Results should be sorted by score",
                results[i].score >= results[i + 1].score)
        }
    }

    @Test
    fun testPredictNextWithoutLM() {
        // predictNext() returns empty when LM not configured (no lmModelPath at init)
        decoder.addContext("hello")
        decoder.addContext("world")
        val results = decoder.predictNext(topK = 5)
        assertTrue("predictNext should return empty without LM", results.isEmpty())
    }

    @Test(expected = IllegalArgumentException::class)
    fun testRecognizeEmptyPath() {
        decoder.recognize(FloatArray(0), FloatArray(0), FloatArray(0), topK = 4)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testRecognizeMismatchedArrays() {
        decoder.recognize(FloatArray(10), FloatArray(5), FloatArray(10), topK = 4)
    }

    @Test
    fun testDoubleClose() {
        // Destructive: use a local decoder so we don't break the shared one.
        val local = SwipeDecoder(
            encoderPath = ENCODER_PATH,
            vocabPath = VOCAB_PATH,
            layoutPath = LAYOUT_PATH,
            threads = 1,
            beamWidth = 100
        )
        local.close()
        local.close()
    }

    @Test(expected = IllegalStateException::class)
    fun testUseAfterClose() {
        // Destructive: use a local decoder so we don't break the shared one.
        val local = SwipeDecoder(
            encoderPath = ENCODER_PATH,
            vocabPath = VOCAB_PATH,
            layoutPath = LAYOUT_PATH,
            threads = 1,
            beamWidth = 100
        )
        local.close()
        val n = 10
        val x = FloatArray(n) { 0.5f }
        val y = FloatArray(n) { 0.5f }
        val t = FloatArray(n) { i -> i * 16.67f }
        local.recognize(x, y, t, topK = 4)
    }

    @Test
    fun testContextManagement() {
        // Verify add/set/clear don't crash; state is managed correctly
        decoder.addContext("hello")
        decoder.addContext("world")
        decoder.setContext(listOf("the", "quick"))
        decoder.clearContext()
        // Should still be able to recognize after context changes
        val n = 60
        val x = FloatArray(n) { i -> i.toFloat() / (n - 1) }
        val y = FloatArray(n) { 0.5f }
        val t = FloatArray(n) { i -> i * 16.67f }
        val results = decoder.recognize(x, y, t, topK = 4)
        assertFalse("Should return results after context changes", results.isEmpty())
    }

    @Test
    fun testAddContextSkipsEmpty() {
        // addContext("") should be a no-op
        decoder.addContext("")
        decoder.addContext("hello")
        // No crash, and if we had a way to inspect context size we'd check it's 1
        assertTrue(true)
    }

    @Test
    fun testLatencyAndMemory() {
        // Destructive: measures init + steady-state RSS from scratch, so it needs
        // its own decoder independent of the shared fixture.
        val rssBaseline = getRssMb()

        val local = SwipeDecoder(
            encoderPath = ENCODER_PATH,
            vocabPath = VOCAB_PATH,
            layoutPath = LAYOUT_PATH,
            threads = 1,
            beamWidth = 100
        )

        val rssAfterInit = getRssMb()

        val n = 80
        val x = FloatArray(n) { i -> 0.2f + 0.6f * i.toFloat() / (n - 1) }
        val y = FloatArray(n) { i -> 0.5f + 0.1f * Math.sin(i.toDouble() * Math.PI / (n - 1)).toFloat() }
        val t = FloatArray(n) { i -> i.toFloat() * 8f }

        repeat(50) { local.recognize(x, y, t) }

        val rssAfterWarmup = getRssMb()

        // Measure
        val times = mutableListOf<Long>()
        repeat(500) {
            val start = System.nanoTime()
            local.recognize(x, y, t)
            val end = System.nanoTime()
            times.add(end - start)
        }

        val rssAfterBench = getRssMb()

        times.sort()
        val p50Us = times[times.size / 2] / 1000.0
        val p99Us = times[(times.size * 0.99).toInt()] / 1000.0
        val maxUs = times.last() / 1000.0
        val top5 = times.takeLast(5).map { it / 1000.0 }
        val deltaInit = rssAfterInit - rssBaseline
        val deltaSteady = rssAfterBench - rssBaseline
        println("======== BENCHMARK RESULTS (in-app, threads=1) ========")
        println("Runs: ${times.size}, Warmup: 50")
        println("P50 = %.3f ms".format(p50Us / 1000.0))
        println("P99 = %.3f ms".format(p99Us / 1000.0))
        println("Max = %.3f ms".format(maxUs / 1000.0))
        println("Top 5 slowest (us): $top5")
        println("RSS baseline (before init): %.1f MB".format(rssBaseline))
        println("RSS after init:             %.1f MB".format(rssAfterInit))
        println("RSS after warmup:           %.1f MB".format(rssAfterWarmup))
        println("RSS after bench:            %.1f MB".format(rssAfterBench))
        println("Delta (init):               %.1f MB".format(deltaInit))
        println("Delta (steady-state):       %.1f MB".format(deltaSteady))
        println("======================================================")
        // Universal encoder on little cores must stay inside the 60 Hz frame budget.
        assertTrue("P99 latency should be < 16.67ms (60 Hz), was ${p99Us}us", p99Us < 16670.0)
        local.close()
    }

    @Test
    fun testLatencyWithLM() {
        // Skip if LM model not present
        if (!java.io.File(LM_MODEL_PATH).exists()) {
            println("SKIP: LM model not found at $LM_MODEL_PATH")
            return
        }

        val lmDecoder = SwipeDecoder(
            encoderPath = ENCODER_PATH,
            vocabPath = VOCAB_PATH,
            layoutPath = LAYOUT_PATH,
            threads = 1,
            beamWidth = 100,
            lmModelPath = LM_MODEL_PATH,
            lmVocabPath = LM_VOCAB_PATH,
            lmAlpha = 0.56f
        )

        lmDecoder.setContext(listOf("the", "quick", "brown"))

        val n = 80
        val x = FloatArray(n) { i -> 0.2f + 0.6f * i.toFloat() / (n - 1) }
        val y = FloatArray(n) { i -> 0.5f + 0.1f * Math.sin(i.toDouble() * Math.PI / (n - 1)).toFloat() }
        val t = FloatArray(n) { i -> i.toFloat() * 8f }

        // Warmup
        repeat(50) { lmDecoder.recognize(x, y, t) }

        // Measure recognize (swipe + LM reranking)
        val recognizeTimes = mutableListOf<Long>()
        repeat(500) {
            val start = System.nanoTime()
            lmDecoder.recognize(x, y, t)
            val end = System.nanoTime()
            recognizeTimes.add(end - start)
        }

        // Measure predictNext
        val predictTimes = mutableListOf<Long>()
        repeat(500) {
            val start = System.nanoTime()
            lmDecoder.predictNext(topK = 10)
            val end = System.nanoTime()
            predictTimes.add(end - start)
        }

        lmDecoder.close()

        recognizeTimes.sort()
        predictTimes.sort()

        val recP50 = recognizeTimes[recognizeTimes.size / 2] / 1000.0
        val recP99 = recognizeTimes[(recognizeTimes.size * 0.99).toInt()] / 1000.0
        val predP50 = predictTimes[predictTimes.size / 2] / 1000.0
        val predP99 = predictTimes[(predictTimes.size * 0.99).toInt()] / 1000.0

        println("======== BENCHMARK RESULTS (in-app + LM, threads=1) ========")
        println("Recognize (swipe + LM reranking):")
        println("  P50 = %.3f ms".format(recP50 / 1000.0))
        println("  P99 = %.3f ms".format(recP99 / 1000.0))
        println("PredictNext (context LM only):")
        println("  P50 = %.3f ms".format(predP50 / 1000.0))
        println("  P99 = %.3f ms".format(predP99 / 1000.0))
        println("=============================================================")

        // Recognize + LM rerank runs once per swipe (not per frame), so
        // we allow modestly above the 60 Hz budget.
        assertTrue("Recognize P99 should be < 20ms, was ${recP99/1000.0}ms", recP99 < 20000.0)
        assertTrue("PredictNext P99 should be < 10ms, was ${predP99/1000.0}ms", predP99 < 10000.0)
    }

    private fun getRssMb(): Double {
        return try {
            val status = java.io.File("/proc/self/status").readText()
            val match = Regex("VmRSS:\\s+(\\d+)\\s+kB").find(status)
            (match?.groupValues?.get(1)?.toDouble() ?: 0.0) / 1024.0
        } catch (e: Exception) {
            0.0
        }
    }

    // ------------------------------------------------------------------
    // New coverage: runtime swap, context, timing, hasDecoder.
    // ------------------------------------------------------------------

    @Test
    fun testHasDecoderInitiallyFalse() {
        assertFalse("encoder-only ctor should leave hasDecoder() false", decoder.hasDecoder())
    }

    @Test
    fun testSetDecoderNullPathNullCycle() {
        if (!java.io.File(DECODER_PATH).exists()) {
            println("SKIP: decoder not found at $DECODER_PATH")
            return
        }
        try {
            val (x, y, t) = bezierSwipe()

            // 1. Start in encoder-only; recognize works.
            assertFalse(decoder.hasDecoder())
            val beforeLoad = decoder.recognize(x, y, t)
            assertFalse("recognize before decoder load empty", beforeLoad.isEmpty())

            // 2. Load decoder.
            assertTrue(
                "setMode(decoderPath=path) should succeed",
                decoder.setMode(
                    letters = EN_LETTERS, cx = EN_CX, cy = EN_CY,
                    decoderPath = DECODER_PATH
                )
            )
            assertTrue(decoder.hasDecoder())
            val withDecoder = decoder.recognize(x, y, t)
            assertFalse("recognize with decoder empty", withDecoder.isEmpty())

            // 3. Unload via empty path; still recognizes.
            assertTrue(
                "setMode(decoderPath=\"\") unload should succeed",
                decoder.setMode(decoderPath = "")
            )
            assertFalse(decoder.hasDecoder())
            val afterUnload = decoder.recognize(x, y, t)
            assertFalse("recognize after unload empty", afterUnload.isEmpty())

            // 4. Unloading the already-unloaded decoder should remain a no-op.
            assertTrue(decoder.setMode(decoderPath = ""))
            assertFalse(decoder.hasDecoder())
        } finally {
            unloadSharedDecoder()
        }
    }

    @Test
    fun testSetLayoutRoundtrip() {
        val (x, y, t) = bezierSwipe()
        val baseline = decoder.recognize(x, y, t, topK = 4)
        assertFalse(baseline.isEmpty())

        // Same letters + coords: setMode should detect the layout is identical
        // and no-op, preserving top-1 word + score bit-exactly.
        assertTrue(decoder.setMode(letters = EN_LETTERS, cx = EN_CX, cy = EN_CY))
        val after = decoder.recognize(x, y, t, topK = 4)
        assertEquals(baseline.size, after.size)
        assertEquals(
            "top-1 word changed after same-layout setMode",
            baseline[0].word, after[0].word
        )
        assertEquals(
            "top-1 score drifted after same-layout setMode",
            baseline[0].score, after[0].score, 1e-3f
        )
    }

    @Test
    fun testLastTimingStagesPopulated() {
        val (x, y, t) = bezierSwipe()
        decoder.recognize(x, y, t)
        val tm = decoder.lastTiming()

        assertTrue("resample_us missing",  tm.resampleUs >= 0f)
        assertTrue("encoder_us missing",   tm.encoderUs > 0f)
        assertTrue("beam_us missing",      tm.beamUs > 0f)
        // decoder_us / lm_us brackets run even when the stage is skipped, so they
        // capture a couple of Clock::now() calls worth of overhead (< 1 µs) rather
        // than exact zero. Bound generously.
        assertTrue("decoder_us should be ~0 in encoder-only, was ${tm.decoderUs}", tm.decoderUs < 50f)
        assertTrue("lm_us should be ~0 without context+LM, was ${tm.lmUs}",        tm.lmUs < 50f)
        assertTrue(
            "total_us < sum of stages",
            tm.totalUs + 1e-3f >= tm.resampleUs + tm.encoderUs + tm.beamUs
        )
    }

    @Test
    fun testSetContextVsAddContextConsistent() {
        if (!java.io.File(LM_MODEL_PATH).exists()) {
            println("SKIP: LM model not found at $LM_MODEL_PATH")
            return
        }

        val lm = SwipeDecoder(
            encoderPath = ENCODER_PATH,
            vocabPath = VOCAB_PATH,
            layoutPath = LAYOUT_PATH,
            threads = 1,
            beamWidth = 100,
            lmModelPath = LM_MODEL_PATH,
            lmVocabPath = VOCAB_PATH,
            lmAlpha = 0.5f,
        )
        try {
            val words = listOf("the", "quick", "brown")

            lm.setContext(words)
            val viaSet = lm.predictNext(topK = 10).map { it.word to it.score }

            lm.clearContext()
            words.forEach { lm.addContext(it) }
            val viaAdd = lm.predictNext(topK = 10).map { it.word to it.score }

            assertEquals(
                "setContext and repeated addContext should produce the same prediction",
                viaSet, viaAdd
            )
        } finally {
            lm.close()
        }
    }

    @Test
    fun testContextChangesPredictNext() {
        if (!java.io.File(LM_MODEL_PATH).exists()) {
            println("SKIP: LM model not found at $LM_MODEL_PATH")
            return
        }

        val lm = SwipeDecoder(
            encoderPath = ENCODER_PATH,
            vocabPath = VOCAB_PATH,
            layoutPath = LAYOUT_PATH,
            threads = 1,
            beamWidth = 100,
            lmModelPath = LM_MODEL_PATH,
            lmVocabPath = VOCAB_PATH,
            lmAlpha = 0.5f,
        )
        try {
            lm.setContext(listOf("we", "are"))
            val a = lm.predictNext(topK = 10)
            lm.setContext(listOf("he", "says"))
            val b = lm.predictNext(topK = 10)

            assertFalse("predictNext empty with context A", a.isEmpty())
            assertFalse("predictNext empty with context B", b.isEmpty())
            assertNotEquals(
                "predictNext ignored context switch",
                a.map { it.word }, b.map { it.word }
            )
        } finally {
            lm.close()
        }
    }

    @Test
    fun testDecoderTimingPopulated() {
        if (!java.io.File(DECODER_PATH).exists()) {
            println("SKIP: decoder not found at $DECODER_PATH")
            return
        }
        try {
            assertTrue(decoder.setMode(
                letters = EN_LETTERS, cx = EN_CX, cy = EN_CY,
                decoderPath = DECODER_PATH
            ))
            val (x, y, t) = bezierSwipe()
            decoder.recognize(x, y, t)
            val tm = decoder.lastTiming()
            assertTrue("decoder_us not recorded", tm.decoderUs > 0f)
            assertTrue("encoder_us not recorded", tm.encoderUs > 0f)
        } finally {
            unloadSharedDecoder()
        }
    }

}
