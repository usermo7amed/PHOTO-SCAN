package com.photounifier

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageMetricsTest {
    private fun flat(n: Int, l: Int, a: Int = 128, b: Int = 128, s: Int = 0) = arrayOf(
        ByteArray(n) { l.toByte() }, ByteArray(n) { a.toByte() }, ByteArray(n) { b.toByte() }, ByteArray(n) { s.toByte() }
    )

    @Test fun measuresBurntAndDarkShares() {
        val n = 1000
        val l = ByteArray(n) { if (it < 300) 250.toByte() else if (it < 400) 10.toByte() else 120.toByte() }
        val metrics = ImageMetrics.measure(l, ByteArray(n) { 128.toByte() }, ByteArray(n) { 128.toByte() }, ByteArray(n))
        assertEquals(0.3, metrics.bright, 1e-9)
        assertEquals(0.1, metrics.dark, 1e-9)
    }

    @Test fun greyImageHasNoCast() {
        val (l, a, b, s) = flat(2000, 128)
        val metrics = ImageMetrics.measure(l, a, b, s)
        assertEquals(0.0, metrics.cast, 1e-9)
        assertEquals(1.0, metrics.neutral, 1e-9)
    }

    @Test fun warmWallShowsAsCast() {
        // a = 135, b = 150: a clearly yellow-red wall that is still the quietest part of the picture
        val (l, a, b, s) = flat(2000, 128, 135, 150)
        val metrics = ImageMetrics.measure(l, a, b, s)
        assertTrue("cast ${metrics.cast}", metrics.cast > 20.0)
    }

    @Test fun neutralLimitFollowsTheQuietestQuarterButIsCapped() {
        val n = 2000
        val a = ByteArray(n) { if (it % 2 == 0) 128.toByte() else 200.toByte() }
        val (l, _, b, s) = flat(n, 128)
        val metrics = ImageMetrics.measure(l, a, b, s)
        assertTrue(metrics.neutralChroma in ImageMetrics.NEUTRAL_CHROMA..ImageMetrics.NEUTRAL_CHROMA_MAX)
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsMismatchedPlanes() {
        ImageMetrics.measure(ByteArray(4), ByteArray(3), ByteArray(4), ByteArray(4))
    }
}
