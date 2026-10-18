package com.photounifier

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PageMathTest {
    @Test fun ordersShuffledCornersClockwiseFromTopLeft() {
        val shuffled = doubleArrayOf(90.0, 90.0, 10.0, 12.0, 8.0, 88.0, 92.0, 10.0)
        assertArrayEquals(
            doubleArrayOf(10.0, 12.0, 92.0, 10.0, 90.0, 90.0, 8.0, 88.0),
            PageMath.orderCorners(shuffled), 1e-9
        )
    }

    @Test fun ordersAPageTurnedByTwentyDegrees() {
        val turned = doubleArrayOf(75.0, 95.0, 30.0, 10.0, 10.0, 75.0, 95.0, 30.0)
        assertArrayEquals(
            doubleArrayOf(30.0, 10.0, 95.0, 30.0, 75.0, 95.0, 10.0, 75.0),
            PageMath.orderCorners(turned), 1e-9
        )
    }

    @Test fun outputSizeUsesTheLongerSides() {
        val quad = doubleArrayOf(0.0, 0.0, 100.0, 0.0, 110.0, 200.0, 0.0, 190.0)
        val (w, h) = PageMath.outputSize(quad, 4096)
        assertEquals(110, w)
        assertEquals(200, h)
    }

    @Test fun outputSizeIsCapped() {
        val (w, _) = PageMath.outputSize(doubleArrayOf(0.0, 0.0, 9000.0, 0.0, 9000.0, 100.0, 0.0, 100.0), 4096)
        assertEquals(4096, w)
    }

    @Test fun aspectRatioMustBePageLike() {
        assertTrue(PageMath.aspectOk(doubleArrayOf(0.0, 0.0, 70.0, 0.0, 70.0, 100.0, 0.0, 100.0)))
        assertFalse(PageMath.aspectOk(doubleArrayOf(0.0, 0.0, 400.0, 0.0, 400.0, 20.0, 0.0, 20.0)))
    }

    @Test fun theWholeFrameIsNotAPage() {
        assertTrue(PageMath.coversFrame(doubleArrayOf(1.0, 1.0, 99.0, 0.0, 99.0, 99.0, 0.0, 99.0), 100, 100))
        assertFalse(PageMath.coversFrame(doubleArrayOf(20.0, 20.0, 80.0, 20.0, 80.0, 80.0, 20.0, 80.0), 100, 100))
    }

    @Test fun edgeSupportCountsOutlinePointsOnEdges() {
        val w = 100; val h = 100
        val edges = ByteArray(w * h)
        for (i in 20..80) { // draw the rectangle outline
            edges[20 * w + i] = 255.toByte(); edges[80 * w + i] = 255.toByte()
            edges[i * w + 20] = 255.toByte(); edges[i * w + 80] = 255.toByte()
        }
        val quad = doubleArrayOf(20.0, 20.0, 80.0, 20.0, 80.0, 80.0, 20.0, 80.0)
        assertEquals(1.0, PageMath.edgeSupport(quad, edges, w, h), 1e-9)
        assertEquals(0.0, PageMath.edgeSupport(quad, ByteArray(w * h), w, h), 1e-9)
    }

    @Test fun foregroundIsWhatDiffersFromTheBorder() {
        val w = 60; val h = 60
        val lab = ByteArray(w * h * 3)
        for (y in 0 until h) for (x in 0 until w) {
            val inside = x in 15..44 && y in 15..44
            val i = (y * w + x) * 3
            lab[i] = (if (inside) 200 else 90).toByte()
            lab[i + 1] = 128.toByte()
            lab[i + 2] = (if (inside) 130 else 150).toByte()
        }
        val mask = PageMath.foreground(lab, w, h)
        assertEquals(255, mask[30 * w + 30].toInt() and 0xFF)
        assertEquals(0, mask[2 * w + 2].toInt() and 0xFF)
    }
}
