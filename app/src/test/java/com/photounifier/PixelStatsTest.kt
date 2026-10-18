package com.photounifier

import org.junit.Assert.assertEquals
import org.junit.Test

class PixelStatsTest {
    @Test fun percentileOfBytes() {
        val values = ByteArray(101) { it.toByte() }
        assertEquals(50, PixelStats.median(values))
        assertEquals(95, PixelStats.percentile(values, 95.0))
        assertEquals(0, PixelStats.percentile(values, 0.0))
        assertEquals(100, PixelStats.percentile(values, 100.0))
    }

    @Test fun percentileOfFloats() {
        val values = FloatArray(101) { it.toFloat() }
        assertEquals(95f, PixelStats.percentile(values, 95.0), 0f)
    }

    @Test fun percentileHandlesValuesAbove127() {
        assertEquals(250, PixelStats.median(ByteArray(5) { 250.toByte() }))
    }
}
