package com.photounifier

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TonePlannerTest {
    private val good = ImageMetrics(
        mean = 125.0, std = 65.0, bright = 0.0, dark = 0.01, saturation = 0.45,
        neutral = 0.3, neutralChroma = 14.0, cast = 1.0
    )

    @Test fun aGoodPhotoIsLeftAlone() {
        val plan = TonePlanner.plan(ImageKind.GENERAL, good)
        assertEquals(0.0, plan.alphaHigh, 0.0)
        assertEquals(0.0, plan.alphaLow, 0.0)
        assertEquals(0.0, plan.claheClip, 0.0)
        assertEquals(0.0, plan.vibrance, 0.0)
        assertFalse(plan.whiteBalance)
        assertTrue(TonePlanner.describe(plan).isEmpty())
    }

    @Test fun burntSkyPullsBrightRegionsDownWithinTheCeiling() {
        val burnt = good.copy(bright = 0.26, mean = 225.0, std = 46.0)
        val plan = TonePlanner.plan(ImageKind.EXTERIOR, burnt)
        assertTrue(plan.alphaHigh > 0.4)
        assertTrue(plan.alphaHigh <= 0.55)
        assertEquals(0.0, plan.alphaLow, 0.0)
        assertTrue("تقوية التباين" in TonePlanner.describe(plan))
    }

    @Test fun darkPhotoLiftsShadowsButNeverPastTheCeiling() {
        val dark = good.copy(mean = 40.0, dark = 0.6)
        val plan = TonePlanner.plan(ImageKind.INTERIOR, dark)
        assertEquals(0.55, plan.alphaLow, 1e-9)
        assertEquals(0.0, plan.alphaHigh, 0.0)
    }

    @Test fun warmLightIsBalancedOnlyWhereTheRouteAllowsIt() {
        val warm = good.copy(cast = 20.0)
        assertTrue(TonePlanner.plan(ImageKind.INTERIOR, warm).whiteBalance)
        assertFalse(TonePlanner.plan(ImageKind.EXTERIOR, warm).whiteBalance)
        assertFalse(TonePlanner.plan(ImageKind.POSTER, warm).whiteBalance)
    }

    @Test fun tooFewNeutralPixelsMeansNoWhiteBalance() {
        assertFalse(TonePlanner.plan(ImageKind.INTERIOR, good.copy(cast = 20.0, neutral = 0.01)).whiteBalance)
    }

    @Test fun postersKeepTheirColours() {
        val washed = good.copy(saturation = 0.05)
        assertEquals(0.0, TonePlanner.plan(ImageKind.POSTER, washed).vibrance, 0.0)
        assertTrue(TonePlanner.plan(ImageKind.INTERIOR, washed).vibrance > 0.0)
    }

    @Test fun vibranceHelpsDullColoursMoreThanVividOnes() {
        val table = TonePlanner.vibranceTable(0.4)
        fun at(i: Int) = table[i].toInt() and 0xFF
        assertEquals(0, at(0))
        assertEquals(255, at(255))
        assertTrue(at(40) - 40 > at(220) - 220)
        assertTrue((1..255).all { at(it) >= at(it - 1) })
    }
}
