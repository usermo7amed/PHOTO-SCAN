package com.photounifier

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LookMatcherTest {
    private val own = LookStats(lMean = 100.0, lStd = 40.0, aMean = 0.0, bMean = 10.0)
    private val target = LookStats(lMean = 130.0, lStd = 50.0, aMean = 2.0, bMean = 0.0)
    private fun at(table: ByteArray, i: Int) = table[i].toInt() and 0xFF

    @Test fun medianIgnoresTheOddOneOut() {
        val looks = listOf(10.0, 12.0, 11.0, 200.0).map { LookStats(it, 40.0, 0.0, 0.0) }
        assertEquals(11.5, LookMatcher.median(looks)!!.lMean, 1e-9)
    }

    @Test fun medianOfNothingIsNull() = assertNull(LookMatcher.median(emptyList()))

    @Test fun zeroStrengthLeavesLightnessAlone() {
        val table = LookMatcher.lightnessTable(own, target, 0.0)
        assertTrue((0..255).all { at(table, it) == it })
    }

    @Test fun fullStrengthMovesTheMeanTowardsTheTarget() {
        val table = LookMatcher.lightnessTable(own, target, 1.0)
        assertTrue(at(table, 100) in 125..135)
    }

    @Test fun halfStrengthMovesHalfWay() {
        val table = LookMatcher.lightnessTable(own, target, 0.5)
        assertTrue(at(table, 100) in 110..120)
    }

    @Test fun tableIsMonotonic() {
        val table = LookMatcher.lightnessTable(own, target, 1.0)
        assertTrue((1..255).all { at(table, it) >= at(table, it - 1) })
    }

    @Test fun shiftsAreCapped() {
        val far = LookStats(lMean = 250.0, lStd = 200.0, aMean = 60.0, bMean = -60.0)
        val table = LookMatcher.lightnessTable(own, far, 1.0)
        assertTrue(at(table, 100) <= (100 + LookMatcher.L_SHIFT_MAX * LookMatcher.STD_MAX + 1).toInt() + 10)
        assertEquals(LookMatcher.AB_SHIFT_MAX, LookMatcher.shiftA(own, far, 1.0), 1e-9)
        assertEquals(-LookMatcher.AB_SHIFT_MAX, LookMatcher.shiftB(own, far, 1.0), 1e-9)
    }
}
