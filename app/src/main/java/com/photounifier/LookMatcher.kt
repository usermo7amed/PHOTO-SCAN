package com.photounifier

import kotlin.math.max

/** The few numbers that describe how a photo looks: lightness mean/spread and the average colour. */
data class LookStats(val lMean: Double, val lStd: Double, val aMean: Double, val bMean: Double)

/**
 * Pulls finished photos towards one common look, so a batch shot under different light ends up
 * consistent. Pure arithmetic on [LookStats]; Enhancer applies the results with OpenCV.
 * The limits keep the match gentle: it moves a photo part of the way, never remakes it.
 */
object LookMatcher {
    const val L_SHIFT_MAX = 30.0
    const val STD_MIN = 0.75
    const val STD_MAX = 1.35
    const val AB_SHIFT_MAX = 10.0

    /** The middle look of a batch: the median of every number, so one odd photo does not drag the rest. */
    fun median(all: List<LookStats>): LookStats? {
        if (all.isEmpty()) return null
        fun middle(values: List<Double>): Double {
            val sorted = values.sorted()
            val mid = sorted.size / 2
            return if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2.0
        }
        return LookStats(
            lMean = middle(all.map { it.lMean }),
            lStd = middle(all.map { it.lStd }),
            aMean = middle(all.map { it.aMean }),
            bMean = middle(all.map { it.bMean })
        )
    }

    /** 256-entry lightness lookup moving a photo from [own] towards [target] by [strength] (0..1). */
    fun lightnessTable(own: LookStats, target: LookStats, strength: Double): ByteArray {
        val s = strength.coerceIn(0.0, 1.0)
        val ratio = (target.lStd / max(own.lStd, 1.0)).coerceIn(STD_MIN, STD_MAX)
        val scale = 1.0 + (ratio - 1.0) * s
        val shift = (target.lMean - own.lMean).coerceIn(-L_SHIFT_MAX, L_SHIFT_MAX) * s
        return ByteArray(256) { value ->
            ((value - own.lMean) * scale + own.lMean + shift + 0.5).toInt().coerceIn(0, 255).toByte()
        }
    }

    fun shiftA(own: LookStats, target: LookStats, strength: Double): Double =
        (target.aMean - own.aMean).coerceIn(-AB_SHIFT_MAX, AB_SHIFT_MAX) * strength.coerceIn(0.0, 1.0)

    fun shiftB(own: LookStats, target: LookStats, strength: Double): Double =
        (target.bMean - own.bMean).coerceIn(-AB_SHIFT_MAX, AB_SHIFT_MAX) * strength.coerceIn(0.0, 1.0)
}
