package com.photounifier

import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * What a photo needs, measured on a small copy of it. Plain Kotlin on byte planes, so the
 * measuring and the decisions built on it can be unit-tested without OpenCV.
 */
data class ImageMetrics(
    /** Mean lightness, 0..255. */
    val mean: Double,
    val std: Double,
    /** Share of pixels with lightness >= [BRIGHT_LEVEL]. */
    val bright: Double,
    /** Share of pixels with lightness <= [DARK_LEVEL]. */
    val dark: Double,
    /** Mean HSV saturation, 0..1. */
    val saturation: Double,
    /** Share of near-neutral midtone pixels (walls, paper, grey surfaces). */
    val neutral: Double,
    /** The chroma limit that picked those pixels. */
    val neutralChroma: Double,
    /** How far the neutral pixels sit from grey, in Lab units: the colour cast. */
    val cast: Double
) {
    companion object {
        const val BRIGHT_LEVEL = 235
        const val DARK_LEVEL = 45
        const val NEUTRAL_CHROMA = 14.0
        const val NEUTRAL_CHROMA_MAX = 35.0
        const val MIN_NEUTRAL_FRACTION = 0.05
        private const val MIN_MIDTONE_PIXELS = 100

        /**
         * All four planes describe the same pixels, 8 bits each: Lab as OpenCV scales it
         * (a and b are offset by 128) and the HSV saturation plane.
         */
        fun measure(lightness: ByteArray, a: ByteArray, b: ByteArray, saturation: ByteArray): ImageMetrics {
            val n = lightness.size
            require(n > 0 && a.size == n && b.size == n && saturation.size == n) {
                "planes must have the same, non-zero size"
            }

            var sum = 0.0
            var sumSquares = 0.0
            var bright = 0
            var dark = 0
            var saturationSum = 0.0
            var midtones = 0
            val chromaHistogram = IntArray(256)
            for (i in 0 until n) {
                val l = lightness[i].toInt() and 0xFF
                sum += l
                sumSquares += l.toDouble() * l
                if (l >= BRIGHT_LEVEL) bright++
                if (l <= DARK_LEVEL) dark++
                saturationSum += saturation[i].toInt() and 0xFF
                if (isMidtone(l)) {
                    midtones++
                    chromaHistogram[min(chromaOf(a[i], b[i]).toInt(), 255)]++
                }
            }
            val mean = sum / n
            val std = sqrt(max(sumSquares / n - mean * mean, 0.0))

            val limit = neutralLimit(chromaHistogram, midtones)
            var neutralCount = 0
            var sumA = 0.0
            var sumB = 0.0
            for (i in 0 until n) {
                if (!isMidtone(lightness[i].toInt() and 0xFF)) continue
                if (chromaOf(a[i], b[i]) >= limit) continue
                neutralCount++
                sumA += (a[i].toInt() and 0xFF) - 128.0
                sumB += (b[i].toInt() and 0xFF) - 128.0
            }
            val neutral = neutralCount.toDouble() / n
            val cast = if (neutral >= MIN_NEUTRAL_FRACTION) hypot(sumA / neutralCount, sumB / neutralCount) else 0.0

            return ImageMetrics(
                mean = mean,
                std = std,
                bright = bright.toDouble() / n,
                dark = dark.toDouble() / n,
                saturation = saturationSum / n / 255.0,
                neutral = neutral,
                neutralChroma = limit,
                cast = cast
            )
        }

        private fun isMidtone(l: Int) = l > 40 && l < 235

        private fun chromaOf(a: Byte, b: Byte): Double =
            hypot((a.toInt() and 0xFF) - 128.0, (b.toInt() and 0xFF) - 128.0)

        /**
         * Chroma limit for "probably neutral" pixels: the quietest quarter of the midtones.
         * A fixed limit misses strong casts (a warm lamp pushes even a white wall past it);
         * the quartile follows the cast, and the cap keeps coloured objects out.
         */
        private fun neutralLimit(chromaHistogram: IntArray, midtones: Int): Double {
            if (midtones < MIN_MIDTONE_PIXELS) return NEUTRAL_CHROMA
            val target = midtones * 0.25
            var seen = 0
            for (value in chromaHistogram.indices) {
                seen += chromaHistogram[value]
                // +1: the bucket holds chroma in [value, value+1), and the test below is strict
                if (seen >= target) return (value + 1.0).coerceIn(NEUTRAL_CHROMA, NEUTRAL_CHROMA_MAX)
            }
            return NEUTRAL_CHROMA_MAX
        }
    }
}
