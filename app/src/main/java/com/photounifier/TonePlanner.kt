package com.photounifier

import kotlin.math.max
import kotlin.math.min

/** What to apply to one photo, and how strongly. A strength of 0 (or false) means "leave it alone". */
data class TonePlan(
    val whiteBalance: Boolean,
    /** Chroma limit that picks the neutral pixels white balance is estimated from. */
    val neutralChroma: Double,
    /** Lightness (0..1) above which bright regions are pulled down. */
    val shoulder: Double,
    /** Lightness (0..1) below which dark regions are lifted. */
    val toe: Double,
    val alphaHigh: Double,
    val alphaLow: Double,
    val gainMin: Double,
    val gainMax: Double,
    val claheClip: Double,
    val vibrance: Double,
    val sharpening: Double
)

/**
 * Measure, then decide: each correction runs only when the image shows the problem it fixes,
 * and its strength grows with the problem up to a ceiling set by the kind of photo.
 * The numbers match tools/prototype/enhance.py: tune them there first, then copy them here.
 */
object TonePlanner {
    private const val BRIGHT_TRIGGER = 0.03
    private const val DARK_TRIGGER = 0.05
    private const val MEAN_DARK_TRIGGER = 90.0
    private const val CONTRAST_STD = 60.0
    private const val WB_TRIGGER = 3.0
    private const val SATURATION_OK = 0.35
    private const val VIBRANCE_GAIN = 1.6
    private const val VIBRANCE_MIN = 0.03

    /** Ceilings for one kind of photo. */
    private class Route(
        val shoulder: Double,
        val toe: Double,
        val high: Double,
        val low: Double,
        val gainMin: Double,
        val gainMax: Double,
        val clahe: Double,
        val vibranceMax: Double,
        val sharpening: Double,
        val whiteBalance: Boolean
    )

    private val interior = Route(0.80, 0.40, 0.50, 0.55, 0.62, 2.25, 2.0, 0.35, 0.24, true)
    private val exterior = Route(0.72, 0.32, 0.55, 0.45, 0.55, 1.95, 1.8, 0.50, 0.22, false)
    private val general = Route(0.78, 0.36, 0.45, 0.50, 0.65, 2.20, 2.0, 0.30, 0.25, true)
    // Poster colours are intentional: no white balance and no vibrance.
    private val poster = Route(0.85, 0.30, 0.30, 0.30, 0.78, 1.60, 1.45, 0.0, 0.16, false)

    private fun routeFor(kind: ImageKind): Route = when (kind) {
        ImageKind.INTERIOR -> interior
        ImageKind.EXTERIOR -> exterior
        ImageKind.POSTER -> poster
        ImageKind.DOCUMENT, ImageKind.GENERAL -> general
    }

    fun plan(kind: ImageKind, metrics: ImageMetrics): TonePlan {
        val route = routeFor(kind)

        val alphaHigh =
            if (metrics.bright < BRIGHT_TRIGGER) 0.0
            else (0.2 + metrics.bright).coerceIn(0.2, route.high)

        val needsLift = metrics.dark > DARK_TRIGGER || metrics.mean < MEAN_DARK_TRIGGER
        val alphaLow =
            if (!needsLift) 0.0
            else (0.15 + metrics.dark + max(0.0, 100.0 - metrics.mean) / 200.0).coerceIn(0.15, route.low)

        val clahe =
            if (metrics.std >= CONTRAST_STD) 0.0
            else (0.8 + (CONTRAST_STD - metrics.std) / 10.0).coerceIn(1.0, route.clahe)

        val vibrance = min(max(0.0, SATURATION_OK - metrics.saturation) * VIBRANCE_GAIN, route.vibranceMax)

        val whiteBalance = route.whiteBalance &&
            metrics.neutral >= ImageMetrics.MIN_NEUTRAL_FRACTION &&
            metrics.cast >= WB_TRIGGER

        return TonePlan(
            whiteBalance = whiteBalance,
            neutralChroma = metrics.neutralChroma,
            shoulder = route.shoulder,
            toe = route.toe,
            alphaHigh = alphaHigh,
            alphaLow = alphaLow,
            gainMin = route.gainMin,
            gainMax = route.gainMax,
            claheClip = clahe,
            vibrance = if (vibrance >= VIBRANCE_MIN) vibrance else 0.0,
            sharpening = route.sharpening
        )
    }

    /** Short Arabic labels for what the plan will do, for the on-screen decision log. */
    fun describe(plan: TonePlan): List<String> {
        val steps = ArrayList<String>()
        if (plan.whiteBalance) steps.add("ضبط توازن اللون")
        if (plan.alphaHigh > 0) steps.add("خفض السطوع العالي")
        if (plan.alphaLow > 0) steps.add("رفع الظلال")
        if (plan.claheClip > 0) steps.add("تقوية التباين")
        if (plan.vibrance > 0) steps.add("تحسين الألوان")
        return steps
    }

    /**
     * Lookup table for the HSV saturation channel: weakly coloured pixels gain most and vivid
     * ones barely move, so a washed-out photo recovers colour without cartoon-bright reds.
     */
    fun vibranceTable(amount: Double): ByteArray = ByteArray(256) { value ->
        val s = value / 255.0
        (s * (1.0 + amount * (1.0 - s)) * 255.0 + 0.5).toInt().coerceIn(0, 255).toByte()
    }
}
