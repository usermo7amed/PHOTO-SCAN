package com.photounifier

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max

/**
 * One image in, one improved image out. It measures the image, plans only the corrections the image
 * needs (TonePlanner), and applies them. The numbers mirror tools/prototype/enhance.py.
 *
 * Everything works on RGB [Mat]s; Bitmap conversion lives in Processor.
 */
object Enhancer {
    class Request(
        /** The route chosen from the photo's labels, or by the person. */
        val kind: ImageKind,
        /** True when [kind] came from the labels, so a clearly visible page may still override it. */
        val auto: Boolean,
        /** Cut out and flatten the page/poster. False when the person already cropped by hand. */
        val rectify: Boolean,
        /** The common look to pull photos towards; null leaves each photo as it is. */
        val look: LookStats? = null,
        val lookStrength: Double = 0.0
    )

    class Result(
        /** Owned by the caller: release it when done. */
        val rgb: Mat,
        val kind: ImageKind,
        /** Arabic labels for what was done, for the decision log. */
        val steps: List<String>,
        /** The look of the finished image. */
        val look: LookStats
    ) : AutoCloseable {
        override fun close() = rgb.release()
    }

    private const val NEUTRAL_LOW = 41
    private const val NEUTRAL_HIGH = 234
    private const val WB_MIN = 0.85
    private const val WB_MAX = 1.2
    private const val POSTER_COLORFUL = 0.15

    private const val SHADOW_LIFT = 0.12
    private const val SHADOW_LIFT_RANGE = 0.25

    private const val PAPER_WHITE = 247.0
    private const val PAPER_GAIN_MAX = 2.0
    private const val PAPER_SCALE = 8
    private const val PAPER_PERCENTILE = 95.0
    private const val PAPER_FIRST_RATIO = 0.5
    private const val PAPER_REL_RATIO = 0.62
    private const val PAPER_PASSES = 2
    private const val PAPER_INK_GROW = 5.0
    private const val PAPER_SIGMA_DIV = 40.0
    private const val PAPER_WB_RATIO = 0.8
    private const val DOCUMENT_CLAHE = 1.2
    private const val DOCUMENT_SHARP = 0.12

    fun enhance(source: Mat, request: Request): Result = MatPool().use { pool ->
        val steps = ArrayList<String>()
        var kind = request.kind
        var work = source

        if (request.rectify) {
            if (kind == ImageKind.DOCUMENT || kind == ImageKind.POSTER) {
                PageFinder.find(pool, source, strict = false)?.let { found ->
                    work = PageFinder.warp(pool, source, found.corners)
                    steps.add("قص الصفحة وتعديل المنظور")
                }
            } else if (request.auto) {
                PageFinder.find(pool, source, strict = true)?.let { found ->
                    work = PageFinder.warp(pool, source, found.corners)
                    val (small, _) = PageFinder.analysisCopy(pool, work)
                    kind = if (colorfulShare(pool, small) >= POSTER_COLORFUL) ImageKind.POSTER else ImageKind.DOCUMENT
                    steps.add("ورقة واضحة في الصورة: قصّها وعدّل المنظور (${kind.title})")
                }
            }
        }

        val processed = if (kind == ImageKind.DOCUMENT) document(pool, work, steps)
        else photo(pool, work, kind, request, steps)

        val (finalSmall, _) = PageFinder.analysisCopy(pool, processed)
        Result(processed.clone(), kind, steps, lookStats(pool, finalSmall))
    }

    /** The look of an RGB image, measured on a small copy. */
    fun lookOf(rgb: Mat): LookStats = MatPool().use { pool ->
        val (small, _) = PageFinder.analysisCopy(pool, rgb)
        lookStats(pool, small)
    }

    // ------------------------------------------------------------ documents

    private fun document(pool: MatPool, source: Mat, steps: MutableList<String>): Mat {
        val (small, _) = PageFinder.analysisCopy(pool, source)
        var rgb = source
        paperMeans(pool, small)?.let {
            rgb = applyBalance(pool, rgb, it)
            steps.add("ضبط بياض الورقة")
        }
        val lab = pool.mat()
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val channels = pool.split(lab)
        val flat = removePaperShadows(pool, channels[0])
        steps.add("إزالة ظلال الورقة")
        val crisp = clahe(pool, flat, DOCUMENT_CLAHE)
        steps.add("إبراز الكتابة")
        val merged = pool.mat()
        Core.merge(listOf(crisp, channels[1], channels[2]), merged)
        val out = pool.mat()
        Imgproc.cvtColor(merged, out, Imgproc.COLOR_Lab2RGB)
        return sharpen(pool, out, DOCUMENT_SHARP)
    }

    /** Average colour of the paper itself (its brightest pixels): what white balance should make neutral. */
    private fun paperMeans(pool: MatPool, small: Mat): DoubleArray? {
        val lab = pool.mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_RGB2Lab)
        val labBytes = bytesOf(lab)
        val lightness = ByteArray(labBytes.size / 3) { labBytes[it * 3] }
        val top = PixelStats.percentile(lightness, PAPER_PERCENTILE)
        val limit = PAPER_WB_RATIO * top
        return maskedMeans(small, labBytes) { l, _, _ -> l >= limit }
    }

    /**
     * Paper lightness over the page, from paper pixels only (normalized convolution). Thin text
     * strokes are removed with a max filter, and the paper level is then filled in under ink and
     * big dark blocks from the paper around them. Later passes judge each pixel against its
     * own local level, so the deep part of a shadow still counts as paper and ink does not.
     */
    private fun removePaperShadows(pool: MatPool, lightness: Mat): Mat {
        val width = lightness.cols()
        val height = lightness.rows()
        val small8 = pool.mat()
        Imgproc.resize(
            lightness, small8,
            Size(max(width / PAPER_SCALE, 1).toDouble(), max(height / PAPER_SCALE, 1).toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA
        )
        val small = pool.mat()
        small8.convertTo(small, CvType.CV_32F)
        val level = paperLevel(pool, small)

        val levelFull = pool.mat()
        Imgproc.resize(level, levelFull, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        Core.max(levelFull, Scalar(1.0), levelFull)
        val gain = pool.mat()
        Core.divide(PAPER_WHITE, levelFull, gain)
        Core.min(gain, Scalar(PAPER_GAIN_MAX), gain)
        pool.release(levelFull)

        val values = pool.mat()
        lightness.convertTo(values, CvType.CV_32F)
        Core.multiply(values, gain, values)
        val out = pool.mat()
        values.convertTo(out, CvType.CV_8U)
        return out
    }

    private fun paperLevel(pool: MatPool, small: Mat): Mat {
        val grown = pool.mat()
        Imgproc.dilate(
            small, grown,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(PAPER_INK_GROW, PAPER_INK_GROW))
        )
        val sigma = max(grown.cols(), grown.rows()) / PAPER_SIGMA_DIV
        val values = FloatArray((grown.total()).toInt())
        grown.get(0, 0, values)
        val top = PixelStats.percentile(values, PAPER_PERCENTILE).toDouble()

        val firstFloor = pool.mat(grown.rows(), grown.cols(), CvType.CV_32F)
        firstFloor.setTo(Scalar(PAPER_FIRST_RATIO * top))
        var mask = paperMask(pool, grown, firstFloor)
        var level = pool.mat(grown.rows(), grown.cols(), CvType.CV_32F)
        repeat(PAPER_PASSES + 1) {
            val weightedInput = pool.mat()
            Core.multiply(grown, mask, weightedInput)
            val weighted = pool.mat()
            Imgproc.GaussianBlur(weightedInput, weighted, Size(0.0, 0.0), sigma)
            val weight = pool.mat()
            Imgproc.GaussianBlur(mask, weight, Size(0.0, 0.0), sigma)

            val denominator = pool.mat()
            Core.max(weight, Scalar(1e-3), denominator)
            val quotient = pool.mat()
            Core.divide(weighted, denominator, quotient)
            val supported = pool.mat()
            Core.compare(weight, Scalar(1e-3), supported, Core.CMP_GT)
            level = pool.mat(grown.rows(), grown.cols(), CvType.CV_32F)
            level.setTo(Scalar(top))
            quotient.copyTo(level, supported)

            val floor = pool.mat()
            Core.max(level, Scalar(1.0), floor)
            Core.multiply(floor, Scalar(PAPER_REL_RATIO), floor)
            mask = paperMask(pool, grown, floor)
        }
        return level
    }

    private fun paperMask(pool: MatPool, values: Mat, floor: Mat): Mat {
        val bytes = pool.mat()
        Core.compare(values, floor, bytes, Core.CMP_GE)
        val mask = pool.mat()
        bytes.convertTo(mask, CvType.CV_32F, 1.0 / 255.0)
        return mask
    }

    // ---------------------------------------------------------------- photos

    private fun photo(pool: MatPool, source: Mat, kind: ImageKind, request: Request, steps: MutableList<String>): Mat {
        val (small, _) = PageFinder.analysisCopy(pool, source)
        val plan = TonePlanner.plan(kind, measure(pool, small))
        steps.addAll(TonePlanner.describe(plan))

        var rgb = source
        if (plan.whiteBalance) {
            neutralMeans(pool, small, plan.neutralChroma)?.let { rgb = applyBalance(pool, rgb, it) }
        }

        val lab = pool.mat()
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val channels = pool.split(lab)
        var lightness = channels[0]
        if (plan.alphaHigh > 0 || plan.alphaLow > 0) lightness = localToneMap(pool, lightness, plan)
        if (plan.claheClip > 0) lightness = clahe(pool, lightness, plan.claheClip)
        val merged = pool.mat()
        Core.merge(listOf(lightness, channels[1], channels[2]), merged)
        var out = pool.mat()
        Imgproc.cvtColor(merged, out, Imgproc.COLOR_Lab2RGB)

        if (plan.vibrance > 0) out = vibrance(pool, out, plan.vibrance)
        out = sharpen(pool, out, plan.sharpening)

        if (request.look != null && request.lookStrength > 0 && kind != ImageKind.POSTER) {
            out = matchLook(pool, out, request.look, request.lookStrength)
            steps.add("تقريب الشكل من باقي الصور")
        }
        return out
    }

    private fun measure(pool: MatPool, small: Mat): ImageMetrics {
        val lab = pool.mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_RGB2Lab)
        val labChannels = pool.split(lab)
        val hsv = pool.mat()
        Imgproc.cvtColor(small, hsv, Imgproc.COLOR_RGB2HSV)
        val hsvChannels = pool.split(hsv)
        return ImageMetrics.measure(
            bytesOf(labChannels[0]), bytesOf(labChannels[1]), bytesOf(labChannels[2]), bytesOf(hsvChannels[1])
        )
    }

    /**
     * Pulls regions brighter than the shoulder down and lifts regions darker than the toe, using an
     * illumination estimate made on a 1/8 copy. Legitimately bright walls and dark doors are left
     * alone because only the part beyond shoulder/toe is moved.
     */
    private fun localToneMap(pool: MatPool, lightness: Mat, plan: TonePlan): Mat {
        val width = lightness.cols()
        val height = lightness.rows()
        val small8 = pool.mat()
        Imgproc.resize(
            lightness, small8,
            Size(max(width / 8, 1).toDouble(), max(height / 8, 1).toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA
        )
        val small = pool.mat()
        small8.convertTo(small, CvType.CV_32F, 1.0 / 255.0)
        val illumination = pool.mat()
        Imgproc.GaussianBlur(small, illumination, Size(0.0, 0.0), max(small.cols(), small.rows()) / 20.0)
        Core.add(illumination, Scalar(1e-3), illumination)

        val gain = pool.mat(illumination.rows(), illumination.cols(), CvType.CV_32F)
        gain.setTo(Scalar(1.0))
        if (plan.alphaHigh > 0) {
            val pull = pool.mat()
            Core.divide(plan.shoulder, illumination, pull)
            Core.pow(pull, plan.alphaHigh, pull)
            Core.min(pull, Scalar(1.0), pull)
            Core.multiply(gain, pull, gain)
        }
        if (plan.alphaLow > 0) {
            val lift = pool.mat()
            Core.divide(plan.toe, illumination, lift)
            Core.pow(lift, plan.alphaLow, lift)
            Core.max(lift, Scalar(1.0), lift)
            Core.multiply(gain, lift, gain)
        }
        Core.max(gain, Scalar(plan.gainMin), gain)
        Core.min(gain, Scalar(plan.gainMax), gain)

        val gainFull = pool.mat()
        Imgproc.resize(gain, gainFull, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val mapped = pool.mat()
        lightness.convertTo(mapped, CvType.CV_32F, 1.0 / 255.0)
        Core.multiply(mapped, gainFull, mapped)
        pool.release(gainFull)
        Core.min(mapped, Scalar(1.0), mapped)

        if (plan.alphaLow > 0) {
            // a gain cannot lift near-black; a small toe lift keeps deep shadows from being a flat black
            val deep = pool.mat()
            mapped.convertTo(deep, CvType.CV_32F, -1.0 / SHADOW_LIFT_RANGE, 1.0)
            Core.max(deep, Scalar(0.0), deep)
            Core.multiply(deep, deep, deep)
            Core.addWeighted(mapped, 1.0, deep, plan.alphaLow * SHADOW_LIFT, 0.0, mapped)
            pool.release(deep)
        }
        val out = pool.mat()
        mapped.convertTo(out, CvType.CV_8U, 255.0, 0.5)
        pool.release(mapped)
        return out
    }

    private fun clahe(pool: MatPool, lightness: Mat, clipLimit: Double): Mat {
        val clahe = Imgproc.createCLAHE(clipLimit, Size(8.0, 8.0))
        val out = pool.mat()
        try {
            clahe.apply(lightness, out)
        } finally {
            clahe.collectGarbage()
        }
        return out
    }

    private fun vibrance(pool: MatPool, rgb: Mat, amount: Double): Mat {
        val hsv = pool.mat()
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        val channels = pool.split(hsv)
        val table = pool.mat(1, 256, CvType.CV_8UC1)
        table.put(0, 0, TonePlanner.vibranceTable(amount))
        val boosted = pool.mat()
        Core.LUT(channels[1], table, boosted)
        val merged = pool.mat()
        Core.merge(listOf(channels[0], boosted, channels[2]), merged)
        val out = pool.mat()
        Imgproc.cvtColor(merged, out, Imgproc.COLOR_HSV2RGB)
        return out
    }

    private fun sharpen(pool: MatPool, rgb: Mat, strength: Double): Mat {
        if (strength <= 0.0) return rgb
        val blurred = pool.mat()
        Imgproc.GaussianBlur(rgb, blurred, Size(0.0, 0.0), 1.6)
        val out = pool.mat()
        Core.addWeighted(rgb, 1.0 + strength, blurred, -strength, 0.0, out)
        return out
    }

    // ----------------------------------------------------------- white balance

    /** Average RGB of near-neutral midtone pixels (walls, paper, grey surfaces); null if there are too few. */
    private fun neutralMeans(pool: MatPool, small: Mat, chromaLimit: Double): DoubleArray? {
        val lab = pool.mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_RGB2Lab)
        return maskedMeans(small, bytesOf(lab)) { l, a, b ->
            l in NEUTRAL_LOW..NEUTRAL_HIGH && hypot(a - 128.0, b - 128.0) < chromaLimit
        }
    }

    private inline fun maskedMeans(small: Mat, labBytes: ByteArray, accept: (Int, Int, Int) -> Boolean): DoubleArray? {
        val rgbBytes = bytesOf(small)
        val pixels = labBytes.size / 3
        val sums = DoubleArray(3)
        var count = 0
        for (i in 0 until pixels) {
            val l = labBytes[i * 3].toInt() and 0xFF
            val a = labBytes[i * 3 + 1].toInt() and 0xFF
            val b = labBytes[i * 3 + 2].toInt() and 0xFF
            if (!accept(l, a, b)) continue
            count++
            for (c in 0..2) sums[c] += (rgbBytes[i * 3 + c].toInt() and 0xFF).toDouble()
        }
        if (count.toDouble() / max(pixels, 1) < ImageMetrics.MIN_NEUTRAL_FRACTION) return null
        return DoubleArray(3) { sums[it] / count }
    }

    /** Scales each channel so the measured means become equal; the gains are kept small. */
    private fun applyBalance(pool: MatPool, rgb: Mat, means: DoubleArray): Mat {
        val gray = (means[0] + means[1] + means[2]) / 3.0
        val channels = pool.split(rgb)
        val balanced = ArrayList<Mat>()
        for (index in 0..2) {
            val gain = (gray / max(means[index], 1e-3)).coerceIn(WB_MIN, WB_MAX)
            val out = pool.mat()
            channels[index].convertTo(out, -1, gain, 0.0)
            balanced.add(out)
        }
        val merged = pool.mat()
        Core.merge(balanced, merged)
        return merged
    }

    // ------------------------------------------------------------- one common look

    private fun lookStats(pool: MatPool, small: Mat): LookStats {
        val lab = pool.mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_RGB2Lab)
        val channels = pool.split(lab)
        val mean = MatOfDouble()
        val std = MatOfDouble()
        try {
            Core.meanStdDev(channels[0], mean, std)
            return LookStats(
                lMean = mean.toArray()[0],
                lStd = std.toArray()[0],
                aMean = Core.mean(channels[1]).`val`[0] - 128.0,
                bMean = Core.mean(channels[2]).`val`[0] - 128.0
            )
        } finally {
            mean.release()
            std.release()
        }
    }

    private fun matchLook(pool: MatPool, rgb: Mat, target: LookStats, strength: Double): Mat {
        val (small, _) = PageFinder.analysisCopy(pool, rgb)
        val own = lookStats(pool, small)
        val lab = pool.mat()
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val channels = pool.split(lab)

        val table = pool.mat(1, 256, CvType.CV_8UC1)
        table.put(0, 0, LookMatcher.lightnessTable(own, target, strength))
        val lightness = pool.mat()
        Core.LUT(channels[0], table, lightness)
        val a = pool.mat()
        channels[1].convertTo(a, -1, 1.0, LookMatcher.shiftA(own, target, strength))
        val b = pool.mat()
        channels[2].convertTo(b, -1, 1.0, LookMatcher.shiftB(own, target, strength))

        val merged = pool.mat()
        Core.merge(listOf(lightness, a, b), merged)
        val out = pool.mat()
        Imgproc.cvtColor(merged, out, Imgproc.COLOR_Lab2RGB)
        return out
    }

    /** Share of strongly coloured pixels: high for a poster, low for a printed page. */
    private fun colorfulShare(pool: MatPool, small: Mat): Double {
        val hsv = pool.mat()
        Imgproc.cvtColor(small, hsv, Imgproc.COLOR_RGB2HSV)
        val bytes = bytesOf(hsv)
        val pixels = bytes.size / 3
        var colourful = 0
        for (i in 0 until pixels) {
            val saturation = bytes[i * 3 + 1].toInt() and 0xFF
            val value = bytes[i * 3 + 2].toInt() and 0xFF
            if (saturation > 70 && value > 70) colourful++
        }
        return colourful.toDouble() / max(pixels, 1)
    }

    /** Raw bytes of a continuous 8-bit Mat. */
    private fun bytesOf(mat: Mat): ByteArray {
        val data = ByteArray((mat.total() * mat.channels()).toInt())
        mat.get(0, 0, data)
        return data
    }
}
