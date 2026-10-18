package com.photounifier

import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The arithmetic behind page detection, on plain arrays so it can be unit-tested without OpenCV.
 * A quad is eight numbers: x0,y0 .. x3,y3, ordered top-left, top-right, bottom-right, bottom-left.
 */
internal object PageMath {
    const val BORDER_BAND = 0.06
    const val FG_MAHALANOBIS2 = 16.0
    private const val COVARIANCE_FLOOR = 4.0

    /** Orders four corners as TL, TR, BR, BL: the top-left has the smallest x+y, the bottom-right the largest. */
    fun orderCorners(points: DoubleArray): DoubleArray {
        require(points.size == 8) { "four corners expected" }
        val sums = DoubleArray(4) { points[it * 2] + points[it * 2 + 1] }
        val diffs = DoubleArray(4) { points[it * 2 + 1] - points[it * 2] }
        val tl = sums.indices.minByOrNull { sums[it] }!!
        val br = sums.indices.maxByOrNull { sums[it] }!!
        val tr = diffs.indices.minByOrNull { diffs[it] }!!
        val bl = diffs.indices.maxByOrNull { diffs[it] }!!
        if (setOf(tl, tr, br, bl).size != 4) return points.copyOf()
        return doubleArrayOf(
            points[tl * 2], points[tl * 2 + 1], points[tr * 2], points[tr * 2 + 1],
            points[br * 2], points[br * 2 + 1], points[bl * 2], points[bl * 2 + 1]
        )
    }

    fun distance(quad: DoubleArray, from: Int, to: Int): Double =
        hypot(quad[to * 2] - quad[from * 2], quad[to * 2 + 1] - quad[from * 2 + 1])

    /** Longest top/bottom and left/right sides: the size the page gets once flattened. */
    fun outputSize(quad: DoubleArray, maxSide: Int): Pair<Int, Int> {
        val width = max(distance(quad, 0, 1), distance(quad, 3, 2)).roundToInt().coerceIn(2, maxSide)
        val height = max(distance(quad, 0, 3), distance(quad, 1, 2)).roundToInt().coerceIn(2, maxSide)
        return width to height
    }

    fun aspectOk(quad: DoubleArray): Boolean {
        val width = max(distance(quad, 0, 1), distance(quad, 3, 2))
        val height = max(distance(quad, 0, 3), distance(quad, 1, 2))
        return height >= 1.0 && width / height in 0.3..3.5
    }

    /** True when every corner sits near a frame corner: that is the whole frame, not a page in it. */
    fun coversFrame(quad: DoubleArray, width: Int, height: Int, margin: Double = 0.03): Boolean =
        (0..3).all { i ->
            val x = quad[i * 2]
            val y = quad[i * 2 + 1]
            (x < margin * width || x > (1 - margin) * width) && (y < margin * height || y > (1 - margin) * height)
        }

    /** Share of points along the quad outline that lie on a set pixel of [edges] (width*height, 0 = no edge). */
    fun edgeSupport(quad: DoubleArray, edges: ByteArray, width: Int, height: Int): Double {
        var hits = 0
        var total = 0
        for (i in 0..3) {
            val j = (i + 1) % 4
            for (step in 0 until SUPPORT_SAMPLES) {
                val t = 0.05 + 0.9 * step / (SUPPORT_SAMPLES - 1)
                val x = (quad[i * 2] + (quad[j * 2] - quad[i * 2]) * t).roundToInt()
                val y = (quad[i * 2 + 1] + (quad[j * 2 + 1] - quad[i * 2 + 1]) * t).roundToInt()
                total++
                if (x in 0 until width && y in 0 until height && edges[y * width + x].toInt() != 0) hits++
            }
        }
        return hits.toDouble() / max(total, 1)
    }

    private const val SUPPORT_SAMPLES = 40

    /**
     * Foreground = pixels that do not look like the frame border (a page is usually surrounded by
     * table or wall). The border colours are modelled as one Gaussian in Lab; anything beyond
     * [FG_MAHALANOBIS2] (squared distance) is foreground. A shadow on the page stays far from the
     * table or wall colour, so it does not break the page into pieces.
     *
     * [lab] holds interleaved L,a,b bytes, row by row. Returns width*height bytes, 255 = foreground.
     */
    fun foreground(lab: ByteArray, width: Int, height: Int): ByteArray {
        require(lab.size == width * height * 3) { "lab size does not match the frame" }
        val band = max((BORDER_BAND * min(width, height)).toInt(), 4)
        fun onBorder(x: Int, y: Int) = x < band || y < band || x >= width - band || y >= height - band

        var count = 0
        val mean = DoubleArray(3)
        for (y in 0 until height) for (x in 0 until width) {
            if (!onBorder(x, y)) continue
            count++
            for (c in 0..2) mean[c] = mean[c] + (lab[(y * width + x) * 3 + c].toInt() and 0xFF).toDouble()
        }
        if (count < 2) return ByteArray(width * height)
        for (c in 0..2) mean[c] = mean[c] / count

        val cov = Array(3) { DoubleArray(3) }
        for (y in 0 until height) for (x in 0 until width) {
            if (!onBorder(x, y)) continue
            val d = DoubleArray(3) { c -> (lab[(y * width + x) * 3 + c].toInt() and 0xFF) - mean[c] }
            for (i in 0..2) for (j in 0..2) cov[i][j] += d[i] * d[j]
        }
        for (i in 0..2) for (j in 0..2) cov[i][j] = cov[i][j] / (count - 1) + if (i == j) COVARIANCE_FLOOR else 0.0
        val inverse = invert3(cov)

        val output = ByteArray(width * height)
        for (i in output.indices) {
            val d0 = (lab[i * 3].toInt() and 0xFF) - mean[0]
            val d1 = (lab[i * 3 + 1].toInt() and 0xFF) - mean[1]
            val d2 = (lab[i * 3 + 2].toInt() and 0xFF) - mean[2]
            val distance2 =
                d0 * (inverse[0][0] * d0 + inverse[0][1] * d1 + inverse[0][2] * d2) +
                d1 * (inverse[1][0] * d0 + inverse[1][1] * d1 + inverse[1][2] * d2) +
                d2 * (inverse[2][0] * d0 + inverse[2][1] * d1 + inverse[2][2] * d2)
            if (distance2 > FG_MAHALANOBIS2) output[i] = 255.toByte()
        }
        return output
    }

    internal fun invert3(m: Array<DoubleArray>): Array<DoubleArray> {
        val a = m[0][0]; val b = m[0][1]; val c = m[0][2]
        val d = m[1][0]; val e = m[1][1]; val f = m[1][2]
        val g = m[2][0]; val h = m[2][1]; val i = m[2][2]
        val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        require(sqrt(det * det) > 1e-9) { "singular covariance" }
        val s = 1.0 / det
        return arrayOf(
            doubleArrayOf((e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s),
            doubleArrayOf((f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s),
            doubleArrayOf((d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s)
        )
    }
}
