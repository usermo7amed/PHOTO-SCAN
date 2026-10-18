package com.photounifier

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Finds the page or poster in a photo and flattens it. One detector serves both the automatic
 * route and the manual crop editor's first guess.
 *
 * Several masks are tried (what differs from the frame border, closed edges, Otsu threshold and its
 * inverse) and a candidate must pass four independent checks: size, squareness, an outline that
 * sits on real edges, and contrast with its surroundings. A strict mode asks for more of each, so
 * the app can cut out a page without any hint from the photo's labels and still leave ordinary
 * photos alone.
 */
internal object PageFinder {
    const val MAX_SIDE = 4096
    const val ANALYSIS_SIDE = 640
    private const val MAX_AREA = 0.97
    private const val MIN_AREA = 0.20
    private const val MIN_RECT = 0.85
    private const val MIN_SUPPORT = 0.70
    private const val STRICT_AREA = 0.30
    private const val STRICT_RECT = 0.93
    private const val STRICT_SUPPORT = 0.85
    private const val MIN_CONTRAST = 12.0
    private const val EDGE_TRIM = 0.006
    private val HULL_EPSILONS = doubleArrayOf(0.015, 0.02, 0.03, 0.04, 0.06)

    class Found(val corners: DoubleArray, val scale: Double)

    /** A copy of [rgb] with its longest side at most [ANALYSIS_SIDE], and the scale that was applied. */
    fun analysisCopy(pool: MatPool, rgb: Mat): Pair<Mat, Double> {
        val scale = min(1.0, ANALYSIS_SIDE.toDouble() / max(rgb.cols(), rgb.rows()))
        if (scale >= 1.0) return rgb to 1.0
        val small = pool.mat()
        Imgproc.resize(
            rgb, small,
            Size(max((rgb.cols() * scale).toInt(), 1).toDouble(), max((rgb.rows() * scale).toInt(), 1).toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA
        )
        return small to scale
    }

    /** Page corners (TL, TR, BR, BL) in full-image pixels, or null when there is no clear page. */
    fun find(pool: MatPool, rgb: Mat, strict: Boolean): Found? {
        if (rgb.cols() < 160 || rgb.rows() < 160) return null
        val (small, scale) = analysisCopy(pool, rgb)
        val corners = findIn(pool, small, strict) ?: return null
        return Found(DoubleArray(8) { corners[it] / scale }, scale)
    }

    /** Same, with the corners in the coordinates of [small] itself. */
    fun findIn(pool: MatPool, small: Mat, strict: Boolean): DoubleArray? {
        val minArea = if (strict) STRICT_AREA else MIN_AREA
        val minRect = if (strict) STRICT_RECT else MIN_RECT
        val minSupport = if (strict) STRICT_SUPPORT else MIN_SUPPORT
        val width = small.cols()
        val height = small.rows()
        val imageArea = width.toDouble() * height

        val gray = pool.mat()
        Imgproc.cvtColor(small, gray, Imgproc.COLOR_RGB2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val grayBytes = ByteArray(width * height)
        gray.get(0, 0, grayBytes)
        val median = PixelStats.median(grayBytes).toDouble()
        val raw = pool.mat()
        Imgproc.Canny(gray, raw, max(0.66 * median, 20.0), max(1.33 * median, 60.0))

        // colour edges too: an orange poster on a grey wall can match the wall's brightness
        val lab = pool.mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_RGB2Lab)
        val labChannels = pool.split(lab)
        for (index in 1..2) {
            val blurred = pool.mat()
            Imgproc.GaussianBlur(labChannels[index], blurred, Size(5.0, 5.0), 0.0)
            val colourEdges = pool.mat()
            Imgproc.Canny(blurred, colourEdges, 15.0, 45.0)
            Core.bitwise_or(raw, colourEdges, raw)
        }

        val edgesNear = pool.mat()
        Imgproc.dilate(raw, edgesNear, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0)))
        val edgeBytes = ByteArray(width * height)
        edgesNear.get(0, 0, edgeBytes)
        val edgesClosed = pool.mat()
        Imgproc.morphologyEx(
            raw, edgesClosed, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0)),
            Point(-1.0, -1.0), 2
        )
        val threshold = pool.mat()
        Imgproc.threshold(gray, threshold, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        Imgproc.morphologyEx(
            threshold, threshold, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
        )
        val inverse = pool.mat()
        Core.bitwise_not(threshold, inverse)

        val foreground = foregroundMask(pool, lab, width, height)

        var best: DoubleArray? = null
        var bestScore = 0.0
        for (mask in listOf(foreground, edgesClosed, threshold, inverse)) {
            for (candidate in candidates(mask, imageArea, minArea)) {
                if (candidate.rectangularity < minRect) continue
                val quad = candidate.quad
                if (PageMath.coversFrame(quad, width, height)) continue
                if (!PageMath.aspectOk(quad)) continue
                val support = PageMath.edgeSupport(quad, edgeBytes, width, height)
                if (support < minSupport) continue
                if (candidate.fraction < 0.9 && !contrastsWithSurroundings(pool, gray, quad)) continue
                val score = candidate.fraction * candidate.rectangularity * support
                if (score > bestScore) {
                    best = quad
                    bestScore = score
                }
            }
        }
        return best
    }

    private class Candidate(val quad: DoubleArray, val fraction: Double, val rectangularity: Double)

    private fun candidates(mask: Mat, imageArea: Double, minArea: Double): List<Candidate> {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val found = ArrayList<Candidate>()
        try {
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            for (contour in contours.sortedByDescending { Imgproc.contourArea(it) }.take(8)) {
                val hull = convexHull(contour)
                if (hull.size < 4) continue
                val quad = quadFromHull(hull) ?: continue
                val quadArea = areaOf(pointsOf(quad).toList())
                val fraction = quadArea / imageArea
                if (fraction < minArea || fraction > MAX_AREA) continue
                found.add(Candidate(quad, fraction, min(areaOf(hull) / max(quadArea, 1.0), 1.0)))
            }
        } finally {
            contours.forEach { it.release() }
            hierarchy.release()
        }
        return found
    }

    private fun convexHull(contour: MatOfPoint): List<Point> {
        val index = MatOfInt()
        try {
            Imgproc.convexHull(contour, index)
            val points = contour.toArray()
            return index.toArray().map { points[it] }
        } finally {
            index.release()
        }
    }

    private fun areaOf(points: List<Point>): Double {
        val curve = MatOfPoint2f(*points.toTypedArray())
        try {
            return abs(Imgproc.contourArea(curve))
        } finally {
            curve.release()
        }
    }

    /** The convex hull reduced to four corners using the smallest tolerance that gets there. */
    private fun quadFromHull(hull: List<Point>): DoubleArray? {
        val curve = MatOfPoint2f(*hull.toTypedArray())
        try {
            val perimeter = Imgproc.arcLength(curve, true)
            for (epsilon in HULL_EPSILONS) {
                val approx = MatOfPoint2f()
                try {
                    Imgproc.approxPolyDP(curve, approx, epsilon * perimeter, true)
                    val corners = approx.toArray()
                    if (corners.size == 4) {
                        return PageMath.orderCorners(doubleArrayOf(
                            corners[0].x, corners[0].y, corners[1].x, corners[1].y,
                            corners[2].x, corners[2].y, corners[3].x, corners[3].y
                        ))
                    }
                    if (corners.size < 4) return null
                } finally {
                    approx.release()
                }
            }
            return null
        } finally {
            curve.release()
        }
    }

    private fun contrastsWithSurroundings(pool: MatPool, gray: Mat, quad: DoubleArray): Boolean {
        val inside = pool.mat(gray.rows(), gray.cols(), CvType.CV_8UC1)
        inside.setTo(Scalar(0.0))
        val polygon = MatOfPoint(*pointsOf(quad))
        try {
            Imgproc.fillConvexPoly(inside, polygon, Scalar(255.0))
        } finally { polygon.release() }
        val outside = pool.mat()
        Core.bitwise_not(inside, outside)
        val insideMean = Core.mean(gray, inside).`val`[0]
        val outsideMean = Core.mean(gray, outside).`val`[0]
        pool.release(inside)
        pool.release(outside)
        return abs(insideMean - outsideMean) >= MIN_CONTRAST
    }

    private fun foregroundMask(pool: MatPool, lab: Mat, width: Int, height: Int): Mat {
        val bytes = ByteArray(width * height * 3)
        lab.get(0, 0, bytes)
        val mask = pool.mat(height, width, CvType.CV_8UC1)
        mask.put(0, 0, PageMath.foreground(bytes, width, height))
        Imgproc.morphologyEx(
            mask, mask, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        )
        Imgproc.morphologyEx(
            mask, mask, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))
        )
        return mask
    }

    private fun pointsOf(quad: DoubleArray): Array<Point> =
        Array(4) { Point(quad[it * 2], quad[it * 2 + 1]) }

    /** Flattens the quad (full-image pixels) into an upright page and trims the table-edge residue. */
    fun warp(pool: MatPool, rgb: Mat, quad: DoubleArray): Mat {
        val (width, height) = PageMath.outputSize(quad, MAX_SIDE)
        val source = pool.own(MatOfPoint2f(*pointsOf(quad)))
        val target = pool.own(MatOfPoint2f(
            Point(0.0, 0.0), Point((width - 1).toDouble(), 0.0),
            Point((width - 1).toDouble(), (height - 1).toDouble()), Point(0.0, (height - 1).toDouble())
        ))
        val transform = pool.own(Imgproc.getPerspectiveTransform(source, target, Core.DECOMP_LU))
        val page = pool.mat()
        Imgproc.warpPerspective(
            rgb, page, transform, Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE, Scalar(255.0, 255.0, 255.0)
        )
        val trim = Math.round(EDGE_TRIM * min(width, height)).toInt()
        if (trim > 0 && width - 2 * trim >= 64 && height - 2 * trim >= 64) {
            val trimmed = pool.mat()
            val view = page.submat(Rect(trim, trim, width - 2 * trim, height - 2 * trim))
            view.copyTo(trimmed)
            view.release()
            return trimmed
        }
        return page
    }
}
