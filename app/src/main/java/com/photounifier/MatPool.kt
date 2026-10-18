package com.photounifier

import org.opencv.core.Core
import org.opencv.core.Mat

/** Keeps native OpenCV allocations alive until the pipeline finishes, then releases them. */
internal class MatPool : AutoCloseable {
    private val items = ArrayList<Mat>()

    fun mat(): Mat = Mat().also { items.add(it) }

    fun mat(rows: Int, cols: Int, type: Int): Mat = Mat(rows, cols, type).also { items.add(it) }

    fun <T : Mat> own(value: T): T = value.also { items.add(it) }

    fun split(src: Mat): List<Mat> {
        val output = ArrayList<Mat>()
        Core.split(src, output)
        items.addAll(output)
        return output
    }

    /** Frees a big temporary before the pipeline ends. */
    fun release(value: Mat) {
        if (items.remove(value)) value.release()
    }

    override fun close() {
        items.forEach { it.release() }
        items.clear()
    }
}
