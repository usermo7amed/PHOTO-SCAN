package com.photounifier

/** Small order-statistics helpers shared by the measuring code. Plain Kotlin, no OpenCV. */
internal object PixelStats {
    /** Nearest-rank percentile (p in 0..100) of 8-bit values. */
    fun percentile(values: ByteArray, p: Double): Int {
        require(values.isNotEmpty()) { "no values" }
        val histogram = IntArray(256)
        for (v in values) histogram[v.toInt() and 0xFF]++
        val target = (p.coerceIn(0.0, 100.0) / 100.0 * (values.size - 1)).toInt() + 1
        var seen = 0
        for (level in 0..255) {
            seen += histogram[level]
            if (seen >= target) return level
        }
        return 255
    }

    fun median(values: ByteArray): Int = percentile(values, 50.0)

    /** Percentile (p in 0..100) of float values, by sorting a copy. */
    fun percentile(values: FloatArray, p: Double): Float {
        require(values.isNotEmpty()) { "no values" }
        val sorted = values.copyOf().also { it.sort() }
        val index = (p.coerceIn(0.0, 100.0) / 100.0 * (sorted.size - 1)).toInt()
        return sorted[index]
    }
}
