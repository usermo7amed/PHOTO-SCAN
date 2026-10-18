package com.photounifier

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint

/** Shared, stateless manual brightness/contrast adjustment. */
internal object BitmapAdjustments {
    fun apply(source: Bitmap, brightness: Int, contrast: Float): Bitmap {
        val output = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val scale = contrast.coerceIn(0.5f, 2.0f)
        // contrast pivots around mid-grey (128), so raising it darkens shadows and brightens highlights
        // instead of brightening everything; the brightness offset is added on top
        val offset = brightness.toFloat() + 128f * (1f - scale)
        val matrix = ColorMatrix(floatArrayOf(
            scale, 0f, 0f, 0f, offset,
            0f, scale, 0f, 0f, offset,
            0f, 0f, scale, 0f, offset,
            0f, 0f, 0f, 1f, 0f
        ))
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(matrix)
        }
        Canvas(output).drawBitmap(source, 0f, 0f, paint)
        return output
    }
}
