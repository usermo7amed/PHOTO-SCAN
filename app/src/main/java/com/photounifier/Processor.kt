package com.photounifier

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/** Image I/O and the Bitmap <-> OpenCV bridge; the processing itself lives in [Enhancer]. */
object Processor {
    const val MAX_SIDE = 4096

    /** What [enhance] produced: the picture, the route actually used, and what was done to it. */
    class Outcome(val bitmap: Bitmap, val kind: ImageKind, val steps: List<String>, val look: LookStats)

    // ------------------------------------------------------------------ I/O

    fun load(context: Context, uri: Uri, maxSide: Int = MAX_SIDE): Bitmap {
        val resolver = context.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalStateException("ملف الصورة غير صالح أو غير مدعوم")
        }

        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxSide) sample *= 2
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = resolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: throw IllegalStateException("تعذر قراءة الصورة")

        val orientation = resolver.openInputStream(uri)?.use {
            ExifInterface(it).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } ?: ExifInterface.ORIENTATION_NORMAL
        return applyOrientation(bitmap, orientation)
    }

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }
            else -> return bitmap
        }
        val rotated = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    fun save(context: Context, bitmap: Bitmap, displayName: String): Uri {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PhotoUnifier")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("تعذر إنشاء ملف الصورة")
        val written = resolver.openOutputStream(uri)?.use {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
        } ?: false
        if (!written) {
            resolver.delete(uri, null, null)
            throw IllegalStateException("تعذر حفظ الصورة")
        }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return uri
    }

    // ----------------------------------------------------------- entry points

    fun enhance(src: Bitmap, request: Enhancer.Request): Outcome {
        val rgba = Mat()
        val rgb = Mat()
        try {
            Utils.bitmapToMat(src, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Enhancer.enhance(rgb, request).use { result ->
                val out = Mat()
                try {
                    Imgproc.cvtColor(result.rgb, out, Imgproc.COLOR_RGB2RGBA)
                    val bitmap = Bitmap.createBitmap(out.cols(), out.rows(), Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(out, bitmap)
                    return Outcome(bitmap, result.kind, result.steps, result.look)
                } finally {
                    out.release()
                }
            }
        } finally {
            rgba.release()
            rgb.release()
        }
    }

    /** The look of an image as it is, without changing it. */
    fun lookOf(src: Bitmap): LookStats {
        val rgba = Mat()
        val rgb = Mat()
        try {
            Utils.bitmapToMat(src, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            return Enhancer.lookOf(rgb)
        } finally {
            rgba.release()
            rgb.release()
        }
    }
}
