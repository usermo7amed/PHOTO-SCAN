package com.photounifier

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Owns image processing and persistence; it has no Activity or View references. */
class PhotoRepository {
    interface Observer {
        fun onSource(index: Int, total: Int, bitmap: Bitmap)
        fun onClassification(index: Int, total: Int, kind: ImageKind)
        /** What was actually decided for one image: the route used and the corrections applied. */
        fun onDecision(index: Int, total: Int, kind: ImageKind, steps: List<String>)
        fun onEnhanced(bitmap: Bitmap)
        fun onSaved(index: Int, bitmap: Bitmap)
        fun onNote(message: String)
        fun onFinished(success: Int, failed: Int, error: String?)
    }

    class Options(
        val brightness: Int,
        val contrastPercent: Int,
        /** A route chosen by the person; null lets the app decide per photo. */
        val mode: ImageKind? = null,
        /** 0 leaves every photo as it is; 100 pulls photos fully towards the common look. */
        val unifyPercent: Int = 0,
        /** A photo whose look the others should follow; null uses the middle look of the batch. */
        val reference: Uri? = null
    )

    fun process(
        context: Context,
        uris: List<Uri>,
        options: Options,
        observer: Observer,
        firstSourceOverride: Bitmap? = null
    ) {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        var success = 0
        var failed = 0
        var lastError: String? = null
        val classifier = try { SceneClassifier() } catch (_: Throwable) { null }
        try {
            val labelKinds = arrayOfNulls<ImageKind>(uris.size)
            val strength = options.unifyPercent.coerceIn(0, 100) / 100.0
            val targetLook = if (strength > 0.0) {
                observer.onNote("تحليل الصور لتوحيد الشكل…")
                try {
                    targetLook(context, uris, options, classifier, labelKinds)
                } catch (error: Throwable) {
                    lastError = error.message ?: error.javaClass.simpleName
                    null
                }
            } else null

            uris.forEachIndexed { index, uri ->
                var source: Bitmap? = null
                var enhanced: Bitmap? = null
                var output: Bitmap? = null
                try {
                    val manualCrop = index == 0 && firstSourceOverride != null
                    val loadedSource = if (manualCrop) {
                        firstSourceOverride!!.copy(Bitmap.Config.ARGB_8888, false)
                    } else {
                        Processor.load(context, uri)
                    }
                    source = loadedSource
                    observer.onSource(index + 1, uris.size, loadedSource.copy(Bitmap.Config.ARGB_8888, false))

                    val kind = options.mode ?: labelKinds[index]
                        ?: classifier?.classify(loadedSource)?.kind ?: ImageKind.GENERAL
                    observer.onClassification(index + 1, uris.size, kind)

                    val outcome = Processor.enhance(
                        loadedSource,
                        Enhancer.Request(
                            kind = kind,
                            auto = options.mode == null,
                            // a hand-made crop is already the page: do not look for another one inside it
                            rectify = !manualCrop,
                            look = targetLook,
                            lookStrength = strength
                        )
                    )
                    enhanced = outcome.bitmap
                    observer.onDecision(index + 1, uris.size, outcome.kind, outcome.steps)
                    observer.onEnhanced(outcome.bitmap.copy(Bitmap.Config.ARGB_8888, false))
                    output = BitmapAdjustments.apply(outcome.bitmap, options.brightness, options.contrastPercent / 100f)
                    Processor.save(context, output, "PU_%s_%03d.jpg".format(Locale.US, stamp, index + 1))
                    observer.onSaved(index + 1, output.copy(Bitmap.Config.ARGB_8888, false))
                    success++
                } catch (error: Throwable) {
                    failed++
                    lastError = error.message ?: error.javaClass.simpleName
                } finally {
                    source?.let { if (!it.isRecycled) it.recycle() }
                    enhanced?.let { if (!it.isRecycled) it.recycle() }
                    output?.let { if (!it.isRecycled) it.recycle() }
                }
            }
        } finally {
            classifier?.close()
            observer.onFinished(success, failed, lastError)
        }
    }

    /**
     * The look the photos should follow: the chosen reference, or the middle look of the batch.
     * It is measured on small copies, so this pass costs little next to the real processing.
     * Pages and posters keep their own look and are left out.
     */
    private fun targetLook(
        context: Context,
        uris: List<Uri>,
        options: Options,
        classifier: SceneClassifier?,
        labelKinds: Array<ImageKind?>
    ): LookStats? {
        options.reference?.let { reference ->
            val bitmap = Processor.load(context, reference, PREVIEW_SIDE)
            try {
                val kind = options.mode ?: classifier?.classify(bitmap)?.kind
                return Processor.enhance(bitmap, previewRequest(kind)).let {
                    it.bitmap.recycle()
                    it.look
                }
            } finally {
                bitmap.recycle()
            }
        }

        val looks = ArrayList<LookStats>()
        uris.forEachIndexed { index, uri ->
            val bitmap = try { Processor.load(context, uri, PREVIEW_SIDE) } catch (_: Throwable) { return@forEachIndexed }
            try {
                val kind = options.mode ?: classifier?.classify(bitmap)?.kind ?: ImageKind.GENERAL
                labelKinds[index] = options.mode ?: kind
                if (kind == ImageKind.DOCUMENT || kind == ImageKind.POSTER) return@forEachIndexed
                val outcome = Processor.enhance(bitmap, previewRequest(kind))
                outcome.bitmap.recycle()
                looks.add(outcome.look)
            } finally {
                bitmap.recycle()
            }
        }
        return if (looks.size >= 2) LookMatcher.median(looks) else null
    }

    private fun previewRequest(kind: ImageKind?): Enhancer.Request {
        val photoKind = when (kind) {
            null, ImageKind.DOCUMENT, ImageKind.POSTER -> ImageKind.GENERAL
            else -> kind
        }
        return Enhancer.Request(kind = photoKind, auto = false, rectify = false)
    }

    private companion object {
        const val PREVIEW_SIDE = 1024
    }
}
