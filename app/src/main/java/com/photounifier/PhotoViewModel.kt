package com.photounifier

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

data class PhotoUiState(
    val selected: List<Uri> = emptyList(),
    val busy: Boolean = false,
    val progress: Int = 0,
    val progressMax: Int = 1,
    val status: String = "جاهز لاختيار الصور",
    val sourcePreview: Bitmap? = null,
    val enhancedPreview: Bitmap? = null,
    val resultPreview: Bitmap? = null,
    val brightness: Int = 0,
    val contrastPercent: Int = 100,
    val cropSource: Bitmap? = null,
    val cropPreview: Bitmap? = null,
    val cropCorners: List<CropPoint> = emptyList(),
    val cropUndoStack: List<List<CropPoint>> = emptyList(),
    val cropOpen: Boolean = false,
    val cropApplied: Boolean = false,
    /** The route the person chose; null means the app decides for each photo. */
    val mode: ImageKind? = null,
    /** 0..100: how far photos are pulled towards one common look. */
    val unifyPercent: Int = 0,
    val referenceUri: Uri? = null,
    /** One line per image: the route used and the corrections applied. */
    val decisions: List<String> = emptyList()
)

/** Owns screen state and coordinates background work; the Activity only renders state. */
class PhotoViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = PhotoRepository()
    private val executor = Executors.newSingleThreadExecutor()
    private val previewExecutor = Executors.newSingleThreadExecutor()
    private val previewGeneration = AtomicInteger(0)
    private val cropGeneration = AtomicInteger(0)
    private val mutableState = MutableLiveData(PhotoUiState())
    val state: LiveData<PhotoUiState> = mutableState
    @Volatile private var currentState = PhotoUiState()

    @Synchronized
    private fun update(transform: (PhotoUiState) -> PhotoUiState) {
        currentState = transform(currentState)
        mutableState.postValue(currentState)
    }

    fun selectImages(uris: List<Uri>) {
        update { it.copy(
            selected = uris.toList(), progress = 0,
            status = if (uris.isEmpty()) "لم يتم اختيار صور" else "تم اختيار ${uris.size} صورة",
            sourcePreview = null, enhancedPreview = null, resultPreview = null,
            cropSource = null, cropPreview = null, cropCorners = emptyList(),
            cropUndoStack = emptyList(), cropOpen = false, cropApplied = false,
            decisions = emptyList()
        ) }
        if (uris.isNotEmpty()) executor.execute {
            try {
                val bitmap = Processor.load(getApplication<Application>().applicationContext, uris.first())
                update { old ->
                    if (old.selected.firstOrNull() != uris.first()) { bitmap.recycle(); old }
                    else old.copy(sourcePreview = bitmap, cropSource = bitmap)
                }
            } catch (error: Exception) {
                update { it.copy(status = "تعذر تحميل معاينة الصورة: ${error.message ?: "خطأ"}") }
            }
        }
    }

    fun openCropEditor(openCvReady: Boolean) {
        val source = currentState.cropSource ?: return
        if (!openCvReady) { update { it.copy(status = "مكتبة OpenCV مش جاهزة") }; return }
        executor.execute {
            val detected = try { DocumentCrop.detect(source) } catch (_: Exception) { null }
            val corners = detected ?: DocumentCrop.fullFrame(source.width, source.height)
            val preview = try { DocumentCrop.warp(source, corners) } catch (_: Exception) { source.copy(Bitmap.Config.ARGB_8888, false) }
            update { old ->
                if (old.cropSource !== source) { preview.recycle(); old }
                else old.copy(cropCorners = corners, cropPreview = preview, cropUndoStack = emptyList(), cropOpen = true,
                    status = if (detected == null) "مفيش ورقة واضحة؛ ظبط الأربع زوايا يدويًا." else "تم اكتشاف حدود الورقة. اسحب النقاط للتعديل.")
            }
        }
    }

    fun moveCropCorner(index: Int, x: Float, y: Float) {
        val snapshot = currentState
        val source = snapshot.cropSource ?: return
        if (!snapshot.cropOpen || index !in 0..3 || snapshot.cropCorners.size != 4) return
        val candidate = snapshot.cropCorners.toMutableList()
        candidate[index] = CropPoint(x.coerceIn(0f, (source.width - 1).toFloat()), y.coerceIn(0f, (source.height - 1).toFloat()))
        if (!DocumentCrop.isValid(candidate, source.width, source.height)) return
        val generation = cropGeneration.incrementAndGet()
        update { it.copy(cropCorners = candidate, cropUndoStack = it.cropUndoStack + listOf(it.cropCorners)) }
        executor.execute {
            val preview = try { DocumentCrop.warp(source, candidate) } catch (_: Exception) { return@execute }
            if (generation != cropGeneration.get()) { preview.recycle(); return@execute }
            update { old ->
                if (!old.cropOpen || old.cropCorners != candidate) { preview.recycle(); old }
                else old.copy(cropPreview = preview)
            }
        }
    }

    fun undoCrop() {
        val snapshot = currentState
        val previous = snapshot.cropUndoStack.lastOrNull() ?: return
        val source = snapshot.cropSource ?: return
        val generation = cropGeneration.incrementAndGet()
        update { it.copy(cropCorners = previous, cropUndoStack = it.cropUndoStack.dropLast(1)) }
        executor.execute {
            val preview = try { DocumentCrop.warp(source, previous) } catch (_: Exception) { return@execute }
            if (generation != cropGeneration.get()) { preview.recycle(); return@execute }
            update { old -> if (old.cropCorners != previous) { preview.recycle(); old } else old.copy(cropPreview = preview) }
        }
    }

    fun resetCrop() {
        val source = currentState.cropSource ?: return
        val corners = DocumentCrop.fullFrame(source.width, source.height)
        val generation = cropGeneration.incrementAndGet()
        update { it.copy(cropCorners = corners, cropUndoStack = it.cropUndoStack + listOf(it.cropCorners)) }
        executor.execute {
            val preview = DocumentCrop.warp(source, corners)
            if (generation != cropGeneration.get()) { preview.recycle(); return@execute }
            update { it.copy(cropPreview = preview) }
        }
    }

    fun cancelCrop() {
        cropGeneration.incrementAndGet()
        update { it.copy(cropOpen = false, cropCorners = emptyList(), cropUndoStack = emptyList(), cropPreview = null, cropApplied = false) }
    }

    fun applyCrop() {
        val preview = currentState.cropPreview ?: return
        val applied = preview.copy(Bitmap.Config.ARGB_8888, false)
        update { it.copy(cropOpen = false, cropApplied = true, sourcePreview = applied, status = "تم تطبيق القص على الصورة الأولى؛ ابدأ التحسين.") }
    }

    fun setMode(mode: ImageKind?) {
        update { it.copy(mode = mode) }
    }

    fun setUnify(percent: Int) {
        update { it.copy(unifyPercent = percent.coerceIn(0, 100)) }
    }

    fun setReference(uri: Uri?) {
        update { it.copy(referenceUri = uri) }
    }

    fun setBrightness(value: Int) {
        update { it.copy(brightness = value.coerceIn(-50, 50)) }
        refreshPreview()
    }

    fun setContrast(value: Int) {
        update { it.copy(contrastPercent = value.coerceIn(50, 200)) }
        refreshPreview()
    }

    private fun refreshPreview() {
        val snapshot = currentState
        val base = snapshot.enhancedPreview ?: return
        val generation = previewGeneration.incrementAndGet()
        previewExecutor.execute {
            if (base.isRecycled) return@execute
            val preview = try {
                BitmapAdjustments.apply(base, snapshot.brightness, snapshot.contrastPercent / 100f)
            } catch (_: Exception) { return@execute }
            if (generation != previewGeneration.get()) {
                preview.recycle()
                return@execute
            }
            update { old ->
                if (old.enhancedPreview !== base || generation != previewGeneration.get()) {
                    preview.recycle()
                    old
                } else old.copy(resultPreview = preview)
            }
        }
    }

    fun process(openCvReady: Boolean) {
        val snapshot = currentState
        if (snapshot.busy || snapshot.selected.isEmpty()) return
        if (!openCvReady) {
            update { it.copy(status = "مكتبة معالجة الصور غير جاهزة") }
            return
        }
        val uris = snapshot.selected
        val options = PhotoRepository.Options(
            brightness = snapshot.brightness,
            contrastPercent = snapshot.contrastPercent,
            mode = snapshot.mode,
            unifyPercent = snapshot.unifyPercent,
            reference = snapshot.referenceUri
        )
        update { it.copy(busy = true, progress = 0, progressMax = uris.size, decisions = emptyList(), status = "بدء معالجة الصور…") }

        executor.execute {
            repository.process(
                getApplication<Application>().applicationContext, uris, options,
                object : PhotoRepository.Observer {
                    override fun onSource(index: Int, total: Int, bitmap: Bitmap) {
                        update { it.copy(status = "تحليل الصورة $index من $total", sourcePreview = bitmap) }
                    }
                    override fun onClassification(index: Int, total: Int, kind: ImageKind) {
                        update { it.copy(status = "معالجة الصورة $index من $total — ${kind.title}") }
                    }
                    override fun onDecision(index: Int, total: Int, kind: ImageKind, steps: List<String>) {
                        val what = if (steps.isEmpty()) "مفيش حاجة محتاجة تعديل" else steps.joinToString("، ")
                        update { it.copy(decisions = it.decisions + "صورة $index — ${kind.title}: $what") }
                    }
                    override fun onNote(message: String) {
                        update { it.copy(status = message) }
                    }
                    override fun onEnhanced(bitmap: Bitmap) {
                        update { it.copy(enhancedPreview = bitmap) }
                        refreshPreview()
                    }
                    override fun onSaved(index: Int, bitmap: Bitmap) {
                        update { it.copy(progress = index, resultPreview = bitmap) }
                        // Reapply the latest manual settings after the saved-result event.
                        refreshPreview()
                    }
                    override fun onFinished(success: Int, failed: Int, error: String?) {
                        val detail = error?.let { "\nآخر خطأ: $it" }.orEmpty()
                        update {
                            it.copy(
                                busy = false,
                                progress = it.progressMax,
                                status = "اكتملت المعالجة: نجاح $success، فشل $failed$detail"
                            )
                        }
                    }
                },
                firstSourceOverride = if (snapshot.cropApplied) snapshot.cropPreview else null
            )
        }
    }

    override fun onCleared() {
        executor.shutdownNow()
        previewExecutor.shutdownNow()
        currentState.sourcePreview?.let { if (!it.isRecycled) it.recycle() }
        currentState.enhancedPreview?.let { if (!it.isRecycled) it.recycle() }
        currentState.resultPreview?.let { if (!it.isRecycled) it.recycle() }
        super.onCleared()
    }
}
