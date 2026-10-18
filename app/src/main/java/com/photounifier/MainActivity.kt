package com.photounifier

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.layout.ContentScale
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.runtime.livedata.observeAsState
import com.photounifier.ui.PhotoUnifierTheme
import org.opencv.android.OpenCVLoader
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    private val viewModel: PhotoViewModel by viewModels()
    private var openCvReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        openCvReady = OpenCVLoader.initLocal()
        setContent {
            PhotoUnifierTheme {
                androidx.compose.runtime.CompositionLocalProvider(
                    LocalLayoutDirection provides LayoutDirection.Rtl
                ) {
                    val state by viewModel.state.observeAsState(PhotoUiState())
                    val picker = rememberLauncherForActivityResult(
                        ActivityResultContracts.GetMultipleContents()
                    ) { uris -> if (uris.isNotEmpty()) viewModel.selectImages(uris) }
                    val referencePicker = rememberLauncherForActivityResult(
                        ActivityResultContracts.GetContent()
                    ) { uri -> if (uri != null) viewModel.setReference(uri) }
                    PhotoUnifierScreen(
                        state = state,
                        openCvReady = openCvReady,
                        onPickImages = { picker.launch("image/*") },
                        onModeChange = viewModel::setMode,
                        onUnifyChange = viewModel::setUnify,
                        onPickReference = { referencePicker.launch("image/*") },
                        onClearReference = { viewModel.setReference(null) },
                        onProcess = { viewModel.process(openCvReady) },
                        onBrightnessChange = viewModel::setBrightness,
                        onContrastChange = viewModel::setContrast,
                        onOpenCrop = { viewModel.openCropEditor(openCvReady) },
                        onMoveCropCorner = viewModel::moveCropCorner,
                        onUndoCrop = viewModel::undoCrop,
                        onResetCrop = viewModel::resetCrop,
                        onCancelCrop = viewModel::cancelCrop,
                        onApplyCrop = viewModel::applyCrop
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PhotoUnifierScreen(
    state: PhotoUiState,
    openCvReady: Boolean,
    onPickImages: () -> Unit,
    onModeChange: (ImageKind?) -> Unit,
    onUnifyChange: (Int) -> Unit,
    onPickReference: () -> Unit,
    onClearReference: () -> Unit,
    onProcess: () -> Unit,
    onBrightnessChange: (Int) -> Unit,
    onContrastChange: (Int) -> Unit,
    onOpenCrop: () -> Unit,
    onMoveCropCorner: (Int, Float, Float) -> Unit,
    onUndoCrop: () -> Unit,
    onResetCrop: () -> Unit,
    onCancelCrop: () -> Unit,
    onApplyCrop: () -> Unit
) {
    Scaffold(topBar = {
        TopAppBar(
            title = {
                Column {
                    Text("مُوحِّد الصور", fontWeight = FontWeight.Bold)
                    Text(
                        "تحسين ذكي وتحكم يدوي",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        )
    }) { padding ->
        Column(
            modifier = Modifier.fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        "خلّي صورك أوضح ومتوازنة",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                    Text(
                        "حلّل الصور وحسّن الإضاءة والتباين، وراجع النتيجة قبل الحفظ.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(onClick = onPickImages, enabled = !state.busy) {
                            Text("اختيار الصور")
                        }
                        OutlinedButton(
                            onClick = onProcess,
                            enabled = !state.busy && openCvReady && state.selected.isNotEmpty()
                        ) {
                            Text(if (state.busy) "جاري التحسين..." else "بدء التحسين")
                        }
                    }
                }
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    SectionHeading("معاينة الصورة", "قارن الأصل بالنتيجة المحسّنة")
                    OutlinedButton(onClick = onOpenCrop, enabled = !state.busy && state.cropSource != null && openCvReady) {
                        Text("تحديد زوايا الورقة")
                    }
                    if (state.cropOpen) {
                        Text("اسحب الزوايا لتعديل حدود الورقة")
                        CropCornerEditor(state.cropSource, state.cropCorners, onMoveCropCorner)
                        PreviewPanel("معاينة القص", state.cropPreview, Modifier.fillMaxWidth())
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = onUndoCrop, enabled = state.cropUndoStack.isNotEmpty()) { Text("تراجع") }
                            OutlinedButton(onClick = onResetCrop) { Text("إعادة ضبط") }
                            OutlinedButton(onClick = onCancelCrop) { Text("إلغاء") }
                            Button(onClick = onApplyCrop, enabled = state.cropPreview != null) { Text("تطبيق القص") }
                        }
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        PreviewPanel("الأصلية", state.sourcePreview, Modifier.weight(1f))
                        PreviewPanel(
                            "بعد التحسين",
                            state.resultPreview ?: state.enhancedPreview,
                            Modifier.weight(1f)
                        )
                    }
                    Text(
                        if (state.selected.isEmpty()) "اختار صورة أو أكتر علشان تبدأ."
                        else "تم اختيار ${state.selected.size} صورة",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionHeading("نوع المعالجة", "سيبها تلقائي، أو اختار النوع بنفسك لو التطبيق غلط")
                    ModeRow(state.mode, !state.busy, listOf(null to "تلقائي", ImageKind.DOCUMENT to "مستند", ImageKind.POSTER to "بوستر"), onModeChange)
                    ModeRow(state.mode, !state.busy, listOf(ImageKind.INTERIOR to "داخلي", ImageKind.EXTERIOR to "خارجي", ImageKind.GENERAL to "عامة"), onModeChange)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("توحيد الشكل بين الصور", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                        ValuePill(if (state.unifyPercent == 0) "مقفول" else "${state.unifyPercent}%")
                    }
                    Slider(
                        value = state.unifyPercent.toFloat(),
                        onValueChange = { onUnifyChange(it.roundToInt()) },
                        valueRange = 0f..100f,
                        enabled = !state.busy
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        OutlinedButton(onClick = onPickReference, enabled = !state.busy && state.unifyPercent > 0) {
                            Text(if (state.referenceUri == null) "اختيار صورة مرجعية" else "تغيير الصورة المرجعية")
                        }
                        if (state.referenceUri != null) {
                            OutlinedButton(onClick = onClearReference, enabled = !state.busy) { Text("إلغاء المرجع") }
                        }
                    }
                    Text(
                        if (state.referenceUri == null) "من غير مرجع، الصور بتتقرّب من الشكل المتوسط للمجموعة. المستندات والبوسترات بتفضل زي ما هي."
                        else "الصور هتتقرّب من شكل الصورة المرجعية.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SectionHeading("التحكم اليدوي", "اضبط السطوع والتباين حسب الصورة")
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("السطوع", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                        ValuePill(if (state.brightness > 0) "+${state.brightness}" else "${state.brightness}")
                    }
                    Slider(
                        value = state.brightness.toFloat(),
                        onValueChange = { onBrightnessChange(it.roundToInt()) },
                        valueRange = -50f..50f
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("التباين", modifier = Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                        ValuePill("${state.contrastPercent}%")
                    }
                    Slider(
                        value = state.contrastPercent.toFloat(),
                        onValueChange = { onContrastChange(it.roundToInt()) },
                        valueRange = 50f..200f
                    )
                    Text(
                        if (state.enhancedPreview == null)
                            "اضبط القيم قبل بدء المعالجة؛ هتتطبق على الملفات المحفوظة."
                        else
                            "التغييرات بعد انتهاء المعالجة بتحدّث المعاينة؛ ابدأ المعالجة من جديد لحفظ نسخة بالقيم الجديدة.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionHeading("حالة المعالجة", "متابعة مباشرة للصور")
                    Text(state.status, style = MaterialTheme.typography.bodyMedium)
                    if (state.decisions.isNotEmpty()) {
                        Text("اللي اتعمل في كل صورة", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                        state.decisions.forEach {
                            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                    if (state.busy) {
                        LinearProgressIndicator(
                            progress = state.progress.toFloat() / state.progressMax.coerceAtLeast(1),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Text(
                            "${state.progress} من ${state.progressMax}",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (!openCvReady) {
                        Text(
                            "مكتبة OpenCV مش جاهزة؛ المعالجة متوقفة لحد ما تتوفر.",
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                    Text(
                        "الصور الأصلية مش بتتغير. النسخ المحسّنة بتتحفظ في Pictures/PhotoUnifier.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ModeRow(
    selected: ImageKind?,
    enabled: Boolean,
    options: List<Pair<ImageKind?, String>>,
    onSelect: (ImageKind?) -> Unit
) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        options.forEach { (kind, label) ->
            if (kind == selected) {
                Button(onClick = { onSelect(kind) }, enabled = enabled, modifier = Modifier.weight(1f)) { Text(label) }
            } else {
                OutlinedButton(onClick = { onSelect(kind) }, enabled = enabled, modifier = Modifier.weight(1f)) { Text(label) }
            }
        }
    }
}

@Composable
private fun SectionHeading(title: String, subtitle: String) {
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        Text(
            subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun PreviewPanel(
    title: String,
    bitmap: android.graphics.Bitmap?,
    modifier: Modifier = Modifier
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        Box(
            modifier = Modifier.fillMaxWidth().height(190.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null && !bitmap.isRecycled) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = title,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Surface(
                        modifier = Modifier.size(42.dp),
                        shape = CircleShape,
                        color = MaterialTheme.colorScheme.surface
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Text("＋", fontSize = 22.sp, color = MaterialTheme.colorScheme.primary)
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                    Text("المعاينة هتظهر هنا", style = MaterialTheme.typography.labelSmall, textAlign = TextAlign.Center)
                }
            }
        }
    }
}


@Composable
private fun CropCornerEditor(
    bitmap: android.graphics.Bitmap?,
    corners: List<CropPoint>,
    onMoveCorner: (Int, Float, Float) -> Unit
) {
    if (bitmap == null || bitmap.isRecycled || corners.size != 4) return
    val latestCorners by rememberUpdatedState(corners)
    BoxWithConstraints(
        modifier = Modifier.fillMaxWidth().height(320.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(Color(0xFF16191D))
    ) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "الصورة مع نقاط تحديد الزوايا",
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit
        )
        Canvas(modifier = Modifier.fillMaxSize().pointerInput(bitmap) {
            var activeCorner = -1
            var dragPosition = Offset.Zero
            detectDragGestures(
                onDragStart = { position ->
                    val scale = minOf(size.width / bitmap.width.toFloat(), size.height / bitmap.height.toFloat())
                    val left = (size.width - bitmap.width * scale) / 2f
                    val top = (size.height - bitmap.height * scale) / 2f
                    val screenPoints = latestCorners.map { Offset(left + it.x * scale, top + it.y * scale) }
                    activeCorner = screenPoints.indices.minByOrNull { (screenPoints[it] - position).getDistance() } ?: -1
                    if (activeCorner >= 0 && (screenPoints[activeCorner] - position).getDistance() > 58f) activeCorner = -1
                    dragPosition = position
                },
                onDrag = { change, amount ->
                    if (activeCorner >= 0) {
                        change.consume()
                        dragPosition += amount
                        val scale = minOf(size.width / bitmap.width.toFloat(), size.height / bitmap.height.toFloat())
                        val left = (size.width - bitmap.width * scale) / 2f
                        val top = (size.height - bitmap.height * scale) / 2f
                        val x = ((dragPosition.x - left) / scale).coerceIn(0f, (bitmap.width - 1).toFloat())
                        val y = ((dragPosition.y - top) / scale).coerceIn(0f, (bitmap.height - 1).toFloat())
                        onMoveCorner(activeCorner, x, y)
                    }
                },
                onDragEnd = { activeCorner = -1 },
                onDragCancel = { activeCorner = -1 }
            )
        }) {
            val scale = minOf(size.width / bitmap.width.toFloat(), size.height / bitmap.height.toFloat())
            val left = (size.width - bitmap.width * scale) / 2f
            val top = (size.height - bitmap.height * scale) / 2f
            val screenPoints = corners.map { Offset(left + it.x * scale, top + it.y * scale) }
            for (i in 0..3) drawLine(Color.Cyan, screenPoints[i], screenPoints[(i + 1) % 4], strokeWidth = 4f)
            screenPoints.forEach { point ->
                drawCircle(Color.White, radius = 17f, center = point)
                drawCircle(Color(0xFF00A8C6), radius = 12f, center = point)
            }
        }
    }
}

@Composable
private fun ValuePill(value: String) {
    Surface(shape = RoundedCornerShape(10.dp), color = MaterialTheme.colorScheme.secondaryContainer) {
        Text(
            value,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            color = MaterialTheme.colorScheme.onSecondaryContainer,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold
        )
    }
}
