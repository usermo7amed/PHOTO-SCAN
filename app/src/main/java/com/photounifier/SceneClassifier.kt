package com.photounifier

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.ImageLabeler
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import java.util.Locale

internal data class LabelSignal(val text: String, val confidence: Float)

data class SceneClassification(
    val kind: ImageKind,
    /** A ranking score derived from matching model labels, not a calibrated probability. */
    val score: Float,
    val matchedLabels: List<String>
)

/**
 * Uses ML Kit's bundled, on-device general image-labeling model. ML Kit's base model
 * does not expose a custom four-class classifier, so its semantic labels are mapped
 * into the app's processing routes. A dataset-trained custom model can replace this
 * mapping when labeled examples are available.
 */
class SceneClassifier : AutoCloseable {
    private val labeler: ImageLabeler =
        ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)

    fun classify(bitmap: Bitmap): SceneClassification {
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val labels = Tasks.await(labeler.process(image))
            classifyLabels(labels.map { LabelSignal(it.text, it.confidence) })
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            SceneClassification(ImageKind.GENERAL, 0f, emptyList())
        } catch (_: Exception) {
            // Classification must not make a readable image unprocessable.
            SceneClassification(ImageKind.GENERAL, 0f, emptyList())
        }
    }

    override fun close() = labeler.close()

    companion object {
        // ML Kit's base model has about 450 labels, and many obvious words (document, text, indoor,
        // outdoor, furniture) are not among them. Words such as paper, receipt, poster, newspaper, sky,
        // roof, couch and kitchen do exist, so they carry the decision; the other entries only help if
        // the model is replaced by a custom one.
        private val keywords: Map<ImageKind, List<Pair<String, Float>>> = mapOf(
            ImageKind.DOCUMENT to listOf(
                "document" to 1.5f,
                "receipt" to 1.25f,
                "invoice" to 1.2f,
                "form" to 1.0f,
                "paper" to 0.8f,
                "worksheet" to 1.0f,
                "handwriting" to 0.7f,
                "printed page" to 0.8f,
                "text" to 0.22f,
                "book" to 0.3f,
                "notebook" to 0.35f,
                // labels that exist in ML Kit's base model
                "newspaper" to 0.9f,
                "menu" to 0.7f,
                "passport" to 0.8f
            ),
            ImageKind.POSTER to listOf(
                "poster" to 1.6f,
                "billboard" to 1.25f,
                "banner" to 1.1f,
                "advertisement" to 1.15f,
                "flyer" to 1.0f,
                "typography" to 0.7f,
                "sign" to 0.55f,
                "text" to 0.12f,
                "comics" to 0.6f
            ),
            ImageKind.INTERIOR to listOf(
                "interior design" to 1.35f,
                "interior" to 1.25f,
                "indoor" to 1.2f,
                "living room" to 1.1f,
                "bedroom" to 1.0f,
                "kitchen" to 1.0f,
                "bathroom" to 0.85f,
                "room" to 0.8f,
                "furniture" to 0.7f,
                "office" to 0.65f,
                "ceiling" to 0.5f,
                "desk" to 0.4f,
                "hallway" to 0.7f,
                "apartment" to 0.6f,
                "couch" to 0.8f,
                "chair" to 0.5f,
                "cabinetry" to 0.7f,
                "curtain" to 0.6f,
                "shelf" to 0.5f,
                "drawer" to 0.5f,
                "countertop" to 0.6f,
                "pillow" to 0.5f,
                "lampshade" to 0.5f,
                "television" to 0.4f,
                "stairs" to 0.4f,
                "whiteboard" to 0.5f
            ),
            ImageKind.EXTERIOR to listOf(
                "outdoor" to 1.25f,
                "landscape" to 0.85f,
                "street" to 0.85f,
                "road" to 0.7f,
                "beach" to 0.75f,
                "mountain" to 0.7f,
                "forest" to 0.65f,
                "garden" to 0.65f,
                "sky" to 0.5f,
                "nature" to 0.5f,
                "tree" to 0.3f,
                "city" to 0.6f,
                "building" to 0.35f,
                "house" to 0.3f,
                "architecture" to 0.25f,
                "skyscraper" to 0.6f,
                "skyline" to 0.7f,
                "roof" to 0.6f,
                "field" to 0.5f,
                "lake" to 0.6f,
                "river" to 0.6f,
                "cliff" to 0.6f,
                "desert" to 0.6f,
                "canyon" to 0.6f,
                "sunset" to 0.5f,
                "park" to 0.5f,
                "bridge" to 0.5f,
                "tower" to 0.4f,
                "factory" to 0.6f,
                "construction" to 0.5f,
                "asphalt" to 0.5f,
                "infrastructure" to 0.5f,
                "brick" to 0.3f,
                "branch" to 0.3f
            )
        )

        internal fun classifyLabels(labels: List<LabelSignal>): SceneClassification {
            if (labels.isEmpty()) return SceneClassification(ImageKind.GENERAL, 0f, emptyList())

            val normalized = labels.map { signal ->
                signal to signal.text.lowercase(Locale.ROOT).trim()
            }
            val scores = keywords.mapValues { (_, entries) ->
                normalized.sumOf { (signal, text) ->
                    val weight = entries.asSequence()
                        .filter { (word, _) -> matches(text, word) }
                        .maxOfOrNull { it.second } ?: 0f
                    (weight * signal.confidence.coerceIn(0f, 1f)).toDouble()
                }
            }
            val ranked = scores.entries.sortedByDescending { it.value }
            val winner = ranked.first()
            val runnerUp = ranked.getOrNull(1)?.value ?: 0.0
            val winnerLabels = normalized.filter { (signal, text) ->
                keywords[winner.key].orEmpty().any { (word, weight) ->
                    weight >= 0.7f && matches(text, word) && signal.confidence > 0f
                }
            }.map { it.first.text }.distinct()

            // Weak or nearly tied evidence is deliberately routed to the conservative
            // general-photo path instead of applying destructive document processing.
            if (winner.value < 0.24 || (winner.value - runnerUp < 0.035)) {
                return SceneClassification(ImageKind.GENERAL, 0f, labels.map { it.text })
            }

            return SceneClassification(
                kind = winner.key,
                score = winner.value.coerceAtMost(1.0).toFloat(),
                matchedLabels = winnerLabels
            )
        }

        private fun matches(label: String, keyword: String): Boolean {
            if (keyword.contains(' ')) return label.contains(keyword)
            return label.split(Regex("[^a-z0-9]+"))
                .any { token -> token == keyword }
        }
    }
}