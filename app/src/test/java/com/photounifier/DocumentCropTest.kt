package com.photounifier

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DocumentCropTest {
    @Test fun acceptsOrderedRectangle() {
        assertTrue(DocumentCrop.isValid(
            listOf(CropPoint(10f, 10f), CropPoint(90f, 10f), CropPoint(90f, 90f), CropPoint(10f, 90f)),
            100, 100
        ))
    }

    @Test fun rejectsCrossedCorners() {
        assertFalse(DocumentCrop.isValid(
            listOf(CropPoint(10f, 10f), CropPoint(90f, 90f), CropPoint(90f, 10f), CropPoint(10f, 90f)),
            100, 100
        ))
    }

    @Test fun rejectsPointsOutsideImage() {
        assertFalse(DocumentCrop.isValid(
            listOf(CropPoint(-1f, 10f), CropPoint(90f, 10f), CropPoint(90f, 90f), CropPoint(10f, 90f)),
            100, 100
        ))
    }

    @Test fun rejectsDegenerateQuadrilateral() {
        assertFalse(DocumentCrop.isValid(
            listOf(CropPoint(10f, 10f), CropPoint(90f, 10f), CropPoint(50f, 10.5f), CropPoint(10f, 11f)),
            100, 100
        ))
    }
}
