import unittest

import cv2
import numpy as np

from tools.prototype.enhance import (
    analysis_copy, describe, detect_mode, enhance, find_quad, look_stats, match_look, measure, median_look,
    plan_for, rectify_page, remove_paper_shadows, vibrance_table,
)


class EnhancementTests(unittest.TestCase):
    def test_paper_shadow_correction_reduces_background_variation(self):
        height, width = 320, 400
        background = np.tile(np.linspace(135, 225, width, dtype=np.float32), (height, 1))
        # Simulate dark text without making it part of the estimated large-scale field.
        luminance = np.clip(background, 0, 255).astype(np.uint8)
        cv2.putText(luminance, "Text 123", (45, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 35, 3)
        corrected = remove_paper_shadows(luminance)
        original_paper = luminance[20:45, :]
        corrected_paper = corrected[20:45, :]
        self.assertLess(float(corrected_paper.std()), float(original_paper.std()))
        self.assertGreater(int(corrected[165, 85]), 0)

    def test_perspective_rectification_finds_skewed_page(self):
        canvas = np.full((300, 320, 3), 80, dtype=np.uint8)
        corners = np.array([[75, 35], [245, 52], [230, 265], [58, 245]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, corners, (235, 235, 235))
        rectified = rectify_page(canvas)
        self.assertNotEqual(rectified.shape[:2], canvas.shape[:2])
        self.assertGreater(rectified.shape[0], 150)
        self.assertGreater(rectified.shape[1], 150)

    def test_mode_routes_return_valid_images(self):
        image = np.full((192, 240, 3), 120, dtype=np.uint8)
        cv2.rectangle(image, (25, 20), (215, 172), (190, 200, 220), -1)
        cv2.putText(image, "TEST", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 45, 55), 2)
        for mode in ("auto", "interior", "exterior", "document", "poster"):
            with self.subTest(mode=mode):
                result = enhance(image, mode)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.ndim, 3)
                self.assertEqual(result.shape[2], 3)
                self.assertGreater(result.size, 0)


class DecisionTests(unittest.TestCase):
    @staticmethod
    def _rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def test_a_balanced_photo_gets_no_tone_correction(self):
        rng = np.random.default_rng(1)
        gray = np.clip(rng.normal(128, 40, (240, 320)), 0, 255).astype(np.uint8)
        image = cv2.merge([gray, gray, gray])
        plan = plan_for("auto", measure(self._rgb(image)))
        self.assertEqual(plan["alpha_high"], 0.0)
        self.assertEqual(plan["alpha_low"], 0.0)
        self.assertEqual(plan["white_balance"], False)

    def test_burnt_sky_is_pulled_down_within_the_ceiling(self):
        image = np.full((240, 320, 3), 252, dtype=np.uint8)
        image[150:, :] = (90, 100, 110)
        plan = plan_for("exterior", measure(self._rgb(image)))
        self.assertGreater(plan["alpha_high"], 0.4)
        self.assertLessEqual(plan["alpha_high"], 0.55)
        self.assertIn("خفض السطوع العالي", describe(plan))

    def test_exterior_never_gets_white_balance_and_poster_never_vibrance(self):
        warm = np.full((240, 320, 3), (150, 180, 210), dtype=np.uint8)  # BGR: a warm grey wall
        metrics = measure(self._rgb(warm))
        self.assertFalse(plan_for("exterior", metrics)["white_balance"])
        self.assertEqual(plan_for("poster", {**metrics, "saturation": 0.05})["vibrance"], 0.0)

    def test_vibrance_helps_dull_colours_more_than_vivid_ones(self):
        table = vibrance_table(0.4)
        self.assertEqual(int(table[0]), 0)
        self.assertEqual(int(table[255]), 255)
        self.assertGreater(int(table[40]) - 40, int(table[220]) - 220)
        self.assertTrue(np.all(np.diff(table.astype(int)) >= 0))

    def test_a_plain_photo_has_no_page_but_a_clear_page_is_found_strictly(self):
        rng = np.random.default_rng(2)
        photo = np.clip(rng.normal(120, 40, (300, 400, 3)), 0, 255).astype(np.uint8)
        self.assertIsNone(find_quad(self._rgb(photo), strict=True))
        canvas = np.full((300, 400, 3), 70, dtype=np.uint8)
        cv2.fillConvexPoly(canvas, np.array([[80, 30], [320, 40], [310, 270], [70, 262]], np.int32), (235, 235, 235))
        self.assertIsNotNone(find_quad(self._rgb(canvas), strict=True))
        self.assertEqual(detect_mode(canvas), "document")
        self.assertEqual(detect_mode(photo), "auto")

    def test_look_matching_moves_a_photo_towards_the_target_only_as_far_as_asked(self):
        dark = np.full((120, 160, 3), 60, dtype=np.uint8)
        light = np.full((120, 160, 3), 190, dtype=np.uint8)
        target = median_look([look_stats(self._rgb(dark)), look_stats(self._rgb(light))])
        untouched = match_look(self._rgb(dark), target, 0.0)
        half = match_look(self._rgb(dark), target, 0.5)
        full = match_look(self._rgb(dark), target, 1.0)
        mean = lambda image: float(image.mean())
        self.assertEqual(mean(untouched), mean(self._rgb(dark)))
        self.assertGreater(mean(half), mean(untouched))
        self.assertGreater(mean(full), mean(half))


if __name__ == "__main__":
    unittest.main()