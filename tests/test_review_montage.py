"""Review mosaic retains coordinate context without multiplying image attachments."""

import unittest
import numpy as np
from vive_correct.review.review_montage import crop_bounds, compose_review_image


class MontageTests(unittest.TestCase):
    def test_crops_include_visible_axis_endpoints_and_clip_at_frame(self):
        quad = [[100, 100], [120, 100], [120, 120], [100, 120]]
        projection = dict(origin=[110, 110], endpoints=[[450, 250], [-100, 110], None])
        self.assertEqual(crop_bounds((300, 500, 3), quad, [projection]), [0, 60, 491, 291])

    def test_missing_pnp_and_shared_crop_metadata(self):
        raw = np.zeros((300, 500, 3), np.uint8)
        row = dict(
            side="left",
            frame_index=8,
            original=raw,
            current=raw.copy(),
            pnp=None,
            quad=[[100, 100], [120, 100], [120, 120], [100, 120]],
            pnp_axis_convention="wrist_y_normal",
        )
        image, info = compose_review_image([row])
        self.assertEqual(image.shape, (556, 1600, 3))
        self.assertEqual(info["rows"][0]["original_size_px"], [500, 300])
        self.assertFalse(info["rows"][0]["pnp_available"])
        self.assertEqual(info["rows"][0]["pnp_axis_convention"], "wrist_y_normal")
