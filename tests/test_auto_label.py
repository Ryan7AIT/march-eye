import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from auto_label import (
    Detection,
    collect_inference_tiles,
    filter_box_geometry,
    global_nms,
    image_slices,
    load_image,
    translate_and_clip_box,
)


class SlicingTests(unittest.TestCase):
    def test_4080_by_3060_image_has_edge_aligned_48_patch_grid(self) -> None:
        image = Image.new("RGB", (4080, 3060))
        slices = list(image_slices(image, 640, 640, 0.20, 0.20))

        self.assertEqual(len(slices), 48)
        last_slice, offset_x, offset_y = slices[-1]
        self.assertEqual((offset_x, offset_y), (3440, 2420))
        self.assertEqual((offset_x + last_slice.width, offset_y + last_slice.height), (4080, 3060))

    def test_multiscale_tiles_include_full_image_pass(self) -> None:
        image = Image.new("RGB", (2000, 1500), color=(10, 20, 30))
        tiles = collect_inference_tiles(
            image,
            slice_sizes=[640, 1280],
            overlap_height_ratio=0.30,
            overlap_width_ratio=0.30,
            full_image_pass=True,
        )
        self.assertGreater(len(tiles), 1)
        full_tile, offset_x, offset_y = tiles[-1]
        self.assertEqual((offset_x, offset_y), (0, 0))
        self.assertEqual(full_tile.size, image.size)

    def test_slice_box_is_translated_and_clipped_to_global_image(self) -> None:
        self.assertEqual(
            translate_and_clip_box((10, 20, 650, 700), 3440, 2420, 4080, 3060),
            (3450.0, 2440.0, 4080.0, 3060.0),
        )
        self.assertIsNone(translate_and_clip_box((20, 10, 20, 40), 0, 0, 640, 640))

    def test_global_nms_suppresses_overlapping_cross_class_duplicate(self) -> None:
        detections = [
            Detection(0, 0.90, 0, 0, 100, 100),
            Detection(1, 0.80, 5, 5, 105, 105),
            Detection(2, 0.70, 200, 200, 260, 260),
        ]

        kept = global_nms(detections, score_threshold=0.35, nms_iou=0.45)

        self.assertEqual(len(kept), 2)
        self.assertIn(detections[0], kept)
        self.assertIn(detections[2], kept)
        self.assertNotIn(detections[1], kept)

    def test_filter_box_geometry_rejects_oversized_and_strip_boxes(self) -> None:
        detections = [
            Detection(0, 0.9, 0, 0, 100, 300),  # plausible product box (3% of image)
            Detection(0, 0.9, 0, 0, 900, 900),  # covers most of the image
            Detection(0, 0.9, 0, 0, 700, 50),  # long thin shelf-edge strip
        ]

        kept = filter_box_geometry(detections, image_width=1000, image_height=1000, max_area_ratio=0.05)

        self.assertEqual(kept, [detections[0]])

    def test_load_image_applies_exif_transpose_and_strips_metadata(self) -> None:
        # EXIF Orientation=6 means "rotate 90 CW" for display; after transpose
        # a 40x20 stored image becomes 20x40 upright pixels.
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.jpg"
            image = Image.new("RGB", (40, 20), color=(255, 0, 0))
            exif = image.getexif()
            exif[274] = 6  # Orientation
            image.save(path, format="JPEG", exif=exif)

            loaded = load_image(path)
            self.assertEqual(loaded.size, (20, 40))
            self.assertEqual(loaded.info, {})


if __name__ == "__main__":
    unittest.main()
