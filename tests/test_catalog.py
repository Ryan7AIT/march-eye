import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from enroll import (
    EnrollError,
    add_image_bytes,
    create_sku,
    delete_image,
    delete_sku,
    list_sku_images,
    move_sku,
    sanitize_filename,
    sanitize_sku_id,
)


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (40, 80)) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class CatalogEnrollTests(unittest.TestCase):
    def test_sanitize_sku_and_filename(self) -> None:
        self.assertEqual(sanitize_sku_id("Ramy Orange 1.25L"), "ramy_orange_1.25l")
        self.assertEqual(sanitize_filename("My Photo.PNG"), "My_Photo.png")
        with self.assertRaises(EnrollError):
            sanitize_sku_id("...")

    def test_create_add_delete_and_move(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = create_sku(root, "Ramy Orange", "bottle")
            self.assertEqual(created["sku_id"], "ramy_orange")
            self.assertTrue((root / "bottle" / "ramy_orange").is_dir())

            first = add_image_bytes(root, "ramy_orange", "bottle", "front.jpg", _jpeg_bytes((10, 20, 30)))
            self.assertFalse(first["already_existed"])
            self.assertEqual(first["filename"], "front.jpg")

            dup = add_image_bytes(root, "ramy_orange", "bottle", "front-copy.jpg", _jpeg_bytes((10, 20, 30)))
            self.assertTrue(dup["already_existed"])
            self.assertEqual(dup["filename"], "front.jpg")

            second = add_image_bytes(root, "ramy_orange", "bottle", "side.jpg", _jpeg_bytes((200, 10, 10)))
            self.assertEqual(len(list_sku_images(root / "bottle" / "ramy_orange")), 2)

            moved = move_sku(root, "ramy_orange", "bottle", new_sku_id="ramy_orange_125l", new_packaging_type="carton")
            self.assertTrue(moved["moved"])
            self.assertFalse((root / "bottle" / "ramy_orange").exists())
            self.assertEqual(len(list_sku_images(root / "carton" / "ramy_orange_125l")), 2)
            self.assertTrue(moved["path_map"])

            deleted = delete_image(root, "ramy_orange_125l", "carton", "side.jpg")
            self.assertTrue(deleted["deleted"])
            self.assertEqual(deleted["n_images"], 1)

            with self.assertRaises(EnrollError):
                delete_sku(root, "ramy_orange_125l", "carton", force=False)

            gone = delete_sku(root, "ramy_orange_125l", "carton", force=True)
            self.assertTrue(gone["deleted"])
            self.assertFalse((root / "carton" / "ramy_orange_125l").exists())

    def test_rejects_non_image_bytes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_sku(root, "x", "can")
            with self.assertRaises(EnrollError):
                add_image_bytes(root, "x", "can", "nope.jpg", b"not-an-image")

    def test_create_duplicate_sku_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_sku(root, "same", "bottle")
            with self.assertRaises(EnrollError):
                create_sku(root, "same", "bottle")


if __name__ == "__main__":
    unittest.main()
