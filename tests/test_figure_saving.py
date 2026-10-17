import mmap
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import csv_helpers


class FigureSavingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        plt.figure()
        plt.plot([0, 1], [0, 1])
        self.addCleanup(plt.close, "all")

    def assert_valid_png(self, path):
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            image.verify()

    def test_creates_parent_and_replaces_existing_image(self):
        path = self.root / "nested" / "figure.png"
        csv_helpers.save_figure_to_path(path, dpi=80, bbox_inches="tight")
        self.assert_valid_png(path)
        plt.plot([0, 1], [1, 0])
        csv_helpers.save_figure_to_path(path, dpi=120, bbox_inches="tight")
        self.assert_valid_png(path)
        self.assertEqual(list(path.parent.iterdir()), [path])

    @unittest.skipUnless(os.name == "nt", "Windows mapped-file regression")
    def test_locked_png_reports_action_and_saves_after_mapping_closes(self):
        path = self.root / "figure.png"
        plt.savefig(path)
        old_contents = path.read_bytes()
        with path.open("rb") as source:
            mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        with mapped:
            with self.assertRaises(OSError) as error:
                plt.savefig(path)
            self.assertEqual(error.exception.errno, 22)
            with self.assertRaisesRegex(PermissionError, "Close any image preview"):
                csv_helpers.save_figure_to_path(path, dpi=120)
            self.assertEqual(path.read_bytes(), old_contents)
            self.assertEqual(mapped[:], old_contents)
            self.assertEqual(list(self.root.iterdir()), [path])
        csv_helpers.save_figure_to_path(path, dpi=120)
        self.assert_valid_png(path)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_render_failure_preserves_destination_and_cleans_temporary_file(self):
        path = self.root / "figure.png"
        path.write_bytes(b"previous image")
        with patch.object(csv_helpers.plt, "savefig", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                csv_helpers.save_figure_to_path(path)
        self.assertEqual(path.read_bytes(), b"previous image")
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_replacement_failure_propagates_and_cleans_temporary_file(self):
        path = self.root / "figure.png"
        path.write_bytes(b"previous image")
        with patch.object(csv_helpers.os, "replace", side_effect=PermissionError("locked")):
            with self.assertRaisesRegex(PermissionError, "Close any image preview") as error:
                csv_helpers.save_figure_to_path(path)
        self.assertIsInstance(error.exception.__cause__, PermissionError)
        self.assertEqual(path.read_bytes(), b"previous image")
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_extensionless_filename_uses_default_format(self):
        with matplotlib.rc_context({"savefig.format": "png"}):
            csv_helpers.save_figure_to_path(self.root / "figure")
        self.assert_valid_png(self.root / "figure.png")
        self.assertEqual(list(self.root.iterdir()), [self.root / "figure.png"])

    def test_dataframe_csv_replaces_existing_file(self):
        path = self.root / "data.csv"
        path.write_text("value\n1\n", encoding="utf-8")
        csv_helpers.save_dataframe_to_csv(pd.DataFrame({"value": [2]}), path, index=False)
        self.assertEqual(pd.read_csv(path)["value"].tolist(), [2])
        self.assertEqual(list(self.root.iterdir()), [path])

    @unittest.skipUnless(os.name == "nt", "Windows mapped-file regression")
    def test_locked_csv_preserves_destination_and_cleans_temporary_file(self):
        path = self.root / "data.csv"
        path.write_text("value\n1\n", encoding="utf-8")
        old_contents = path.read_bytes()
        with path.open("rb") as source:
            mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        with mapped:
            with self.assertRaisesRegex(PermissionError, "Close any CSV preview") as error:
                csv_helpers.save_dataframe_to_csv(pd.DataFrame({"value": [2]}), path, index=False)
            self.assertIsInstance(error.exception.__cause__, PermissionError)
            self.assertEqual(path.read_bytes(), old_contents)
            self.assertEqual(mapped[:], old_contents)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_explicit_format_does_not_append_extension(self):
        path = self.root / "figure"
        csv_helpers.save_figure_to_path(path, format="png")
        self.assert_valid_png(path)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_model_save_uses_existing_directory_and_options(self):
        with patch.object(csv_helpers, "_figures_dir", return_value=self.root):
            csv_helpers.save_figure("figure.png", model1_5=True, dpi=120, bbox_inches="tight")
        self.assert_valid_png(self.root / "1.5 model" / "figure.png")


if __name__ == "__main__":
    unittest.main()
