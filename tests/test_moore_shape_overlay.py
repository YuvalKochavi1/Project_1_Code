from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from plot_moore_shape_overlay import DATA_DIR, load_normalized_shape


class MooreShapeOverlayTests(unittest.TestCase):
    def load_csv(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.csv"
            path.write_text("index,x,y\n" + rows, encoding="utf-8")
            return load_normalized_shape(path)

    def test_reflects_negative_radius_and_sorts_with_values(self):
        radius, values = self.load_csv("0,-0.2,2\n1,-0.1,3\n2,0,4\n")
        np.testing.assert_allclose(radius, [0, 0.1, 0.2])
        np.testing.assert_allclose(values, [1, 0.75, 0.5])

    def test_missing_axis_uses_nearest_sample_without_shifting_radius(self):
        radius, values = self.load_csv("0,0.2,2\n1,0.1,4\n")
        np.testing.assert_allclose(radius, [0, 0.1, 0.2])
        np.testing.assert_allclose(values, [1, 1, 0.5])

    def test_negative_center_normalizes_to_positive_one(self):
        _, values = self.load_csv("0,0,-4\n1,0.1,-2\n")
        np.testing.assert_allclose(values, [1, 0.5])

    def test_invalid_data_raises(self):
        for rows, message in [
            ("", "no shape samples"),
            ("0,0,0\n", "zero center"),
            ("0,nan,1\n", "finite numbers"),
            ("0,0,inf\n", "finite numbers"),
        ]:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, message):
                    self.load_csv(rows)

    def test_requested_csvs_start_at_one_and_preserve_shape(self):
        for name in ("model.csv", "moore_simulation.csv"):
            with self.subTest(name=name):
                radius, values = load_normalized_shape(DATA_DIR / name)
                self.assertEqual(radius[0], 0)
                self.assertEqual(values[0], 1)
                self.assertTrue(np.all(np.diff(radius) >= 0))
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertLess(values[-1], 1)


if __name__ == "__main__":
    unittest.main()
