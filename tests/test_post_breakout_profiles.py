import unittest
from unittest.mock import patch

import numpy as np

import radiation_flux


class PostBreakoutProfileTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(radiation_flux.plt.close, "all")
        self.times = np.array([0.0, 1.0, 1.2, 1.5, 2.0])
        self.snapshot = {
            "r_grid": np.array([0.0, 0.08]),
            "z_F_radial": np.array([0.15, 0.1]),
        }
        self.result = (
            np.array([0.0, 0.1, 0.12, 0.15, 0.2]),
            np.ones(5), None, None, None,
            {time: self.snapshot for time in self.times},
        )

    def test_delay_changes_selected_snapshot(self):
        with patch.object(radiation_flux, "compute_flux_curvature_at_position", return_value={}), \
                patch.object(radiation_flux, "Material", "Ta2O5"):
            early = radiation_flux._post_breakout_profile(
                self.result, self.times, 1.0, 0.2, "Gold", True
            )
            late = radiation_flux._post_breakout_profile(
                self.result, self.times, 1.0, 0.5, "Gold", True
            )
        self.assertAlmostEqual(early[1], 1.2)
        self.assertAlmostEqual(late[1], 1.5)

    def test_out_of_range_delay_is_not_silently_clamped(self):
        with self.assertRaisesRegex(ValueError, "Extend times_to_store"):
            radiation_flux._post_breakout_profile(
                self.result, self.times, 1.0, 2.0, "Gold", True
            )

    def test_sio2_plot_uses_recomputed_overlay_with_requested_options(self):
        options = dict(
            mode="marshak_ablation", wall_material="Gold", use_seconds=True,
            vary_rho=True, lam_eff=True, power=1.2,
        )
        ds = {"r_grid": np.array([0.0, 0.08]), "flux_radial": np.array([2.0, 0.0])}
        with patch.object(radiation_flux, "Material", "SiO2"), \
                patch.object(radiation_flux, "analytic_wave_front_dispatch", return_value=self.result), \
                patch.object(radiation_flux, "_post_breakout_profile", return_value=(1.0, 1.2, ds)), \
                patch.object(radiation_flux, "_recompute_ta2o5_profile",
                             return_value=(np.array([0.0, 0.8]), np.array([1.0, 0.2]))) as recompute, \
                patch.object(radiation_flux, "save_figure"), \
                patch.object(radiation_flux.plt, "tight_layout"), \
                patch.object(radiation_flux.plt, "legend"), \
                patch.object(radiation_flux.plt, "plot") as plot:
            radiation_flux.plot_flux_curvature_post_breakout(
                self.times, detector_positions_mm=[1.0], delay_ns=0.2,
                show_plot=False, **options,
            )
        recompute.assert_called_once_with(self.times, delay_ns=0.2, **options)
        overlay_calls = [
            call for call in plot.call_args_list
            if call.kwargs.get("label", "").startswith("Model Ta")
        ]
        self.assertEqual(len(overlay_calls), 1)
        np.testing.assert_allclose(overlay_calls[0].args[0], [-0.8, 0.0, 0.8])
        np.testing.assert_allclose(overlay_calls[0].args[1], [0.2, 1.0, 0.2])


if __name__ == "__main__":
    unittest.main()
