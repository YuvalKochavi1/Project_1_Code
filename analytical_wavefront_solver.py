from wall_loss_model import WallLossModel
from ablation_model import AblationModel
from albedo_model import AlbedoModel


class AnalyticalWavefrontSolver:
    """
    Class-based façade for analytical wave-front models.

    Keeps equations and behavior identical to the legacy function API,
    while offering an object-oriented entry point.
    """

    def __init__(self, no_marshak_fn, march_fn, wall_model=None, ablation_model=None, albedo_model=None):
        self.no_marshak_fn = no_marshak_fn
        self.march_fn = march_fn
        self.wall_model = wall_model or WallLossModel()
        self.ablation_model = ablation_model or AblationModel()
        self.albedo_model = albedo_model or AlbedoModel()

    def analytic_wave_front_no_marshak(self, times_to_store, *, use_seconds=True, lam_eff=False, power=2):
        return self.no_marshak_fn(
            times_to_store,
            use_seconds=use_seconds,
            lam_eff=lam_eff,
            power=power,
        )

    def marshak_appendixA_march(
        self,
        times_to_store,
        *,
        use_seconds=True,
        wall_loss=False,
        ablation=False,
        vary_rho=False,
        flat_top_profile=False,
        wall_material='Gold',
        lam_eff=False,
        power=2,
        R_average_for_lambda_geom=True,
        good_way=False,
    ):
        return self.march_fn(
            times_to_store,
            use_seconds=use_seconds,
            wall_loss=wall_loss,
            ablation=ablation,
            vary_rho=vary_rho,
            flat_top_profile=flat_top_profile,
            wall_material=wall_material,
            lam_eff=lam_eff,
            power=power,
            R_average_for_lambda_geom=R_average_for_lambda_geom,
            good_way=good_way,
        )

    def analytic_wave_front_marshak(self, times_to_store, *, use_seconds=True, wall_material='Gold', lam_eff=False, power=2):
        return self.marshak_appendixA_march(
            times_to_store,
            use_seconds=use_seconds,
            wall_loss=False,
            ablation=False,
            vary_rho=False,
            flat_top_profile=True,
            wall_material=wall_material,
            lam_eff=lam_eff,
            power=power,
        )

    def analytic_wave_front_marshak_gold_loss(self, times_to_store, *, use_seconds=True, wall_material='Gold', lam_eff=False, power=2):
        return self.marshak_appendixA_march(
            times_to_store,
            use_seconds=use_seconds,
            wall_loss=True,
            ablation=False,
            vary_rho=False,
            flat_top_profile=True,
            wall_material=wall_material,
            lam_eff=lam_eff,
            power=power,
        )

    def analytic_wave_front_marshak_ablation(
        self,
        times_to_store,
        *,
        use_seconds=True,
        vary_rho=False,
        wall_material='Gold',
        lam_eff=False,
        power=2,
        R_average_for_lambda_geom=False,
        good_way=False,
    ):
        return self.marshak_appendixA_march(
            times_to_store,
            use_seconds=use_seconds,
            wall_loss=True,
            ablation=True,
            vary_rho=vary_rho,
            flat_top_profile=True,
            wall_material=wall_material,
            lam_eff=lam_eff,
            power=power,
            R_average_for_lambda_geom=R_average_for_lambda_geom,
            good_way=good_way,
        )

    def analytic_wave_front_dispatch(
        self,
        times_to_store,
        *,
        use_seconds=True,
        mode="no_marshak",
        vary_rho=False,
        wall_material='Gold',
        lam_eff=False,
        power=2,
        R_average_for_lambda_geom=False,
        good_way=True,
    ):
        if mode == "no_marshak":
            return self.analytic_wave_front_no_marshak(times_to_store, use_seconds=use_seconds, lam_eff=lam_eff, power=power)
        if mode == "marshak":
            return self.analytic_wave_front_marshak(times_to_store, use_seconds=use_seconds, wall_material=wall_material, lam_eff=lam_eff, power=power)
        if mode == "marshak_wall_loss":
            return self.analytic_wave_front_marshak_gold_loss(times_to_store, use_seconds=use_seconds, wall_material=wall_material, lam_eff=lam_eff, power=power)
        if mode == "marshak_ablation":
            return self.analytic_wave_front_marshak_ablation(
                times_to_store,
                use_seconds=use_seconds,
                vary_rho=vary_rho,
                wall_material=wall_material,
                lam_eff=lam_eff,
                power=power,
                R_average_for_lambda_geom=R_average_for_lambda_geom,
                good_way=good_way,
            )
        raise ValueError(f"Unknown mode: {mode}")
