from wall_loss_model import WallLossModel
from ablation_model import AblationModel
from albedo_model import AlbedoModel
import parameters as parameters_module
import wall_loss_model as wall_loss_model_module


class AnalyticalWavefrontSolver:
    """
    Class-based façade for analytical wave-front models.

    Keeps equations and behavior identical to the legacy function API,
    while offering an object-oriented entry point.
    """

    def __init__(self, no_marshak_fn, march_fn, wall_model=None, ablation_model=None, albedo_model=None, fluence_fn=None):
        self.no_marshak_fn = no_marshak_fn
        self.march_fn = march_fn
        self.wall_model = wall_model or WallLossModel()
        self.ablation_model = ablation_model or AblationModel()
        self.albedo_model = albedo_model or AlbedoModel()
        self.fluence_fn = fluence_fn

    def analytic_wave_front_marshak_fluence(self, times_to_store, *, use_seconds=True, k=10):
        if self.fluence_fn is None:
            raise NotImplementedError("fluence_fn was not provided to AnalyticalWavefrontSolver")
        return self.fluence_fn(times_to_store, use_seconds=use_seconds, k=k)

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
        k=10,
        R_average_for_lambda_geom=False,
        g_gold_scale=1.0,
    ):
        if wall_material == 'Be':
            vary_rho = False
            if mode == "marshak_ablation":
                mode = "marshak_wall_loss"

        original_g_gold = parameters_module.g_gold
        scaled_g_gold = original_g_gold * float(g_gold_scale)
        parameters_module.g_gold = scaled_g_gold
        wall_loss_model_module.g_gold = scaled_g_gold
        try:
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
                )
            if mode == "marshak_fluence":
                return self.analytic_wave_front_marshak_fluence(times_to_store, use_seconds=use_seconds, k=k)
            raise ValueError(f"Unknown mode: {mode}")
        finally:
            parameters_module.g_gold = original_g_gold
            wall_loss_model_module.g_gold = original_g_gold
