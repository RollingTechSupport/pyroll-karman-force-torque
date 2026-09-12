from pyroll.core import RollPass, Hook
from pyroll.flat_rolling_slab_model.condition import contact_length_over_mean_thickness
from pyroll.flat_rolling_slab_model.foil_solver import FoilRollingSolver
from pyroll.flat_rolling_slab_model.layer_solver import LayerRollingSolver
from pyroll.flat_rolling_slab_model.karman_mixed_friction_solver import KarmanMixedFrictionSolver

RollPass.karman_solution = Hook[object]()
"""Solution values of the pass (one of KarmanMixedFrictionSolver,
LayerRollingSolver or FoilRollingSolver, chosen automatically - see
karman_solution's hookimpl). All three are elastic-plastic and use Bay &
Wanheim mixed Coulomb/sticking friction, so elastic properties
(elastic_modulus, poissons_ratio) must be set on both Roll and Profile for
every pass - there is no rigid-plastic fallback."""

RollPass.foil_rolling_condition = Hook[bool]()
"""Whether the pass is in the foil-rolling regime, i.e. the contact-length /
mean-thickness ratio (Ld/Hm) is large enough that the single-layer von-Karman
solution is no longer valid and the full elastic foil-rolling model should be
used instead."""

RollPass.foil_rolling_ld_hm_limit = Hook[float]()
"""Contact-length / mean-thickness ratio (Ld/Hm) above which the pass is
considered thin enough to need the full elastic foil-rolling model (default
10.0)."""

RollPass.thick_slab_condition = Hook[bool]()
"""Whether the pass is thick enough that through-thickness deformation can no
longer be treated as homogeneous (contact length / mean thickness < limit),
requiring the multi-layer elastic-plastic model instead of the single-layer
elastic-plastic mixed-friction ("medium") one."""

RollPass.layer_model_ld_hm_limit = Hook[float]()
"""Contact-length / mean-thickness ratio (Ld/Hm) below which the multi-layer
model is used instead of the single-layer one (classical thin-strip/
homogeneous-deformation boundary, default 1.0)."""

RollPass.layer_model_layer_count = Hook[int]()
"""Number of through-thickness layers used by LayerRollingSolver on the
thick-slab path (default 5, matching the reference implementation)."""

RollPass.friction_stiction_coefficient = Hook[float]()
"""Stiction (sticking) friction coefficient feeding the Bay/Wanheim conversion
to an equivalent Coulomb coefficient, used by KarmanMixedFrictionSolver and
LayerRollingSolver's mixed friction laws (default 0.8)."""

RollPass.roll_heat_transfer_coefficient = Hook[float]()
"""Heat transfer coefficient at the roll/material interface, used by
LayerRollingSolver's thermal coupling. Default 6000 W/(m^2 K)."""

RollPass.dynamic_flow_stress_correction_enabled = Hook[bool]()
"""Whether to add Troost's (1967) plastokinetic correction to flow stress
(see dynamic_flow_stress.py), accounting for the strip's inertia at high
rolling speeds. Off by default: the correction is well under 1% of flow
stress for typical cold-rolling speeds and only reaches a low single-digit
percentage at speeds far beyond what this plugin is validated for (see
docs.tex). Requires Profile.density when enabled."""


@RollPass.foil_rolling_ld_hm_limit
def foil_rolling_ld_hm_limit(self: RollPass):
    return 10.0


@RollPass.layer_model_ld_hm_limit
def layer_model_ld_hm_limit(self: RollPass):
    return 1.0


@RollPass.layer_model_layer_count
def layer_model_layer_count(self: RollPass):
    return 5


@RollPass.friction_stiction_coefficient
def friction_stiction_coefficient(self: RollPass):
    return 0.8


@RollPass.roll_heat_transfer_coefficient
def roll_heat_transfer_coefficient(self: RollPass):
    return 6000.0


@RollPass.dynamic_flow_stress_correction_enabled
def dynamic_flow_stress_correction_enabled(self: RollPass):
    return False


@RollPass.foil_rolling_condition
def foil_rolling_condition(self: RollPass):
    ratio = contact_length_over_mean_thickness(self, self.roll.working_radius)
    return ratio > self.foil_rolling_ld_hm_limit


@RollPass.thick_slab_condition
def thick_slab_condition(self: RollPass):
    if self.foil_rolling_condition:
        return False

    ratio = contact_length_over_mean_thickness(self, self.roll.working_radius)
    return ratio < self.layer_model_ld_hm_limit


@RollPass.karman_solution
def karman_solution(self: RollPass):
    if self.foil_rolling_condition:
        return FoilRollingSolver(roll_pass=self)
    if self.thick_slab_condition:
        return LayerRollingSolver(roll_pass=self, layer_count=self.layer_model_layer_count)
    return KarmanMixedFrictionSolver(roll_pass=self)


@RollPass.InProfile.velocity
def profile_entry_velocity(self: RollPass.InProfile):

    return self.roll_pass.karman_solution.entry_velocity


@RollPass.OutProfile.velocity
def velocity(self: RollPass.OutProfile):
    return self.roll_pass.karman_solution.exit_velocity


@RollPass.OutProfile.temperature
def out_profile_temperature(self: RollPass.OutProfile):
    solution = self.roll_pass.karman_solution
    if isinstance(solution, LayerRollingSolver):
        return solution.exit_temperature
    return None


@RollPass.roll_force
def roll_force(self: RollPass):
    return (self.karman_solution.roll_force_per_unit_width * self.roll.contact_area) / self.roll.contact_length


@RollPass.Roll.neutral_point
def neutral_point(self: RollPass.Roll):
    return self.roll_pass.karman_solution.neutral_plane_position


@RollPass.Roll.roll_torque
def roll_torque(self: RollPass.Roll):
    return (
            self.roll_pass.karman_solution.roll_torque_per_unit_width * self.roll_pass.contact_area) / self.contact_length
