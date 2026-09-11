from pyroll.core import RollPass, Hook
from pyroll.karman_force_torque.condition import contact_length_over_mean_thickness, hitchcock_radius_ratio
from pyroll.karman_force_torque.foil_solver import FoilRollingSolver
from pyroll.karman_force_torque.karman_solver import KarmanSolver
from pyroll.karman_force_torque.layer_solver import LayerRollingSolver
from pyroll.karman_force_torque.karman_mixed_friction_solver import KarmanMixedFrictionSolver

RollPass.karman_solution = Hook[object]()
"""Solution values of the pass (one of KarmanSolver, KarmanMixedFrictionSolver,
LayerRollingSolver or FoilRollingSolver, chosen automatically - see
karman_solution's hookimpl)."""

RollPass.foil_rolling_condition = Hook[bool]()
"""Whether the pass is in the foil-rolling regime, i.e. elastic roll flattening
is severe enough that the rigid-roll von-Karman solution is no longer valid and
the full elastic foil-rolling model should be used instead."""

RollPass.foil_rolling_hitchcock_limit = Hook[float]()
"""Hitchcock flattened-to-nominal radius ratio (r'/r) above which the rigid-roll
model is considered invalid (Mauk & Overhagen 2013, p. 12: r'/r >= 2)."""

RollPass.thick_slab_condition = Hook[bool]()
"""Whether the pass is thick enough that through-thickness deformation can no
longer be treated as homogeneous (contact length / mean thickness < limit),
requiring the multi-layer elastic-plastic model instead of the rigid-plastic
mixed-friction ("medium") one."""

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


def _elastic_properties_available(self: RollPass) -> bool:
    # Elastic properties (roll & profile) are only needed for the Hitchcock-
    # ratio/Ld-Hm checks and for the elastic-plastic solvers; passes that
    # never set them clearly aren't meant to opt into these effects, so fall
    # back to the traditional rigid-roll model instead of failing hard.
    try:
        self.roll.elastic_modulus
        self.roll.poissons_ratio
        self.in_profile.elastic_modulus
        self.in_profile.poissons_ratio
    except AttributeError:
        return False
    return True


@RollPass.foil_rolling_hitchcock_limit
def foil_rolling_hitchcock_limit(self: RollPass):
    return 2.0


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


@RollPass.foil_rolling_condition
def foil_rolling_condition(self: RollPass):
    if not _elastic_properties_available(self):
        return False

    flat_solution = KarmanSolver(roll_pass=self)
    ratio = hitchcock_radius_ratio(self, flat_solution.roll_force_per_unit_width)
    return ratio >= self.foil_rolling_hitchcock_limit


@RollPass.thick_slab_condition
def thick_slab_condition(self: RollPass):
    if not _elastic_properties_available(self):
        return False
    if self.foil_rolling_condition:
        return False

    ratio = contact_length_over_mean_thickness(self, self.roll.working_radius)
    return ratio < self.layer_model_ld_hm_limit


@RollPass.karman_solution
def karman_solution(self: RollPass):
    if not _elastic_properties_available(self):
        return KarmanSolver(roll_pass=self)
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
