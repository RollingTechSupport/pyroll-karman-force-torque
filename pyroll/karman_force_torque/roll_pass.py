from pyroll.core import RollPass, Hook
from pyroll.karman_force_torque.condition import hitchcock_radius_ratio
from pyroll.karman_force_torque.foil_solver import FoilRollingSolver
from pyroll.karman_force_torque.karman_solver import KarmanSolver

RollPass.karman_solution = Hook[KarmanSolver]()
"""Solution values of von-Karman ODE (either the rigid-roll KarmanSolver or,
when foil_rolling_condition is met, the elastic-plastic FoilRollingSolver)."""

RollPass.foil_rolling_condition = Hook[bool]()
"""Whether the pass is in the foil-rolling regime, i.e. elastic roll flattening
is severe enough that the rigid-roll von-Karman solution is no longer valid and
the full elastic foil-rolling model should be used instead."""

RollPass.foil_rolling_hitchcock_limit = Hook[float]()
"""Hitchcock flattened-to-nominal radius ratio (r'/r) above which the rigid-roll
model is considered invalid (Mauk & Overhagen 2013, p. 12: r'/r >= 2)."""


@RollPass.foil_rolling_hitchcock_limit
def foil_rolling_hitchcock_limit(self: RollPass):
    return 2.0


@RollPass.foil_rolling_condition
def foil_rolling_condition(self: RollPass):
    # Elastic properties (roll & profile) are only needed for the Hitchcock check
    # itself and for the foil model; passes that never set them clearly aren't
    # meant to opt into elastic-flattening effects, so fall back to the
    # traditional rigid-roll model instead of failing hard.
    try:
        self.roll.elastic_modulus
        self.roll.poissons_ratio
        self.in_profile.elastic_modulus
        self.in_profile.poissons_ratio
    except AttributeError:
        return False

    flat_solution = KarmanSolver(roll_pass=self)
    ratio = hitchcock_radius_ratio(self, flat_solution.roll_force_per_unit_width)
    return ratio >= self.foil_rolling_hitchcock_limit


@RollPass.karman_solution
def karman_solution(self: RollPass):
    if self.foil_rolling_condition:
        return FoilRollingSolver(roll_pass=self)
    return KarmanSolver(roll_pass=self)


@RollPass.InProfile.velocity
def profile_entry_velocity(self: RollPass.InProfile):

    return self.roll_pass.karman_solution.entry_velocity


@RollPass.OutProfile.velocity
def velocity(self: RollPass.OutProfile):
    return self.roll_pass.karman_solution.exit_velocity


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
