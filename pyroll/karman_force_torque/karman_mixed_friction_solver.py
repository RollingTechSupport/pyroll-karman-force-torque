"""Von-Kármán slab theory with a mixed Coulomb/sticking friction law.

Generalizes :class:`.karman_solver.KarmanSolver`'s von-Kármán slab method
with a mixed Coulomb/sticking friction law, following Bay & Wanheim's
smoothed formulation of it (Bay & Wanheim, "Real area of contact and
friction stresses at high pressure sliding contact", Wear 38, 1976 -
the same law :mod:`.layer_solver` uses per-layer), instead of pure Coulomb
friction throughout. Plain Coulomb friction, integrated all the way to the
roll gap center, produces an unbounded pressure spike once the friction
hill would demand more shear traction than the material can actually
transmit; the mixed law caps it at the material's sticking (shear-yield)
limit instead, which is what actually happens once passes get thick enough
(or friction high enough) for pure Coulomb to become unrealistic.

Note this is *not* Orowan's own 1943 slab theory (Orowan, "The Calculation
of Roll Pressure in Hot and Cold Flat Rolling", Proc. IMechE 150, 1943),
which is a substantially different and more involved model: it replaces
the straight strip-element boundaries with circular arcs (centered on the
roll-gap symmetry line, meeting the roll surfaces at right angles) and an
inhomogeneity function correcting for genuinely non-uniform through-
thickness deformation, rather than a friction-law refinement on top of the
same strip-element geometry. This solver only reuses Orowan's namesake
insight that pure Coulomb friction becomes unrealistic once passes are
thick or high-friction enough - the actual mechanism here (mixed
Coulomb/sticking friction) is Bay & Wanheim's, not Orowan's.

Like :class:`.karman_solver.KarmanSolver` (and unlike
:class:`.layer_solver.LayerRollingSolver` or
:class:`.foil_solver.FoilRollingSolver`), this solver is rigid-plastic: no
elastic entry/exit zones and no thermal coupling - those are a separate
axis (covered here by the layer model) from the friction-law refinement
this solver adds.

This solver is used for the "medium" dispatch tier (see ``roll_pass.py``),
in place of ``LayerRollingSolver(layer_count=1)``: it keeps the simplicity
and numerical robustness of a single-pass shooting solve (same structure as
KarmanSolver, no elastic-zone chain-of-events machinery) while still
capturing the friction-hill saturation that matters once passes are thick
enough for pure Coulomb friction to overstate the pressure peak.
"""

import logging

import numpy as np
import pandas as pd
import scipy.interpolate as inter
import scipy.optimize as opt
from scipy.integrate import quad

from pyroll.core import RollPass

log = logging.getLogger(__name__)


def bay_wanheim_coulomb_from_stiction(stiction_coefficient: float) -> float:
    """Equivalent Coulomb friction coefficient for a given stiction
    coefficient ``m``, per Bay & Wanheim (used when
    ``coulomb_friction_coefficient`` isn't explicitly set)."""
    m = stiction_coefficient
    return m / (1 + np.pi / 2 + np.arccos(m) + np.sqrt(1 - m ** 2))


class KarmanMixedFrictionSolver:
    """Rigid-plastic von-Kármán slab-theory solution of a flat roll pass
    with Bay & Wanheim's mixed Coulomb/sticking friction law - see module
    docstring (and note there on why this isn't Orowan's own theory).

    Provides the same public attributes as :class:`.karman_solver.KarmanSolver`
    (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``) so that consumers (report plotting, the roll-pass hooks) do
    not need to know which solver actually ran.
    """

    def __init__(self, roll_pass: RollPass, pressure_smoothness: float = 0.1):
        self.roll_pass = roll_pass
        self.pressure_smoothness = pressure_smoothness
        self.slab_element_count = 125
        self.entry_position = self.roll_pass.entry_point
        self.exit_position = 0
        self.step_width = self.roll_pass.roll.contact_length / self.slab_element_count

        self.mr = roll_pass.friction_stiction_coefficient
        self.mu_r = (
            roll_pass.coulomb_friction_coefficient
            if roll_pass.has_set_or_cached("coulomb_friction_coefficient")
            else bay_wanheim_coulomb_from_stiction(self.mr)
        )

        self.forward_solution = self.solve(solution_direction="forward")
        self.backward_solution = self.solve(solution_direction="backward")

        self.horizontal_stress_forward_interpolation = inter.interp1d(
            self.forward_solution.index, self.forward_solution["horizontal_stress"],
            fill_value="extrapolate",
        )
        self.horizontal_stress_backward_interpolation = inter.interp1d(
            self.backward_solution.index, self.backward_solution["horizontal_stress"],
            fill_value="extrapolate",
        )

        self.neutral_plane_position = self.find_neutral_plane()
        self.neutral_plane_angle = self.roll_angle(self.neutral_plane_position)
        self.neutral_plane_velocity = 2 * np.pi * self.roll_pass.roll.rotational_frequency * self.roll_pass.roll.working_radius * np.cos(
            self.neutral_plane_angle)
        self.solution = self.generate_solution()
        self.vertical_stress_interpolation = inter.interp1d(
            self.solution.index, self.solution["vertical_stress"], fill_value="extrapolate")
        self.shear_stress_interpolation = inter.interp1d(
            self.solution.index, self.solution["shear_stress"], fill_value="extrapolate")
        self.roll_force_per_unit_width = self.return_roll_force_per_unit_width()
        self.roll_torque_per_unit_width = self.return_roll_torque_per_unit_width()
        self.entry_velocity = self.material_velocity(self.roll_pass.entry_point)
        self.exit_velocity = self.material_velocity(self.roll_pass.exit_point)

    def equivalent_roll_gap_height(self, roll_gap_coordinate):
        return self.roll_pass.gap + 2 * (self.roll_pass.roll.working_radius - np.sqrt(
            self.roll_pass.roll.working_radius ** 2 - roll_gap_coordinate ** 2))

    def roll_gap_height_derivative(self, roll_gap_coordinate):
        rw = self.roll_pass.roll.working_radius
        return 2 * roll_gap_coordinate / np.sqrt(rw ** 2 - roll_gap_coordinate ** 2)

    def roll_angle(self, roll_gap_coordinate):
        return -np.arcsin(roll_gap_coordinate / self.roll_pass.roll.working_radius)

    def material_velocity(self, roll_gap_coordinate):
        return self.neutral_plane_velocity * self.equivalent_roll_gap_height(
            self.neutral_plane_position) / self.equivalent_roll_gap_height(roll_gap_coordinate)

    def equivalent_local_strain(self, roll_gap_coordinate):
        return 2 / np.sqrt(3) * np.log(self.roll_pass.in_profile.equivalent_height / self.equivalent_roll_gap_height(
            roll_gap_coordinate))

    def _mixed_friction(self, vertical_stress, flow_stress, roll_angle, direction):
        """Bay & Wanheim mixed Coulomb/stiction friction law: smoothly
        transitions from Coulomb sliding to a sticking-limited shear stress
        as the Coulomb-implied normal pressure grows past the critical
        value where the two would otherwise cross (see layer_solver.py's
        identical formula, used there per-layer)."""
        pn_coulomb = -vertical_stress / (1 + self.mu_r * np.tan(roll_angle) * direction)
        pn_critical = self.mr * flow_stress / (self.mu_r * np.sqrt(3))
        tau_trans = np.arctan(
            (pn_coulomb - pn_critical) / (self.pressure_smoothness * pn_critical)
        ) / np.pi + 0.5
        shear_stress = (
                self.mu_r * pn_coulomb * (1 - tau_trans) + self.mr * flow_stress / np.sqrt(3) * tau_trans
        ) * direction
        normal_pressure = -vertical_stress - shear_stress * np.tan(roll_angle)
        return shear_stress, normal_pressure

    def solve(self, solution_direction: str):
        if solution_direction == "forward":
            start_position = self.roll_pass.entry_point
            exit_position = self.roll_pass.exit_point
            horizontal_stress = self.roll_pass.back_tension
            direction = 1
        elif solution_direction == "backward":
            start_position = self.roll_pass.exit_point
            exit_position = self.roll_pass.entry_point
            horizontal_stress = self.roll_pass.front_tension
            direction = -1
        else:
            raise ValueError(f"Unknown solution_direction: {solution_direction!r}")

        position = start_position
        step_width_with_direction = self.step_width * direction
        stepwise_solution_storage = {}

        while position < exit_position if direction > 0 else position > exit_position:
            height_derivative = self.roll_gap_height_derivative(position)
            roll_angle = self.roll_angle(position)
            equivalent_strain = self.equivalent_local_strain(position)
            flow_stress = self.roll_pass.in_profile.flow_stress_function(
                strain=equivalent_strain, strain_rate=self.roll_pass.strain_rate,
                temperature=self.roll_pass.in_profile.temperature,
            )
            vertical_stress = horizontal_stress - 2 / np.sqrt(3) * flow_stress
            shear_stress, normal_pressure = self._mixed_friction(vertical_stress, flow_stress, roll_angle, direction)

            stepwise_solution_storage[position] = {
                "horizontal_stress": horizontal_stress, "vertical_stress": vertical_stress,
                "shear_stress": shear_stress, "normal_pressure": normal_pressure,
                "flow_stress": flow_stress, "equivalent_strain": equivalent_strain,
            }

            horizontal_stress_change = - (
                    horizontal_stress * height_derivative + 2 * shear_stress - 2 * normal_pressure * np.tan(
                roll_angle)) / self.equivalent_roll_gap_height(position)
            horizontal_stress += horizontal_stress_change * step_width_with_direction
            position += step_width_with_direction

        log.debug("Finished solution of mixed-friction slab ODE.")
        return pd.DataFrame.from_dict(stepwise_solution_storage, orient="index")

    def return_roll_force_per_unit_width(self):
        integral = -quad(
            lambda position: self.vertical_stress_interpolation(position),
            self.roll_pass.entry_point, self.roll_pass.exit_point,
        )[0]
        return integral

    def return_roll_torque_per_unit_width(self):
        integral = quad(
            lambda position: self.shear_stress_interpolation(position),
            self.roll_pass.entry_point, self.roll_pass.exit_point,
        )[0]
        return integral

    def generate_solution(self):
        return pd.concat([
            self.forward_solution[self.roll_pass.entry_point: self.neutral_plane_position],
            self.backward_solution[self.roll_pass.exit_point: self.neutral_plane_position],
        ]).sort_index()

    def find_neutral_plane(self):
        pos, result = opt.brentq(
            lambda position:
            self.horizontal_stress_backward_interpolation(position)
            - self.horizontal_stress_forward_interpolation(position),
            self.roll_pass.entry_point, self.roll_pass.exit_point, full_output=True,
        )

        if result.converged is True:
            return pos

        raise RuntimeError("Could not find neutral plane position!")
