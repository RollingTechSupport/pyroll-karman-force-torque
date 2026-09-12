"""Elastic-plastic foil-rolling model.

Implements the Fleck/Johnson/Sutcliffe-type model described in Mauk & Overhagen,
"Prozessmodell zum Kaltwalzen von Metallfolien mit keramischen Arbeitswalzen auf
Mehrwalzengerüsten" (2013), and the accompanying MATLAB implementation by
C. Overhagen ("lee_sutcliffe"). Unlike :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`,
this solver does not assume the roll stays circular in the contact zone: the roll-gap
shape is found by superposing elastic half-space (Johnson) point-load solutions
over the strip's own normal-pressure distribution and iterating to convergence.

All quantities entering/leaving this module are SI (Pa, m, ...), matching
``pyroll-core`` conventions. Internally the strip/roll-gap equations are solved
in the dimensionless form of the report, using

- ``X = x * E'_R / (R * k_fe)``
- ``T = h * E'_R**2 / (R * k_fe**2)``
- ``P = p / k_fe`` (normal pressure), ``S = sigma_x / k_fe`` (horizontal stress)

where ``E'_R = E_R / (1 - nu_R**2)`` is the roll's plane-strain modulus and
``k_fe = k_f - 0.5 * (back_tension + front_tension)`` is the tension-adjusted
representative flow stress of the pass (report, p. 15-17). Tensions here are
normalized by ``k_fe`` throughout (rather than the raw ``k_f`` as the reference
MATLAB code does), which keeps the normalization dimensionally consistent with
the yield condition ``S + P = k_f / k_fe`` used in the ODEs below. Local
variables named ``position``/``pressure_dimless``/``sigma_x_dimless``/
``height_dimless`` hold X/P/S/T respectively.
"""

import logging

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from pyroll.core import RollPass

log = logging.getLogger(__name__)


def _influence_matrix(element_count: int, step: float) -> np.ndarray:
    """Dimensionless elastic half-space influence-coefficient matrix ``D_ij - D_1j``
    (eq. 34/35 of the report; ``Dij.m`` in the reference implementation).

    All elastic material properties cancel out of the non-dimensional form of the
    problem (verified by substitution into eq. 6), so the only remaining
    parameter is the dimensionless element width ``step`` (the "C" of eq. 35,
    which the reference MATLAB code re-purposes as the grid spacing rather than
    an elastic constant). The additive constant ``D0`` (eq. 35) is arbitrary and
    cancels exactly through the ``D_ij - D_1j`` differencing, so it is omitted.
    """
    index = np.arange(element_count)
    offset = index[:, None] - index[None, :]
    offset_from_first = index[0] - index[None, :]

    def d(offset):
        k1, k2, k3 = offset + 1, offset - 1, offset
        with np.errstate(divide="ignore"):
            lk1 = np.where(k1 == 0, 0.0, np.log(k1.astype(float) ** 2))
            lk2 = np.where(k2 == 0, 0.0, np.log(k2.astype(float) ** 2))
            lk3 = np.where(k3 == 0, 0.0, np.log(k3.astype(float) ** 2))
        return -step / (2 * np.pi) * (k1 ** 2 * lk1 + k2 ** 2 * lk2 - 2 * k3 ** 2 * lk3)

    d_ij = d(offset)
    d_1j = d(offset_from_first)
    return d_ij - d_1j


class FoilRollingSolver:
    """Elastic-plastic solution of the foil-rolling problem for a roll pass.

    Provides the same public attributes as the other solvers in this plugin
    (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``) so that consumers (report plotting, the roll-pass hooks) do
    not need to know which solver actually ran.
    """

    def __init__(
            self,
            roll_pass: RollPass,
            element_count: int = 100,
            relaxation_factor: float = 0.2,
            max_outer_iterations: int = 150,
            tolerance: float = 1e-3,
    ):
        self.roll_pass = roll_pass
        self.element_count = element_count
        self.relaxation_factor = relaxation_factor
        self.max_outer_iterations = max_outer_iterations
        self.tolerance = tolerance

        self._setup()
        self._solve_outer_loop()
        self._finalize()

    def _setup(self):
        roll_pass = self.roll_pass
        roll = roll_pass.roll
        profile = roll_pass.in_profile

        self.radius = roll.working_radius
        self.coulomb_friction_coefficient = roll_pass.coulomb_friction_coefficient

        roll_elastic_modulus, roll_poissons_ratio = roll.elastic_modulus, roll.poissons_ratio
        strip_elastic_modulus, strip_poissons_ratio = profile.elastic_modulus, profile.poissons_ratio
        self.strip_poissons_ratio = strip_poissons_ratio
        self.roll_plane_strain_modulus = roll_elastic_modulus / (1 - roll_poissons_ratio ** 2)
        self.strip_plane_strain_modulus = strip_elastic_modulus / (1 - strip_poissons_ratio ** 2)

        self.entry_height = profile.equivalent_height
        self.exit_height = roll_pass.out_profile.equivalent_height

        total_strain = 2 / np.sqrt(3) * np.log(self.entry_height / self.exit_height)
        self.flow_stress = profile.flow_stress_function(
            strain=total_strain / 2,
            strain_rate=roll_pass.strain_rate,
            temperature=profile.temperature,
        )

        back_tension = roll_pass.back_tension
        front_tension = roll_pass.front_tension
        self.effective_flow_stress = self.flow_stress - 0.5 * (back_tension + front_tension)
        if self.effective_flow_stress <= 0:
            raise ValueError(
                "Combined entry/exit tension exceeds the flow stress; "
                "the foil rolling model requires effective_flow_stress = "
                "flow_stress - 0.5*(back_tension+front_tension) > 0."
            )

        self.position_scale = self.roll_plane_strain_modulus / (self.radius * self.effective_flow_stress)
        self.height_scale = self.roll_plane_strain_modulus ** 2 / (self.radius * self.effective_flow_stress ** 2)

        self.entry_height_dimless = self.entry_height * self.height_scale
        self.exit_height_dimless = self.exit_height * self.height_scale
        self.back_tension_dimless = back_tension / self.effective_flow_stress
        self.front_tension_dimless = front_tension / self.effective_flow_stress
        self.flow_stress_ratio = self.flow_stress / self.effective_flow_stress
        self.dimensionless_friction = (
                self.coulomb_friction_coefficient * self.roll_plane_strain_modulus / self.effective_flow_stress
        )

        nu_s_factor = (1 - strip_poissons_ratio) / (1 - 2 * strip_poissons_ratio)
        self.sticking_zone_factor = nu_s_factor / (
                2 - nu_s_factor * (1 - 2 * roll_poissons_ratio) / (1 - roll_poissons_ratio)
                * self.strip_plane_strain_modulus / self.roll_plane_strain_modulus
        )

        # Wide dimensionless arc, generously larger than the classical (Hitchcock)
        # contact-length estimate, matching sutcliffe.m's factor-of-3 margin.
        classical_half_length = np.sqrt((self.entry_height - self.exit_height) * self.radius)
        half_width = 3 * classical_half_length * self.position_scale
        self.grid_positions = np.linspace(-half_width, half_width, self.element_count)
        self.grid_step = self.grid_positions[1] - self.grid_positions[0]
        self.influence_matrix = _influence_matrix(self.element_count, self.grid_step)

        # Undeformed (circular) roll-gap shape, referenced such that the classical
        # (single-radius) entry point matches entry_height_dimless.
        entry_guess = -classical_half_length * self.position_scale
        self.min_height_base = self.entry_height_dimless - entry_guess ** 2
        self.base_shape = self.min_height_base + self.grid_positions ** 2

        self.current_shape = self.base_shape.copy()
        self.entry_guess = entry_guess
        self.neutral_guess = entry_guess / 5

    def _rhs_elastic(self, position, state, zone_sign, height_of, height_derivative_of):
        sigma_x_dimless, pressure_dimless = state
        height_dimless = height_of(position)
        height_derivative = height_derivative_of(position)
        friction_term = 2 * self.dimensionless_friction * pressure_dimless / height_dimless
        d_sigma_x = -(sigma_x_dimless + pressure_dimless) * height_derivative / height_dimless \
                    - zone_sign * friction_term
        d_pressure = (
                -self.strip_plane_strain_modulus / self.effective_flow_stress * height_derivative / height_dimless
                + zone_sign * self.strip_poissons_ratio / (1 - self.strip_poissons_ratio) * friction_term
        )
        return [d_sigma_x, d_pressure]

    def _rhs_plastic(self, position, state, zone_sign, height_of, height_derivative_of):
        pressure_dimless = state[0]
        height_dimless = height_of(position)
        height_derivative = height_derivative_of(position)
        shear_dimless = zone_sign * self.dimensionless_friction * pressure_dimless
        d_pressure = self.flow_stress_ratio * height_derivative / height_dimless + 2 * shear_dimless / height_dimless
        return [d_pressure]

    def _rhs_sticking(self, position, state, height_of, height_derivative_of):
        height_dimless = height_of(position)
        height_derivative = height_derivative_of(position)
        d_pressure = (
                -self.sticking_zone_factor * self.strip_plane_strain_modulus / self.effective_flow_stress
                * height_derivative / height_dimless
        )
        return [d_pressure]

    def _sticking_event(self, height_derivative_of, direction):
        """Stick/slip switch event: the sticking shear capacity minus the shear
        sliding would demand, crossing zero when one exceeds the other.

        Using a signed ``direction`` (rather than detecting either-direction
        crossings) is essential here: right after this event fires and a new
        integration segment starts exactly at the crossing point, re-evaluating
        the same zero-crossing quantity without a direction constraint finds a
        spurious immediate root at the starting point itself (the value is ~0
        there by construction), causing the stick/slip zones to ping-pong in
        zero-length steps instead of integrating forward.
        """

        def event(position, state, *args):
            pressure_dimless = state[0] if len(state) == 1 else state[1]
            sticking_capacity = -self.sticking_zone_factor * self.strip_plane_strain_modulus \
                                 / self.effective_flow_stress / 2 * height_derivative_of(position)
            return abs(sticking_capacity) - abs(pressure_dimless * self.dimensionless_friction)

        event.terminal = True
        event.direction = direction
        return event

    @staticmethod
    def _yield_event(flow_stress_ratio):
        def event(position, state, *args):
            return state[0] + state[1] - flow_stress_ratio

        event.terminal = True
        event.direction = 0
        return event

    @staticmethod
    def _horizontal_tangent_event(height_derivative_of):
        def event(position, state, *args):
            return height_derivative_of(position)

        event.terminal = True
        event.direction = 0
        return event

    @staticmethod
    def _pressure_zero_event():
        def event(position, state, *args):
            return state[-1]

        event.terminal = True
        event.direction = 0
        return event

    def _solve_zone_chain(self, entry_position, neutral_point_position, height_of, height_derivative_of):
        """Integrate all five zones from ``entry_position`` through
        ``neutral_point_position`` to separation, returning the exit-tension
        residual against the prescribed front tension plus the full stress
        trace for reporting."""
        segments = []

        # 1: elastic entry (sliding)
        solution = solve_ivp(
            self._rhs_elastic, [entry_position, neutral_point_position], [self.back_tension_dimless, 0.0],
            args=(1, height_of, height_derivative_of),
            events=self._yield_event(self.flow_stress_ratio), dense_output=True, max_step=self.grid_step,
        )
        yield_position = solution.t_events[0][0] if solution.t_events[0].size else neutral_point_position
        positions = np.linspace(
            entry_position, yield_position, max(2, int((yield_position - entry_position) / self.grid_step) + 1),
        )
        states = solution.sol(positions)
        segments.append(("elastic", positions, states[1], states[0]))
        pressure_dimless = states[1, -1]
        position = yield_position

        # 2/3: plastic sliding on the entry side, with possible sticking zone(s)
        for _ in range(6):
            if position >= neutral_point_position - 1e-12:
                break
            solution = solve_ivp(
                self._rhs_plastic, [position, neutral_point_position], [pressure_dimless],
                args=(1, height_of, height_derivative_of),
                events=self._sticking_event(height_derivative_of, direction=-1),
                dense_output=True, max_step=self.grid_step,
            )
            segment_end = solution.t_events[0][0] if solution.t_events[0].size else neutral_point_position
            positions = np.linspace(position, segment_end, max(2, int((segment_end - position) / self.grid_step) + 1))
            states = solution.sol(positions)
            segments.append(("plastic_slip", positions, states[0], None))
            pressure_dimless = states[0, -1]
            position = segment_end
            if position >= neutral_point_position - 1e-12:
                break

            solution = solve_ivp(
                self._rhs_sticking, [position, neutral_point_position], [pressure_dimless],
                args=(height_of, height_derivative_of),
                events=self._sticking_event(height_derivative_of, direction=1),
                dense_output=True, max_step=self.grid_step,
            )
            segment_end = solution.t_events[0][0] if solution.t_events[0].size else neutral_point_position
            positions = np.linspace(position, segment_end, max(2, int((segment_end - position) / self.grid_step) + 1))
            states = solution.sol(positions)
            segments.append(("sticking", positions, states[0], None))
            pressure_dimless = states[0, -1]
            position = segment_end

        # 4: plastic sliding on the exit side, forward slip, until horizontal tangent
        far_end = self.grid_positions[-1]
        solution = solve_ivp(
            self._rhs_plastic, [neutral_point_position, far_end], [pressure_dimless],
            args=(-1, height_of, height_derivative_of),
            events=self._horizontal_tangent_event(height_derivative_of), dense_output=True, max_step=self.grid_step,
        )
        if not solution.t_events[0].size:
            raise RuntimeError("Foil rolling model: could not locate the horizontal roll-gap tangent.")
        tangent_position = solution.t_events[0][0]
        positions = np.linspace(
            neutral_point_position, tangent_position,
            max(2, int((tangent_position - neutral_point_position) / self.grid_step) + 1),
        )
        states = solution.sol(positions)
        segments.append(("plastic_slip", positions, states[0], None))
        pressure_at_tangent = states[0, -1]

        # 5: elastic recovery, until separation (pressure=0)
        sigma_x_at_tangent = self.flow_stress_ratio - pressure_at_tangent
        solution = solve_ivp(
            self._rhs_elastic, [tangent_position, far_end], [sigma_x_at_tangent, pressure_at_tangent],
            args=(-1, height_of, height_derivative_of),
            events=self._pressure_zero_event(), dense_output=True, max_step=self.grid_step,
        )
        if not solution.t_events[0].size:
            raise RuntimeError("Foil rolling model: could not locate the exit separation point.")
        exit_position = solution.t_events[0][0]
        positions = np.linspace(
            tangent_position, exit_position, max(2, int((exit_position - tangent_position) / self.grid_step) + 1),
        )
        states = solution.sol(positions)
        segments.append(("elastic", positions, states[1], states[0]))
        sigma_x_at_exit = states[0, -1]

        residual = sigma_x_at_exit - self.front_tension_dimless
        return residual, exit_position, segments

    @staticmethod
    def _bracket_from_candidates(candidates, values):
        for i in range(len(candidates) - 1):
            v0, v1 = values[i], values[i + 1]
            if np.isnan(v0) or np.isnan(v1):
                continue
            if v0 == 0:
                return candidates[i], candidates[i]
            if v0 * v1 < 0:
                return candidates[i], candidates[i + 1]
        return None

    def _gap_minimum(self, entry_position, height_derivative_of):
        """Upper bound for the neutral point: the (first, moving rightward from
        the entry point) point where the *current* roll-gap shape has a
        horizontal tangent. The neutral point must lie strictly before this -
        it is not, in general, at X=0, which is merely where the undeformed
        reference parabola happens to be centered and drifts arbitrarily as
        the elastic correction is applied, so it must never be used as a
        search bound."""
        candidates = np.linspace(
            entry_position + 1e-6 * abs(entry_position), self.grid_positions[-1] * 0.999, 60,
        )
        values = height_derivative_of(candidates)
        for i in range(len(candidates) - 1):
            if values[i] == 0:
                return candidates[i]
            if values[i] * values[i + 1] < 0:
                return brentq(lambda x: float(height_derivative_of(x)), candidates[i], candidates[i + 1], xtol=1e-10)
        return self.grid_positions[-1] * 0.999

    def _find_neutral_point(self, entry_position, height_of, height_derivative_of):
        def residual(neutral_point_position):
            return self._solve_zone_chain(entry_position, neutral_point_position, height_of, height_derivative_of)[0]

        upper_bound = self._gap_minimum(entry_position, height_derivative_of)

        # Warm-start: the neutral point moves only slightly between successive
        # outer (elastic-flattening) iterations, and the roll-gap height can
        # develop local non-monotonicities once flattening is significant, so
        # a wide scan from the domain edge risks locking onto a spurious root
        # far from the physically continued solution. Try a narrow bracket
        # around the previous iteration's value first, widening it if
        # necessary, before falling back to a full scan.
        guess = np.clip(
            self.neutral_guess, entry_position + 1e-6 * abs(entry_position), upper_bound - 1e-6 * abs(entry_position),
        )
        for half_width in (0.05, 0.2, 0.5, 1.0, 2.0):
            lo = max(entry_position + 1e-6 * abs(entry_position), guess - half_width)
            hi = min(upper_bound - 1e-6 * abs(entry_position), guess + half_width)
            if hi <= lo:
                continue
            candidates = np.linspace(lo, hi, 9)
            try:
                values = [residual(candidate) for candidate in candidates]
            except RuntimeError:
                continue
            bracket = self._bracket_from_candidates(candidates, values)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    return bracket[0]
                return brentq(residual, *bracket, xtol=1e-10)

        candidates = np.linspace(
            entry_position + 1e-6 * abs(entry_position), upper_bound - 1e-6 * abs(entry_position), 25,
        )
        values = []
        for candidate in candidates:
            try:
                values.append(residual(candidate))
            except RuntimeError:
                values.append(np.nan)

        bracket = self._bracket_from_candidates(candidates, values)
        if bracket is None:
            raise RuntimeError("Foil rolling model: could not bracket the neutral point.")
        if bracket[0] == bracket[1]:
            return bracket[0]
        return brentq(residual, *bracket, xtol=1e-10)

    def _find_entry_point(self, height_of):
        def residual(entry_position):
            return height_of(entry_position) - self.entry_height_dimless

        # Warm-start around the previous iteration's entry point first (see the
        # rationale in _find_neutral_point), falling back to a full domain scan.
        guess = np.clip(self.entry_guess, self.grid_positions[0] * 0.999, -1e-9)
        for half_width in (0.05, 0.2, 0.5, 1.0, 2.0):
            lo = max(self.grid_positions[0] * 0.999, guess - half_width)
            hi = min(-1e-9, guess + half_width)
            if hi <= lo:
                continue
            candidates = np.linspace(lo, hi, 9)
            values = [residual(candidate) for candidate in candidates]
            bracket = self._bracket_from_candidates(candidates, values)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    return bracket[0]
                return brentq(residual, *bracket, xtol=1e-10)

        candidates = np.linspace(self.grid_positions[0] * 0.999, -1e-9, 20)
        values = [residual(candidate) for candidate in candidates]
        bracket = self._bracket_from_candidates(candidates, values)
        if bracket is None:
            raise RuntimeError("Foil rolling model: could not bracket the entry point.")
        if bracket[0] == bracket[1]:
            return bracket[0]
        return brentq(residual, *bracket, xtol=1e-10)

    def _solve_outer_loop(self):
        base_reference = self.base_shape.copy()
        shape = self.current_shape

        for iteration in range(self.max_outer_iterations):
            height_of = CubicSpline(self.grid_positions, shape)
            height_derivative_of = height_of.derivative()

            entry_position = self._find_entry_point(height_of)
            neutral_point_position = self._find_neutral_point(entry_position, height_of, height_derivative_of)
            _, exit_position, segments = self._solve_zone_chain(
                entry_position, neutral_point_position, height_of, height_derivative_of,
            )

            all_positions = np.concatenate([segment[1] for segment in segments])
            all_pressures = np.concatenate([segment[2] for segment in segments])
            fill_value = all_pressures.min()
            pressures = np.interp(self.grid_positions, all_positions, all_pressures, left=fill_value, right=fill_value)

            deflection = self.influence_matrix @ pressures
            new_shape = base_reference + 2 * deflection

            height_at_exit = float(np.interp(exit_position, self.grid_positions, new_shape))
            offset = height_at_exit - self.exit_height_dimless
            new_shape = new_shape - offset
            base_reference = base_reference - offset

            relaxed_shape = self.relaxation_factor * new_shape + (1 - self.relaxation_factor) * shape
            change = np.max(np.abs(relaxed_shape - shape))

            shape = relaxed_shape
            self.entry_guess, self.neutral_guess = entry_position, neutral_point_position

            if change < self.tolerance:
                log.debug(f"Foil rolling model converged after {iteration + 1} outer iterations.")
                break
        else:
            log.warning(
                f"Foil rolling model did not converge within {self.max_outer_iterations} outer iterations "
                f"(last change {change:.3e})."
            )

        self.current_shape = shape
        self.entry_point_dimless = entry_position
        self.neutral_point_dimless = neutral_point_position
        self.exit_point_dimless = exit_position
        self.final_segments = segments

    def _finalize(self):
        height_of = CubicSpline(self.grid_positions, self.current_shape)
        height_derivative_of = height_of.derivative()

        positions, pressures_dimless, signs_by_position, zones_by_position = [], [], [], []
        for zone, zone_positions, zone_pressures, _ in self.final_segments:
            positions.append(zone_positions)
            pressures_dimless.append(zone_pressures)
            sign = 1.0 if np.mean(zone_positions) < self.neutral_point_dimless else -1.0
            signs_by_position.append(np.full_like(zone_positions, sign))
            zones_by_position.append([zone] * len(zone_positions))

        positions = np.concatenate(positions)
        pressures_dimless = np.concatenate(pressures_dimless)
        signs = np.concatenate(signs_by_position)
        zones = np.concatenate(zones_by_position)

        sort_order = np.argsort(positions)
        positions, pressures_dimless, signs, zones = (
            positions[sort_order], pressures_dimless[sort_order], signs[sort_order], zones[sort_order],
        )

        x_physical = positions / self.position_scale
        pressure = pressures_dimless * self.effective_flow_stress

        height_derivative = height_derivative_of(positions) * self.effective_flow_stress / self.roll_plane_strain_modulus
        shear = np.where(
            zones == "sticking",
            self.sticking_zone_factor * self.strip_plane_strain_modulus / 2 * height_derivative,
            signs * self.coulomb_friction_coefficient * pressure,
        )
        vertical_stress = -pressure

        height = height_of(positions) / self.height_scale
        equivalent_strain = 2 / np.sqrt(3) * np.log(self.entry_height / height)

        self.solution = pd.DataFrame(
            {
                "vertical_stress": vertical_stress,
                "normal_pressure": pressure,
                "shear_stress": shear,
                "equivalent_strain": equivalent_strain,
            },
            index=pd.Index(x_physical, name="x"),
        )

        # The converged, generally non-circular roll-gap contour against the
        # rigid/circular baseline it started from - the headline result of
        # this solver's outer elastic-flattening iteration (see module
        # docstring), so worth exposing for reporting alongside the stress
        # and strain profiles above.
        self.roll_contour = pd.DataFrame(
            {
                "gap_height": self.current_shape / self.height_scale,
                "rigid_gap_height": self.base_shape / self.height_scale,
            },
            index=pd.Index(self.grid_positions / self.position_scale, name="x"),
        )

        self.roll_force_per_unit_width = float(trapezoid(pressure, x_physical))
        self.roll_torque_per_unit_width = float(trapezoid(shear, x_physical))

        self.neutral_plane_position = self.neutral_point_dimless / self.position_scale
        self.entry_position = self.entry_point_dimless / self.position_scale
        self.exit_position = self.exit_point_dimless / self.position_scale

        # Kinematics via mass continuity, with the roll surface velocity projected
        # onto the horizontal using the *rigid* roll angle at the neutral point
        # (the report neglects this projection entirely for foil rolling, eq. 41-43,
        # since angles are tiny there; keeping it matches this plugin's other solvers'
        # convention and is a negligible correction in the foil-rolling regime anyway).
        rotational_frequency = self.roll_pass.roll.rotational_frequency
        neutral_angle = -np.arcsin(np.clip(self.neutral_plane_position / self.radius, -1, 1))
        neutral_velocity = 2 * np.pi * rotational_frequency * self.radius * np.cos(neutral_angle)
        neutral_height = float(height_of(self.neutral_point_dimless)) / self.height_scale

        self.entry_velocity = neutral_velocity * neutral_height / self.entry_height
        self.exit_velocity = neutral_velocity * neutral_height / self.exit_height
        self.forward_slip = self.exit_velocity / (2 * np.pi * rotational_frequency * self.radius) - 1
