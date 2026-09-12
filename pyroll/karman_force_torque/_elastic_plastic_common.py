"""Shared elastic-plastic zone-chain solver for
:class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver` and
:class:`.orowan_solver.OrowanSolver`: von Kármán's horizontal equilibrium
with Bay & Wanheim mixed Coulomb/sticking friction, Hooke's-law elastic
entry/exit zones, and a shooting search for the neutral point. The two
solvers differ only in the closure relating sigma_x to sigma_y.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

from pyroll.core import RollPass


def bay_wanheim_coulomb_from_stiction(stiction_coefficient: float) -> float:
    """Equivalent Coulomb friction coefficient for a given stiction
    coefficient, per Bay & Wanheim."""
    m = stiction_coefficient
    return m / (1 + np.pi / 2 + np.arccos(m) + np.sqrt(1 - m ** 2))


def _equivalent_strain(entry_height, height):
    return 2 / np.sqrt(3) * np.log(entry_height / height)


class ElasticPlasticSolverBase:
    """Common setup, geometry and result-extraction for a single
    homogeneous cross-section, elastic-plastic, mixed-friction solver.
    Subclasses set ``section_cls`` to an :class:`ElasticPlasticSection`
    subclass, and provide the public attributes every solver in this
    plugin does (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``)."""

    section_cls = None

    def __init__(self, roll_pass: RollPass, pressure_smoothness: float = 0.1):
        self.roll_pass = roll_pass
        self.pressure_smoothness = pressure_smoothness
        self._setup()
        self._solve()
        self._finalize()

    def _setup(self):
        roll_pass = self.roll_pass
        roll = roll_pass.roll
        profile = roll_pass.in_profile

        self.roll_radius = roll.working_radius
        self.gap = roll_pass.gap
        self.rotational_frequency = roll.rotational_frequency

        self.stiction_coefficient = roll_pass.friction_stiction_coefficient
        self.coulomb_friction_coefficient = (
            roll_pass.coulomb_friction_coefficient
            if roll_pass.has_set_or_cached("coulomb_friction_coefficient")
            else bay_wanheim_coulomb_from_stiction(self.stiction_coefficient)
        )

        self.poissons_ratio = profile.poissons_ratio
        self.elastic_modulus = profile.elastic_modulus
        self.entry_height = profile.equivalent_height
        self.temperature = profile.temperature
        self.flow_stress_function = profile.flow_stress_function

        self.back_tension = roll_pass.back_tension
        self.front_tension = roll_pass.front_tension

    def height(self, position, roll_radius):
        return self.gap + 2 * (roll_radius - np.sqrt(roll_radius ** 2 - position ** 2))

    def height_derivative(self, position, roll_radius):
        return 2 * position / np.sqrt(roll_radius ** 2 - position ** 2)

    def roll_surface_angle(self, position, roll_radius):
        return -np.arcsin(np.clip(position / roll_radius, -1, 1))

    def contact_length(self, roll_radius):
        draft = self.entry_height - self.gap
        return np.sqrt(max(roll_radius * draft - draft ** 2 / 4, 1e-30))

    def roll_surface_velocity(self, position, roll_radius):
        angle = self.roll_surface_angle(position, roll_radius)
        return 2 * np.pi * self.rotational_frequency * self.roll_radius * np.cos(angle)

    def mean_strain_rate(self, roll_radius):
        contact_length = self.contact_length(roll_radius)
        draft_strain = np.log(self.entry_height / self.gap)
        return abs(2 * np.pi * self.rotational_frequency * roll_radius / contact_length * draft_strain)

    def flow_stress(self, strain, strain_rate):
        return self.flow_stress_function(strain=strain, strain_rate=strain_rate, temperature=self.temperature)

    def _solve(self):
        self.section = self.section_cls(self, self.roll_radius)
        self.section.solve()

    def _finalize(self):
        section = self.section
        self.roll_force_per_unit_width = section.force_per_width
        self.roll_torque_per_unit_width = section.torque_per_width
        self.neutral_plane_position = section.neutral_point_position
        self.solution = section.solution_dataframe()
        self.entry_position = section.entry_position
        self.exit_position = section.exit_position

        neutral_point_height = self.height(section.neutral_point_position, section.roll_radius)
        neutral_point_velocity = self.roll_surface_velocity(section.neutral_point_position, section.roll_radius)
        self.entry_velocity = (
                neutral_point_velocity * neutral_point_height / self.height(section.entry_position, section.roll_radius)
        )
        self.exit_velocity = (
                neutral_point_velocity * neutral_point_height / self.height(section.exit_position, section.roll_radius)
        )


class ElasticPlasticSection:
    """Shared zone-chain/shooting machinery. Subclasses implement
    ``_closure_sigma_y(sigma_x, angle, flow_stress, zone_sign)``, the
    algebraic relation closing von Kármán's horizontal equilibrium (plain
    Mises for ``KarmanMixedFrictionSolver``, Orowan's inhomogeneity-corrected
    relation for ``OrowanSolver``); everything else (elastic zones, plastic
    zones, entry/exit zone chains, the neutral-point shooting search,
    force/torque integration and the reported solution) is identical
    between them.

    ``zone_sign`` is +1 on entry (material slower than the roll) and -1 on
    exit (material faster), consistently determining friction direction
    throughout rather than a velocity-difference sign.
    """

    def __init__(self, solver: ElasticPlasticSolverBase, roll_radius: float):
        self.solver = solver
        self.roll_radius = roll_radius
        self.entry_height = solver.entry_height
        self.contact_length = solver.contact_length(roll_radius)
        self.entry_position = -self.contact_length

    def _closure_sigma_y(self, sigma_x, angle, flow_stress, zone_sign):
        raise NotImplementedError

    def _yield_margin(self, sigma_x, sigma_y_elastic, angle, flow_stress, zone_sign):
        """Distance from the elastic sigma_y to the closure's own predicted
        sigma_y at the same sigma_x - the yield-onset event target, so the
        elastic/plastic transition stays continuous with each subclass's
        own closure rather than a fixed Mises criterion."""
        return sigma_y_elastic - self._closure_sigma_y(sigma_x, angle, flow_stress, zone_sign)

    def _flow_stress_at(self, height):
        # h can momentarily exceed entry_height during the exit zone's
        # adaptive-step elastic recovery, before the separation event fires;
        # clipping the strain at 0 there avoids feeding flow-stress models
        # a negative value (e.g. Freiberg's power-law form raises it to a
        # fractional exponent, producing NaN that poisons the integration).
        strain = max(_equivalent_strain(self.entry_height, height), 0.0)
        return self.solver.flow_stress(strain, self.strain_rate)

    @property
    def strain_rate(self):
        return self.solver.mean_strain_rate(self.roll_radius)

    def _mixed_friction(self, sigma_y, angle, flow_stress, zone_sign):
        """Bay & Wanheim's mixed Coulomb/sticking friction law (also used,
        in the same form, by layer_solver.py). Returns
        (shear_stress, normal_pressure, transition_weight) - transition_weight
        (0 at low pressure, 1 once sticking-limited) is reused by
        OrowanSolver's section to blend its inhomogeneity function."""
        solver = self.solver
        sign = 1.0 if zone_sign > 0 else -1.0
        coulomb_pressure = -sigma_y / (1 + solver.coulomb_friction_coefficient * np.tan(angle) * sign)
        critical_pressure = solver.stiction_coefficient * flow_stress / (solver.coulomb_friction_coefficient * np.sqrt(3))
        transition_weight = np.arctan(
            (coulomb_pressure - critical_pressure) / (solver.pressure_smoothness * critical_pressure)
        ) / np.pi + 0.5
        shear_stress = (
                solver.coulomb_friction_coefficient * coulomb_pressure * (1 - transition_weight)
                + solver.stiction_coefficient * flow_stress / np.sqrt(3) * transition_weight
        ) * sign
        normal_pressure = -sigma_y - shear_stress * np.tan(angle)
        return shear_stress, normal_pressure, transition_weight

    def _elastic_rhs(self, position, state, zone_sign):
        """Elastic compression/recovery (Hooke's law) with Bay & Wanheim
        friction already acting - well-behaved as pressure approaches
        zero, unlike sticking friction, so it applies from first contact."""
        solver = self.solver
        sigma_x, sigma_y, sigma_z, height, temperature = state
        roll_radius = self.roll_radius
        height_now = solver.height(position, roll_radius)
        height_derivative = solver.height_derivative(position, roll_radius)
        angle = solver.roll_surface_angle(position, roll_radius)
        flow_stress = self._flow_stress_at(height)

        shear_stress, normal_pressure, _ = self._mixed_friction(sigma_y, angle, flow_stress, zone_sign)
        d_sigma_x = -(sigma_x * height_derivative + 2 * shear_stress - 2 * normal_pressure * np.tan(angle)) / height_now

        d_strain_y = height_derivative / self.entry_height
        plane_strain_modulus = solver.elastic_modulus / (1 - solver.poissons_ratio ** 2)
        d_sigma_y = plane_strain_modulus * d_strain_y + solver.poissons_ratio / (1 - solver.poissons_ratio) * d_sigma_x
        d_sigma_z = solver.poissons_ratio * (d_sigma_y + d_sigma_x)
        return [d_sigma_x, d_sigma_y, d_sigma_z, height_derivative, 0.0]

    def _plastic_rhs(self, position, state, zone_sign):
        """Von Kármán's horizontal equilibrium, integrating sigma_x with
        Bay & Wanheim mixed friction and sigma_y derived at each step via
        the subclass's own closure."""
        solver = self.solver
        roll_radius = self.roll_radius
        sigma_x, height, temperature = state

        height_now = solver.height(position, roll_radius)
        height_derivative = solver.height_derivative(position, roll_radius)
        angle = solver.roll_surface_angle(position, roll_radius)
        flow_stress = self._flow_stress_at(height)

        sigma_y = self._closure_sigma_y(sigma_x, angle, flow_stress, zone_sign)
        shear_stress, normal_pressure, _ = self._mixed_friction(sigma_y, angle, flow_stress, zone_sign)

        d_sigma_x = -(sigma_x * height_derivative + 2 * shear_stress - 2 * normal_pressure * np.tan(angle)) / height_now
        return [d_sigma_x, height_derivative, 0.0]

    def _solve_entry(self, neutral_point_position):
        solver = self.solver
        initial_state = [solver.back_tension, 0.0, solver.back_tension / 2, self.entry_height, solver.temperature]

        def yield_event(position, state):
            sigma_x, sigma_y, sigma_z, height, temperature = state
            angle = solver.roll_surface_angle(position, self.roll_radius)
            flow_stress = self._flow_stress_at(height)
            return self._yield_margin(sigma_x, sigma_y, angle, flow_stress, zone_sign=1)

        yield_event.terminal = True
        yield_event.direction = 1

        # scipy's direction=1 event only fires on a negative-to-positive
        # crossing during integration, not when already positive at the
        # start - which happens whenever the elastic zone is vanishingly
        # short (typical for hot rolling, where flow_stress/elastic_modulus
        # is small), so that case is handled explicitly here.
        if yield_event(self.entry_position, initial_state) >= 0:
            segments = [("elastic", np.array([self.entry_position]), np.array(initial_state).reshape(-1, 1))]
            return self._solve_entry_plastic(
                self.entry_position, neutral_point_position,
                initial_state[0], initial_state[3], initial_state[4], segments,
            )

        solution = solve_ivp(
            lambda position, state: self._elastic_rhs(position, state, zone_sign=1),
            [self.entry_position, neutral_point_position], initial_state, events=yield_event,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        yield_position = solution.t_events[0][0] if solution.t_events[0].size else neutral_point_position
        positions = np.linspace(
            self.entry_position, yield_position,
            max(2, int((yield_position - self.entry_position) / (self.contact_length / 100)) + 1),
        )
        states = solution.sol(positions)

        segments = [("elastic", positions, states)]

        if yield_position >= neutral_point_position - 1e-12:
            return {
                "segments": segments,
                "sigma_x_at_neutral_point": states[0, -1],
                "height_at_neutral_point": states[3, -1],
                "temperature_at_neutral_point": states[4, -1],
            }

        return self._solve_entry_plastic(
            yield_position, neutral_point_position, states[0, -1], states[3, -1], states[4, -1], segments,
        )

    def _solve_entry_plastic(self, yield_position, neutral_point_position, sigma_x_at_yield, height_at_yield,
                              temperature_at_yield, segments):
        initial_state = [sigma_x_at_yield, height_at_yield, temperature_at_yield]
        solution = solve_ivp(
            lambda position, state: self._plastic_rhs(position, state, zone_sign=1),
            [yield_position, neutral_point_position], initial_state,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        positions = np.linspace(
            yield_position, neutral_point_position,
            max(2, int((neutral_point_position - yield_position) / (self.contact_length / 100)) + 1),
        )
        states = solution.sol(positions)
        segments.append(("plastic", positions, states))

        return {
            "segments": segments,
            "sigma_x_at_neutral_point": states[0, -1],
            "height_at_neutral_point": states[1, -1],
            "temperature_at_neutral_point": states[2, -1],
        }

    def _solve_exit(self, neutral_point_position, entry_result):
        solver = self.solver
        far_end = self.contact_length * 1.5

        segments = []
        if neutral_point_position < -1e-12:
            initial_state = [
                entry_result["sigma_x_at_neutral_point"],
                entry_result["height_at_neutral_point"],
                entry_result["temperature_at_neutral_point"],
            ]
            solution = solve_ivp(
                lambda position, state: self._plastic_rhs(position, state, zone_sign=-1),
                [neutral_point_position, 0.0], initial_state,
                dense_output=True, rtol=1e-8, atol=1e-6,
            )
            positions = np.linspace(
                neutral_point_position, 0.0,
                max(2, int((0.0 - neutral_point_position) / (self.contact_length / 100)) + 1),
            )
            states = solution.sol(positions)
            segments.append(("plastic", positions, states))
            sigma_x_at_center, height_at_center, temperature_at_center = states[:, -1]
        else:
            sigma_x_at_center = entry_result["sigma_x_at_neutral_point"]
            height_at_center = entry_result["height_at_neutral_point"]
            temperature_at_center = entry_result["temperature_at_neutral_point"]

        angle_at_center = solver.roll_surface_angle(0.0, self.roll_radius)
        flow_stress_at_center = self._flow_stress_at(height_at_center)
        sigma_y_at_center = self._closure_sigma_y(sigma_x_at_center, angle_at_center, flow_stress_at_center, zone_sign=-1)
        sigma_z_at_center = (sigma_x_at_center + sigma_y_at_center) / 2
        initial_state = [sigma_x_at_center, sigma_y_at_center, sigma_z_at_center, height_at_center, temperature_at_center]

        def separation_event(position, state):
            return state[1]

        separation_event.terminal = True
        separation_event.direction = 1

        solution = solve_ivp(
            lambda position, state: self._elastic_rhs(position, state, zone_sign=-1),
            [0.0, far_end], initial_state, events=separation_event,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        exit_position = solution.t_events[0][0] if solution.t_events[0].size else far_end
        positions = np.linspace(0.0, exit_position, max(2, int(exit_position / (self.contact_length / 100)) + 1))
        states = solution.sol(positions)
        segments.append(("elastic", positions, states))

        return {"segments": segments, "sigma_x_at_exit": states[0, -1], "exit_position": exit_position}

    def solve(self):
        solver = self.solver

        def residual(neutral_point_position):
            entry_result = self._solve_entry(neutral_point_position)
            exit_result = self._solve_exit(neutral_point_position, entry_result)
            return exit_result["sigma_x_at_exit"] - solver.front_tension

        def safe_residual(neutral_point_position):
            # An off-root candidate in the coarse scan below can make the
            # plastic-zone shooting diverge badly enough that solve_ivp
            # rejects a non-finite state outright; treated as unusable
            # (nan) here rather than letting that exception abort the
            # whole search.
            try:
                value = residual(neutral_point_position)
            except (ValueError, RuntimeError):
                return np.nan
            return value if np.isfinite(value) else np.nan

        search_start, search_end = self.entry_position * 0.999, -1e-9
        candidates = np.linspace(search_start, search_end, 12)
        residuals = [safe_residual(candidate) for candidate in candidates]
        bracket = None
        for i in range(len(candidates) - 1):
            if np.isnan(residuals[i]) or np.isnan(residuals[i + 1]):
                continue
            if residuals[i] == 0:
                bracket = (candidates[i], candidates[i])
                break
            if np.sign(residuals[i]) != np.sign(residuals[i + 1]):
                bracket = (candidates[i], candidates[i + 1])
                break
        if bracket is None:
            raise RuntimeError(f"{type(self).__name__}: could not bracket the neutral point.")
        neutral_point_position = (
            bracket[0] if bracket[0] == bracket[1] else brentq(safe_residual, *bracket, xtol=1e-9)
        )

        self.neutral_point_position = neutral_point_position
        self.entry = self._solve_entry(neutral_point_position)
        self.exit = self._solve_exit(neutral_point_position, self.entry)
        self.exit_position = self.exit["exit_position"]

        self.force_per_width, self.torque_per_width = self._integrate_force_torque()
        return neutral_point_position

    def _state_at(self, zone_name, position, state, zone_sign):
        """Returns (sigma_y, height, shear_stress) for a state vector from
        either zone type - elastic states carry sigma_y directly, plastic
        states derive it (and shear_stress) via the closure/friction law."""
        angle = self.solver.roll_surface_angle(position, self.roll_radius)
        if zone_name == "elastic":
            sigma_x, sigma_y, sigma_z, height, temperature = state
            flow_stress = self._flow_stress_at(height)
            shear_stress, _, _ = self._mixed_friction(sigma_y, angle, flow_stress, zone_sign)
            return sigma_y, height, shear_stress
        sigma_x, height, temperature = state
        flow_stress = self._flow_stress_at(height)
        sigma_y = self._closure_sigma_y(sigma_x, angle, flow_stress, zone_sign)
        shear_stress, _, _ = self._mixed_friction(sigma_y, angle, flow_stress, zone_sign)
        return sigma_y, height, shear_stress

    def _integrate_force_torque(self):
        force = 0.0
        torque = 0.0
        for zone_sign, zone_result in ((1, self.entry), (-1, self.exit)):
            for zone_name, positions, states in zone_result["segments"]:
                normal_pressures = np.array([
                    -self._state_at(zone_name, positions[i], states[:, i], zone_sign)[0]
                    for i in range(positions.shape[0])
                ])
                force += trapezoid(normal_pressures, positions)
                angles = self.solver.roll_surface_angle(positions, self.roll_radius)
                local_radius = np.sqrt(np.maximum(self.roll_radius ** 2 - positions ** 2, 0.0))
                torque += trapezoid(np.abs(normal_pressures) * local_radius * np.tan(np.abs(angles)), positions)
        return abs(force), abs(torque)

    def solution_dataframe(self):
        positions, pressures, shear_stresses, strains = [], [], [], []
        for zone_sign, zone_result in ((1, self.entry), (-1, self.exit)):
            for zone_name, zone_positions, zone_states in zone_result["segments"]:
                for i, position in enumerate(zone_positions):
                    sigma_y, height, shear_stress = self._state_at(zone_name, position, zone_states[:, i], zone_sign)
                    positions.append(position)
                    pressures.append(-sigma_y)
                    shear_stresses.append(shear_stress)
                    strains.append(_equivalent_strain(self.entry_height, height))
        sort_order = np.argsort(positions)
        sorted_positions = np.array(positions)[sort_order]
        sorted_pressures = np.array(pressures)[sort_order]
        sorted_shear_stresses = np.array(shear_stresses)[sort_order]
        sorted_strains = np.array(strains)[sort_order]
        return pd.DataFrame(
            {
                "vertical_stress": -sorted_pressures,
                "normal_pressure": sorted_pressures,
                "shear_stress": sorted_shear_stresses,
                "equivalent_strain": sorted_strains,
            },
            index=pd.Index(sorted_positions, name="x"),
        )
