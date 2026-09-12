"""Elastic-plastic layer model for medium and thick passes.

Ports Max Weiner's thesis work (TU Bergakademie Freiberg,
``RollingLayerModelElasticPlastic.wl`` / ``RollingHomogeneousElasticPlastic.wl``)
to a single, layer-count-parametrized solver: ``layer_count=1`` is the
"medium"/Orowan-like homogeneous case, ``layer_count>1`` is the "thick slab"
case where surface and core develop meaningfully different flow stress
through the pass.

Unlike :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
(elastic zones and mixed friction, but a single homogeneous slab with no
thermal coupling), this solver additionally models:

- an elastic entry zone (no layer has yielded yet, the whole stack behaves
  as one elastic body under uniform incoming layers - see the module-level
  note below), then layers yielding as the reduction proceeds, each with
  its *own* independent horizontal-stress equation once split off from the
  bulk, with a mirrored unloading cascade at exit;
- a mixed Coulomb/stiction friction law (Bay & Wanheim) at the roll
  interface *and* between adjacent layers (relative sliding from their
  different local velocities, using a fixed inner-friction coefficient);
- full per-layer thermal coupling (conduction to neighbors and to the
  rolls, deformation heat, friction heat, convective transport), coupled
  back into each layer's own flow stress.

Roll flattening is out of scope here, same as for the other solvers in this
plugin: this solver reads ``Roll.working_radius``
once and does not re-derive or iterate on it. A separate plugin providing a
Hitchcock (or other) flattened radius via that hook is picked up
transparently; this plugin's own ``contact_length_over_mean_thickness`` (see
``condition.py``) is only ever used to *classify* a pass for dispatch
(``foil_rolling_condition``, ``thick_slab_condition``), never fed back into
a calculation.

All quantities are SI, matching ``pyroll-core`` conventions. Layers are
indexed ``0..layer_count-1`` from the (arbitrarily chosen) "top" roll to
the "bottom" roll; the two roll contacts are boundaries ``0`` and
``layer_count``, inner layer-to-layer boundaries are ``1..layer_count-1``.

Each pass starts every layer from the incoming ``Profile``'s own scalar
strain/temperature (uniform across layers) - ``pyroll-core`` has no native
concept of a through-thickness state to carry from one pass to the next.
Even so, layers plastify in genuine sequence rather than all at once
("Schmiedekreuz"/forging-cross effect): every layer tracks its *own*
independent horizontal-stress DOF from the very start of contact (``entry_position``),
not just after some layer first yields, because only the outermost layers
feel roll-contact friction directly (coefficient ``coulomb_friction_coefficient``
/ ``friction_stiction_coefficient``) while inner layers only feel the fixed,
near-full-sticking inner-friction law (``MU_R_INNER``, equivalent to a
stiction coefficient of 1) on *both* their boundaries - so their stress
states diverge, and cross the yield surface at different roll-gap
positions, before any of them yields. Which layers yield first depends on
how the outer roll friction compares to the fixed inner one: with a
moderate-to-high roll friction and light reductions this can still mean
the surface layers lead (the classic Schmiedekreuz picture), but because
``MU_R_INNER`` is fixed at the sticking limit, inner layers - which have
*two* such boundaries against the surface layers' one - are just as often
found to yield first in practice; this is an emergent property of the
ported model's fixed inner-friction assumption, not tuned or asserted here.
"""

import logging

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq, root

from pyroll.core import RollPass

from pyroll.flat_rolling_slab_model.dynamic_flow_stress import troost_dynamic_flow_stress_correction

log = logging.getLogger(__name__)

MU_R_INNER = 0.388985
"""Fixed Coulomb-equivalent coefficient for inner (layer-to-layer) friction,
per the reference implementation."""


def bay_wanheim_coulomb_from_stiction(stiction_coefficient: float) -> float:
    """Equivalent Coulomb friction coefficient for a given stiction
    coefficient ``m``, per Bay & Wanheim (used at the roll interface when
    ``coulomb_friction_coefficient`` isn't explicitly set)."""
    m = stiction_coefficient
    return m / (1 + np.pi / 2 + np.arccos(m) + np.sqrt(1 - m ** 2))


class LayerRollingSolver:
    """Elastic-plastic, thermally-coupled layer model of a roll pass.

    Provides the same public attributes as the other solvers in this plugin
    (``roll_force_per_unit_width``,
    ``roll_torque_per_unit_width``, ``entry_velocity``, ``exit_velocity``,
    ``neutral_plane_position``, ``solution``) plus ``exit_temperature``
    (thickness-weighted mean across layers).
    """

    def __init__(
            self,
            roll_pass: RollPass,
            layer_count: int = 1,
            smoothness: float = 0.01,
            plateau_width: float = 0.0,
            pressure_smoothness: float = 0.1,
            max_neutral_point_iterations: int = 20,
            tolerance: float = 1e-3,
    ):
        self.roll_pass = roll_pass
        self.layer_count = layer_count
        self.smoothness = smoothness
        self.plateau_width = plateau_width
        self.pressure_smoothness = pressure_smoothness
        self.max_neutral_point_iterations = max_neutral_point_iterations
        self.tolerance = tolerance

        self._setup()
        self._solve()
        self._finalize()

    # ------------------------------------------------------------------ setup

    def _setup(self):
        rp = self.roll_pass
        roll = rp.roll
        profile = rp.in_profile
        layer_count = self.layer_count

        self.nominal_radius = roll.working_radius
        self.gap = rp.gap
        self.rotational_frequency = roll.rotational_frequency

        self.poissons_ratio = profile.poissons_ratio
        self.total_entry_height = profile.equivalent_height
        self.total_exit_height = rp.out_profile.equivalent_height

        # Uniform incoming state across layers (see module docstring).
        self.entry_height = np.full(layer_count, self.total_entry_height / layer_count)
        self.temperature = np.full(layer_count, profile.temperature)
        self.elastic_modulus = np.full(layer_count, profile.elastic_modulus)
        self.flow_stress_function = profile.flow_stress_function
        self.heat_capacity = profile.specific_heat_capacity
        self.thermal_conductivity = profile.thermal_conductivity
        self.density = profile.density

        self.roll_temperature = roll.temperature
        self.roll_thermal_conductivity = roll.thermal_conductivity
        self.roll_density = roll.density
        self.roll_heat_capacity = roll.specific_heat_capacity
        self.roll_heat_transfer_coefficient = rp.roll_heat_transfer_coefficient

        self.stiction_coefficient = rp.friction_stiction_coefficient
        self.coulomb_friction_coefficient = (
            rp.coulomb_friction_coefficient
            if rp.has_set_or_cached("coulomb_friction_coefficient")
            else bay_wanheim_coulomb_from_stiction(self.stiction_coefficient)
        )

        self.back_tension = rp.back_tension
        self.front_tension = rp.front_tension

        self.br_roll = np.sqrt(self.roll_thermal_conductivity * self.roll_heat_capacity * self.roll_density)

        self.dynamic_flow_stress_correction_enabled = rp.dynamic_flow_stress_correction_enabled
        if self.dynamic_flow_stress_correction_enabled:
            self.reference_velocity = 2 * np.pi * self.rotational_frequency * self.nominal_radius

    def br(self, t):
        return np.sqrt(self.thermal_conductivity * self.heat_capacity * self.density)

    # -------------------------------------------------------------- geometry

    def total_height(self, x, roll_radius):
        return self.gap + 2 * (roll_radius - np.sqrt(roll_radius ** 2 - x ** 2))

    def total_height_derivative(self, x, roll_radius):
        return 2 * x / np.sqrt(roll_radius ** 2 - x ** 2)

    def roll_surface_angle(self, x, roll_radius):
        return -np.arcsin(np.clip(x / roll_radius, -1, 1))

    def contact_length(self, roll_radius):
        return np.sqrt(max(roll_radius * (self.total_entry_height - self.gap) - (self.total_entry_height - self.gap) ** 2 / 4, 1e-30))

    def roll_surface_velocity(self, x, roll_radius):
        return 2 * np.pi * self.rotational_frequency * self.nominal_radius * np.cos(self.roll_surface_angle(x, roll_radius))

    def mean_strain_rate(self, roll_radius):
        contact_length = self.contact_length(roll_radius)
        return abs(2 * np.pi * self.rotational_frequency * roll_radius / contact_length * np.log(self.total_entry_height / self.gap))

    def flow_stress(self, t, equivalent_strain, phi_v_dot):
        flow_stress = self.flow_stress_function(strain=equivalent_strain, strain_rate=phi_v_dot, temperature=t)
        if self.dynamic_flow_stress_correction_enabled:
            flow_stress += troost_dynamic_flow_stress_correction(equivalent_strain, self.density, self.reference_velocity)
        return flow_stress

    def flow_stress_temperature_derivative(self, t, equivalent_strain, phi_v_dot, eps=1e-2):
        return (self.flow_stress(t + eps, equivalent_strain, phi_v_dot) - self.flow_stress(t - eps, equivalent_strain, phi_v_dot)) / (2 * eps)

    def flow_stress_strain_derivative(self, t, equivalent_strain, phi_v_dot, eps=1e-6):
        return (self.flow_stress(t, equivalent_strain + eps, phi_v_dot) - self.flow_stress(t, equivalent_strain - eps, phi_v_dot)) / (2 * eps)

    # --------------------------------------------------------------- friction

    def _relative_sliding_direction(self, vx_upper, vx_lower, vx_ref):
        a = np.arctan((vx_upper - vx_lower - self.plateau_width * vx_ref) / self.smoothness)
        b = np.arctan((vx_lower - vx_upper - self.plateau_width * vx_ref) / self.smoothness)
        return (a - b) / np.pi

    def _mixed_friction(self, sigma_y, kf_ref, coulomb_friction_coefficient, angle, direction_sign):
        coulomb_pressure = -sigma_y / (1 + coulomb_friction_coefficient * np.tan(angle) * direction_sign)
        critical_pressure = self.stiction_coefficient * kf_ref / (coulomb_friction_coefficient * np.sqrt(3))
        transition_weight = np.arctan(
            (coulomb_pressure - critical_pressure) / (self.pressure_smoothness * critical_pressure)
        ) / np.pi + 0.5
        shear_stress = (
                coulomb_friction_coefficient * coulomb_pressure * (1 - transition_weight) + self.stiction_coefficient * kf_ref / np.sqrt(3) * transition_weight
        ) * direction_sign
        normal_pressure = -sigma_y - shear_stress * np.tan(angle)
        return shear_stress, normal_pressure

    # ------------------------------------------------------------------ solve

    def _solve(self):
        # Roll flattening (if any) is out of scope for this solver: it is
        # the concern of whatever hookimpl provides Roll.working_radius (a
        # separate Hitchcock-flattening plugin, say) - this solver just
        # reads that hook's result once, like the other solvers in this
        # plugin already do, rather than re-deriving a flattened
        # radius itself and feeding it back into its own calculation.
        self.working_radius = self.nominal_radius
        self.section = LayerPassSection(self, self.working_radius)
        self.section.solve()

    # ---------------------------------------------------------------- results

    def _finalize(self):
        section = self.section
        self.roll_force_per_unit_width = section.force_per_width
        self.roll_torque_per_unit_width = section.torque_per_width
        self.neutral_plane_position = section.neutral_point_position
        self.solution = section.solution_dataframe()

        vxn = self.roll_surface_velocity(section.neutral_point_position, self.working_radius)
        h_at_xn = self.total_height(section.neutral_point_position, self.working_radius)
        self.entry_velocity = vxn * h_at_xn / self.total_entry_height
        self.exit_velocity = vxn * h_at_xn / self.total_exit_height
        self.forward_slip = self.exit_velocity / self.roll_surface_velocity(0.0, self.working_radius) - 1

        self.exit_temperature = section.exit_temperature


class LayerPassSection:
    """One full entry-to-exit solve of the layer model at a fixed (possibly
    Hitchcock-flattened) roll radius: finds the neutral point satisfying the
    front-tension boundary condition, and along the way, each layer's
    plastification (entry) and unloading (exit) positions."""

    def __init__(self, solver: LayerRollingSolver, roll_radius: float):
        self.solver = solver
        self.roll_radius = roll_radius
        self.layer_count = solver.layer_count
        self.contact_length = solver.contact_length(roll_radius)
        self.entry_position = -self.contact_length
        self.mean_strain_rate = solver.mean_strain_rate(roll_radius)

    # ------------------------------------------------------------- top-level

    def solve(self, xn_guess=None):
        s = self.solver

        kfm0 = s.flow_stress(s.temperature[0], 0.0, self.mean_strain_rate)
        if xn_guess is None:
            h_m = s.total_entry_height - s.gap
            alpha_n_osborn = (
                    0.5 * np.sqrt(h_m / self.roll_radius) * (3 * s.gap) / (s.total_entry_height + 2 * s.gap)
                    - 0.25 * (s.gap * h_m) / (self.roll_radius / 3 * (s.total_entry_height + 2 * s.gap))
                    + 0.275 * s.gap / (s.coulomb_friction_coefficient * self.roll_radius) * (s.front_tension / kfm0 - s.back_tension / kfm0)
            )
            xn_guess = self.roll_radius * np.sin(-alpha_n_osborn)

        def residual(neutral_point_position):
            entry = self._solve_entry(neutral_point_position)
            exit_ = self._solve_exit(neutral_point_position, entry)
            self._last_entry, self._last_exit = entry, exit_
            return exit_["sigma_xm_exit"] - s.front_tension

        neutral_point_position = self._bracket_and_solve(residual, xn_guess)
        entry, exit_ = self._last_entry, self._last_exit
        self.neutral_point_position = neutral_point_position
        self.entry = entry
        self.exit = exit_

        self.force_per_width = abs(entry["force"] + exit_["force"])
        self.torque_per_width = abs(entry["torque"] + exit_["torque"])
        self.exit_temperature = np.average(exit_["t_final"], weights=self.solver.entry_height)
        return neutral_point_position

    def _bracket_and_solve(self, residual, guess):
        lo, hi = self.entry_position * 0.999, -1e-9
        guess = np.clip(guess, lo, hi)
        widths = (abs(guess - lo) * 0.1, abs(guess - lo) * 0.3, abs(guess - lo))
        for width in widths:
            a = max(lo, guess - width)
            b = min(hi, guess + width)
            candidates = np.linspace(a, b, 7)
            values = []
            for neutral_point_position in candidates:
                try:
                    values.append(residual(neutral_point_position))
                except RuntimeError:
                    values.append(np.nan)
            bracket = _bracket_from_values(candidates, values)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    return bracket[0]
                return brentq(residual, *bracket, xtol=1e-9)
        raise RuntimeError("Layer model: could not bracket the neutral point.")

    # ------------------------------------------------------------- entry side

    def _solve_entry(self, neutral_point_position):
        s = self.solver
        layer_count = self.layer_count

        # --- pure-elastic zone: each layer already tracks its own sigma_x
        # (see _section_rhs's "not active" branch), so layers can diverge
        # from entry_position due to their own boundary friction, before any of them
        # yields - not just after (the "Schmiedekreuz"/forging-cross effect:
        # surface layers plastify before the core even under uniform
        # incoming layers, since only they feel roll-contact friction
        # directly while inner layers feel the weaker inner-friction law).
        y0 = np.concatenate([
            [s.back_tension, 0.0, (s.back_tension + 0.0) / 2], s.entry_height.copy(), s.temperature.copy(),
            np.full(layer_count, s.back_tension),
        ])
        active = frozenset()

        segments = []
        x_current = self.entry_position
        for _ in range(layer_count + 1):
            rhs, event = self._entry_rhs_and_event(active, neutral_point_position)
            sol = solve_ivp(
                rhs, [x_current, neutral_point_position], y0, events=event, dense_output=True,
                rtol=1e-8, atol=1e-6,
            )
            has_event = event is not None and sol.t_events[0].size
            x_end = sol.t_events[0][0] if has_event else neutral_point_position
            positions = np.linspace(x_current, x_end, max(2, int((x_end - x_current) / (self.contact_length / 100)) + 1))
            states = sol.sol(positions)
            segments.append((active, positions, states))
            y0 = states[:, -1]
            x_current = x_end
            if x_current >= neutral_point_position - 1e-12:
                break
            new_active = self._grow_active_set(active, y0, x_current)
            y0 = self._reshape_state_for_active(y0, active, new_active)
            active = new_active
        else:
            log.warning("Layer model: entry zone-chain hit max switches before reaching neutral_point_position.")

        force, torque = _integrate_force_torque(segments, self._sigma_y_of_state)
        return {
            "segments": segments,
            "state_at_xn": y0,
            "active_at_xn": active,
            "force": force,
            "torque": torque,
        }

    def _entry_rhs_and_event(self, active, neutral_point_position):
        s = self.solver
        layer_count = self.layer_count
        elastic = [i for i in range(layer_count) if i not in active]

        def rhs(x, y):
            return self._section_rhs(x, y, active, elastic, neutral_point_position, direction=1)

        if len(active) == layer_count:
            return rhs, None

        def event(x, y, *args):
            # Fire as soon as the CLOSEST-to-yield elastic layer reaches the
            # Mises boundary (max margin, not min): tracking the min would
            # instead require the SAFEST layer to reach zero first, by which
            # point every closer-to-yield layer has already overshot past
            # zero under the (deliberately non-physical past yield) elastic
            # extrapolation - masking genuinely sequential yielding as one
            # simultaneous jump.
            sigma_y, sigma_x_layers = self._decompose_state(y, active, elastic)
            margins = []
            for idx, i in enumerate(elastic):
                t_i = y[3 + layer_count + i]
                phi_v_i = _equivalent_strain(s.entry_height[i], y[3 + i])
                kf_i = s.flow_stress(t_i, phi_v_i, self.mean_strain_rate)
                margins.append((sigma_x_layers[i] - sigma_y) - 2 / np.sqrt(3) * kf_i)
            return max(margins) if margins else 1.0

        event.terminal = True
        event.direction = 1
        return rhs, event

    def _grow_active_set(self, active, y, x):
        s = self.solver
        layer_count = self.layer_count
        elastic = [i for i in range(layer_count) if i not in active]
        sigma_y, sigma_x_layers = self._decompose_state(y, active, elastic)
        new_active = set(active)
        for i in elastic:
            t_i = y[3 + layer_count + i]
            phi_v_i = _equivalent_strain(s.entry_height[i], y[3 + i])
            kf_i = s.flow_stress(t_i, phi_v_i, self.mean_strain_rate)
            if (sigma_x_layers[i] - sigma_y) >= 2 / np.sqrt(3) * kf_i - 1e-6 * max(abs(kf_i), 1.0):
                new_active.add(i)
        return frozenset(new_active) if new_active else active

    def _decompose_state(self, y, active, elastic):
        """Return (sigma_y, {layer index -> its own sigma_x}) for the given
        state vector layout (see _section_rhs).

        Every not-yet-active ("elastic") layer has its own tracked sigma_x
        DOF regardless of whether *any* layer is active yet - this is what
        lets layers diverge, and yield in sequence, even under uniform
        incoming conditions (see module docstring). ``active`` empty is
        therefore unambiguous: it just means no layer has plastified (yet,
        on entry; or any more, on exit's final recovery zone), and sigma_y
        is a genuinely tracked DOF at y[1] rather than derived."""
        layer_count = self.layer_count
        sigma_xm = y[0]
        sigma_x_layers = {}
        offset = 3 + 2 * layer_count
        for idx, i in enumerate(elastic):
            sigma_x_layers[i] = y[offset + idx]
        if not active:
            return y[1], sigma_x_layers
        # sigma_y from mean-Mises over ALL layer_count layers (elastic and plastic
        # alike, each with its own current H/T), per the reference's
        # MisesMean - not just the active (plastic) subset. For layer_count=1 this
        # coincides with "active only" (active is always empty or the full
        # single-layer set there), which is why that case never exposed
        # this: the mean was silently wrong only once some but not all of
        # several layers had plastified.
        h = y[3:3 + layer_count]
        t_all = y[3 + layer_count:3 + 2 * layer_count]
        kf_all = {}
        for i in range(layer_count):
            phi_v_i = _equivalent_strain(self.solver.entry_height[i], h[i])
            kf_all[i] = self.solver.flow_stress(t_all[i], phi_v_i, self.mean_strain_rate)
        mean_kf = sum(kf_all[i] * h[i] for i in range(layer_count)) / sum(h)
        sigma_y = sigma_xm - 2 / np.sqrt(3) * mean_kf
        for i in active:
            sigma_x_layers[i] = sigma_y + 2 / np.sqrt(3) * kf_all[i]
        return sigma_y, sigma_x_layers

    # -------------------------------------------------------------- exit side

    def _solve_exit(self, neutral_point_position, entry):
        s = self.solver
        layer_count = self.layer_count
        active = entry["active_at_xn"]
        y0 = entry["state_at_xn"]
        far_end = self.contact_length * 1.5

        segments = []
        x_current = neutral_point_position

        if active:
            # Plastic layers unload back to elastic around the roll gap's
            # geometric minimum (x=0): dh/dx (hence each active layer's own,
            # hardness-partitioned share of it) changes sign there from
            # closing to opening, and plastic flow cannot follow the gap
            # back open - only elastic recovery can. This is also exactly
            # the reference implementation's own starting guess for the
            # exit unload points (``Current[XB] = Table[X2, NS]`` with
            # X2=0) before its own iterative refinement; using it directly,
            # unrefined, avoids a stress-based event that (verified
            # numerically) can't actually detect this transition - an
            # active layer's own sigma_x is forced to satisfy the Mises
            # relation exactly by construction, so any margin computed from
            # it is degenerate and never crosses zero.
            elastic = [i for i in range(layer_count) if i not in active]

            def rhs(x, y):
                return self._section_rhs(x, y, active, elastic, neutral_point_position, direction=-1)

            sol = solve_ivp(rhs, [x_current, 0.0], y0, dense_output=True, rtol=1e-8, atol=1e-6)
            positions = np.linspace(x_current, 0.0, max(2, int((0.0 - x_current) / (self.contact_length / 100)) + 1))
            states = sol.sol(positions)
            segments.append((active, positions, states))
            y0 = self._to_pure_elastic_state(states[:, -1], active)
            x_current = 0.0

        # final elastic recovery zone until separation (sigma_y -> 0)
        rhs, event = self._exit_rhs_and_event(frozenset(), neutral_point_position, separation_event=True)
        sol = solve_ivp(
            rhs, [x_current, far_end], y0, events=event,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        x_end = sol.t_events[0][0] if sol.t_events[0].size else far_end
        positions = np.linspace(x_current, x_end, max(2, int((x_end - x_current) / (self.contact_length / 100)) + 1))
        states = sol.sol(positions)
        segments.append((frozenset(), positions, states))
        y_final = states[:, -1]

        force, torque = _integrate_force_torque(segments, self._sigma_y_of_state)
        sigma_xm_exit = y_final[0]
        t_final = y_final[3 + layer_count:3 + 2 * layer_count]
        return {
            "segments": segments,
            "sigma_xm_exit": sigma_xm_exit,
            "force": force,
            "torque": torque,
            "t_final": t_final,
        }

    def _exit_rhs_and_event(self, active, neutral_point_position, separation_event=False):
        s = self.solver
        layer_count = self.layer_count
        elastic = [i for i in range(layer_count) if i not in active]

        def rhs(x, y):
            return self._section_rhs(x, y, active, elastic, neutral_point_position, direction=-1)

        if separation_event:
            def event(x, y, *args):
                if not active:
                    return y[1]  # sigma_y == 0 (pure elastic layout: y[1]=SigmaY)
                sigma_y, _ = self._decompose_state(y, active, elastic)
                return sigma_y

            event.terminal = True
            event.direction = 1
            return rhs, event

        return rhs, None

    def _to_pure_elastic_state(self, y, active_still_notionally):
        """Convert from the active (some layers still Mises-forced) layout
        into the "not active" layout - still with every layer's own sigma_x
        tracked (see _section_rhs), just with sigma_y newly a real,
        independent DOF instead of Mises-derived. Only ever called with a
        non-empty ``active_still_notionally`` (the layers still plastic
        right before this final recovery-to-separation zone)."""
        layer_count = self.layer_count
        sigma_xm = y[0]
        active = frozenset(active_still_notionally)
        elastic = [i for i in range(layer_count) if i not in active]
        sigma_y, sigma_x_layers = self._decompose_state(y, active, elastic)
        sigma_z = (sigma_xm + sigma_y) / 2
        h = y[3:3 + layer_count]
        t = y[3 + layer_count:3 + 2 * layer_count]
        new_sigma_x = [sigma_x_layers[i] for i in range(layer_count)]
        return np.concatenate([[sigma_xm, sigma_y, sigma_z], h, t, new_sigma_x])

    def _sigma_y_of_state(self, y, active):
        layer_count = self.layer_count
        elastic = [i for i in range(layer_count) if i not in active]
        sigma_y, _ = self._decompose_state(y, active, elastic)
        return sigma_y

    def _interface_shear_stress(self, x, y, active):
        """Shear stress at the (representative, top) roll-material interface
        for reporting - the same mixed Coulomb/stiction law used inside the
        ODEs (see _section_rhs), evaluated once for an already-computed
        state rather than as part of a derivative."""
        s = self.solver
        layer_count = self.layer_count
        roll_radius = self.roll_radius
        elastic = [i for i in range(layer_count) if i not in active]

        angle = s.roll_surface_angle(x, roll_radius)
        vx_roll = s.roll_surface_velocity(x, roll_radius)
        vxn = s.roll_surface_velocity(self.neutral_point_position, roll_radius)

        h = y[3:3 + layer_count]
        t = y[3 + layer_count:3 + 2 * layer_count]
        sigma_y, _ = self._decompose_state(y, active, elastic)

        phi_v_0 = _equivalent_strain(s.entry_height[0], h[0])
        kf_0 = s.flow_stress(t[0], phi_v_0, self.mean_strain_rate)

        h_at_xn = s.total_height(self.neutral_point_position, roll_radius)
        vx_0 = vxn * (h_at_xn * s.entry_height[0] / s.total_entry_height) / h[0]

        top_direction_sign = s._relative_sliding_direction(vx_roll, vx_0, vxn)
        shear_top, _ = s._mixed_friction(sigma_y, kf_0, s.coulomb_friction_coefficient, angle, top_direction_sign)
        return shear_top

    def _mean_equivalent_strain(self, y):
        """Thickness-weighted mean equivalent strain across all layer_count layers,
        for reporting alongside the stress curves (mirrors exit_temperature's
        weighting)."""
        layer_count = self.layer_count
        h = y[3:3 + layer_count]
        equivalent_strain = _equivalent_strain(self.solver.entry_height, h)
        return float(np.average(equivalent_strain, weights=h))

    def _reshape_state_for_active(self, y, old_active, new_active):
        """Re-express a state vector at a zone boundary where the active
        (plastic) layer set grows (entry-side yielding), keeping the shared
        padded layout (SigmaXM, [pad, pad], H(layer_count), T(layer_count), SigmaX(elastic
        layers)) intact so every segment's state vector has a well-defined,
        consistent length and index meaning regardless of which layers are
        active. A layer newly entering ``active`` needs no special
        initialization of its own - it has no independent sigma_x DOF once
        active (see _decompose_state) - only the survivors' existing
        elastic DOFs need to be picked out and kept in order."""
        layer_count = self.layer_count
        sigma_xm = y[0]
        h = y[3:3 + layer_count]
        t = y[3 + layer_count:3 + 2 * layer_count]
        old_elastic = [i for i in range(layer_count) if i not in old_active]
        _, sigma_x_layers = self._decompose_state(y, frozenset(old_active), old_elastic)

        new_elastic = [i for i in range(layer_count) if i not in new_active]
        new_sigma_x = [sigma_x_layers[i] for i in new_elastic]
        return np.concatenate([[sigma_xm, 0.0, 0.0], h, t, new_sigma_x])

    # -------------------------------------------------------- shared physics

    def _section_rhs(self, x, y, active, elastic, neutral_point_position, direction):
        """RHS for a section with a fixed active (plastic) layer set.

        State layout, always: [SigmaXM, SigmaY-or-pad, SigmaZ-or-pad, H(layer_count),
        T(layer_count), SigmaX(elastic layers, in order)] - "elastic" meaning
        "not yet active", which is *all* layer_count layers before any of them have
        plastified. SigmaY/SigmaZ (indices 1, 2) are real, independently
        tracked DOFs only while no layer is active; once any layer is
        active, sigma_y is instead derived from the Mises-mean condition
        (see _decompose_state) and those two slots are unused padding.
        """
        s = self.solver
        layer_count = self.layer_count
        roll_radius = self.roll_radius

        htot = s.total_height(x, roll_radius)
        dhtot = s.total_height_derivative(x, roll_radius)
        angle = s.roll_surface_angle(x, roll_radius)
        vx_roll = s.roll_surface_velocity(x, roll_radius)
        vxn = s.roll_surface_velocity(neutral_point_position, roll_radius)

        sigma_xm = y[0]
        h = y[3:3 + layer_count]
        t = y[3 + layer_count:3 + 2 * layer_count]
        sigma_y, sigma_x_layers = self._decompose_state(y, active, elastic)

        equivalent_strain = _equivalent_strain(s.entry_height, h)
        kf_layers = np.array([s.flow_stress(t[i], equivalent_strain[i], self.mean_strain_rate) for i in range(layer_count)])

        h_at_xn = s.total_height(neutral_point_position, roll_radius)
        h_rigid_at_xn = h_at_xn * s.entry_height / s.total_entry_height
        vx_layers = vxn * h_rigid_at_xn / h

        top_direction_sign = s._relative_sliding_direction(vx_roll, vx_layers[0], vxn)
        bottom_direction_sign = s._relative_sliding_direction(vx_layers[-1], vx_roll, vxn)
        shear_top, pressure_top = s._mixed_friction(sigma_y, kf_layers[0], s.coulomb_friction_coefficient, angle, top_direction_sign)
        # Bottom roll surface: -angle, not angle - its own local slope is the
        # negative of the top's (Y = -total_height/2 vs Y = +total_height/2), so this is the
        # same "-(dY/dx) of this boundary's own y-position" convention used
        # for inner_angle and the per-layer _boundary_terms below. Combined
        # with bottom_direction_sign's own sign (from the opposite vx_upper/vx_lower
        # argument order), this makes pressure_bottom == pressure_top and
        # shear_bottom == -shear_top under symmetric conditions, exactly as the
        # d_sigma_xm formula below already assumes (shear_top - shear_bottom,
        # -pressure_top*tan(angle) - pressure_bottom*tan(angle)).
        shear_bottom, pressure_bottom = s._mixed_friction(sigma_y, kf_layers[-1], s.coulomb_friction_coefficient, -angle, bottom_direction_sign)

        d_sigma_xm = -(sigma_xm * dhtot + shear_top - shear_bottom - pressure_top * np.tan(angle) - pressure_bottom * np.tan(angle)) / htot

        # inner boundary friction/pressure and angles (only used for layer_count > 1)
        inner_shear = np.zeros(max(layer_count - 1, 0))
        inner_pressure = np.zeros(max(layer_count - 1, 0))
        inner_angle = np.zeros(max(layer_count - 1, 0))
        if layer_count > 1:
            cum = np.concatenate([[0.0], np.cumsum(h)])
            y_boundaries = htot / 2 - cum
            dcum = np.concatenate([[0.0], np.cumsum(self._dh_dx(active, elastic, h, t, equivalent_strain, kf_layers, dhtot, x, roll_radius))])
            dy_boundaries = dhtot / 2 - dcum
            for j in range(1, layer_count):
                inner_angle[j - 1] = -np.arctan(dy_boundaries[j])
                tsign = s._relative_sliding_direction(vx_layers[j - 1], vx_layers[j], vxn)
                kf_ref = min(kf_layers[j - 1], kf_layers[j])
                shear_j, pressure_j = s._mixed_friction(sigma_y, kf_ref, MU_R_INNER, inner_angle[j - 1], tsign)
                inner_shear[j - 1] = shear_j
                inner_pressure[j - 1] = pressure_j

        dh = self._dh_dx(active, elastic, h, t, equivalent_strain, kf_layers, dhtot, x, roll_radius)

        dt = np.zeros(layer_count)
        for i in range(layer_count):
            shear_upper = shear_top if i == 0 else inner_shear[i - 1]
            pressure_upper = pressure_top if i == 0 else inner_pressure[i - 1]
            shear_lower = shear_bottom if i == layer_count - 1 else inner_shear[i]
            t_upper = self.solver.roll_temperature if i == 0 else t[i - 1]
            t_lower = self.solver.roll_temperature if i == layer_count - 1 else t[i + 1]
            vx_upper = vx_roll if i == 0 else vx_layers[i - 1]
            vx_lower = vx_roll if i == layer_count - 1 else vx_layers[i + 1]

            dqu = kf_layers[i] * abs(vx_layers[i] * _equivalent_strain_derivative(s.entry_height[i], h[i], dh[i])) * h[i]
            br_i = s.br(t[i])
            br_upper = s.br_roll if i == 0 else s.br(t[i - 1])
            br_lower = s.br_roll if i == layer_count - 1 else s.br(t[i + 1])
            heat_transfer_coefficient = s.roll_heat_transfer_coefficient
            if i == 0:
                dq_upper = -heat_transfer_coefficient * (t_upper - t[i])
            else:
                dq_upper = -s.thermal_conductivity * 2 * (t_upper - t[i]) / (h[i - 1] + h[i])
            if i == layer_count - 1:
                dq_lower = -heat_transfer_coefficient * (t[i] - t_lower)
            else:
                dq_lower = -s.thermal_conductivity * 2 * (t[i] - t_lower) / (h[i] + h[i + 1])

            dqr_upper = abs(shear_upper * (vx_upper - vx_layers[i]))
            dqr_lower = abs(shear_lower * (vx_layers[i] - vx_lower))

            weight_upper = br_i / (br_upper + br_i)
            weight_lower = br_i / (br_i + br_lower)

            numerator = dqu + dq_lower + weight_upper * dqr_upper + weight_lower * dqr_lower - dq_upper
            denom = s.density * s.heat_capacity * vxn * h_rigid_at_xn[i]
            dt[i] = numerator / denom

        def _boundary_terms(i):
            # angle_lower at the bottom roll surface (i == layer_count - 1) is the
            # NEGATIVE of the top's angle here too, for the same reason
            # shear_bottom/pressure_bottom's own computation above already uses
            # -angle: this boundary's own local slope (Y = -total_height/2) is the
            # negative of the top's (Y = +total_height/2) by symmetry. Confirmed by
            # requiring this per-layer equation reduce to the standard
            # aggregate form when layer_count=1 (upper=lower boundary values): that
            # only holds with angle_lower=-angle.
            shear_upper = shear_top if i == 0 else inner_shear[i - 1]
            pressure_upper = pressure_top if i == 0 else inner_pressure[i - 1]
            shear_lower = shear_bottom if i == layer_count - 1 else inner_shear[i]
            pressure_lower = pressure_bottom if i == layer_count - 1 else inner_pressure[i]
            angle_upper = angle if i == 0 else inner_angle[i - 1]
            angle_lower = -angle if i == layer_count - 1 else inner_angle[i]
            return shear_upper, pressure_upper, shear_lower, pressure_lower, angle_upper, angle_lower

        # Every not-yet-active ("elastic") layer has its own tracked sigma_x
        # DOF and its own boundary tau/normal_pressure (roll-contact friction for the
        # outermost layers, the weaker inner-friction law for inner ones),
        # so this is shared between the pure pre-yield zone (elastic == all
        # layer_count layers) and the mixed zones (elastic == the not-yet-active
        # subset) - the mechanism that lets layers diverge, and yield in
        # sequence, from entry_position onward (see module docstring).
        d_sigma_x_elastic = []
        for i in elastic:
            shear_upper, pressure_upper, shear_lower, pressure_lower, angle_upper, angle_lower = _boundary_terms(i)
            sigma_x_i = sigma_x_layers[i]
            dsx = -(sigma_x_i * dh[i] + shear_upper - shear_lower - pressure_upper * np.tan(angle_upper) + pressure_lower * np.tan(angle_lower)) / h[i]
            d_sigma_x_elastic.append(dsx)

        if not active:
            d_eps_y = dhtot / s.total_entry_height
            mean_elastic_modulus = 1 / np.sum((s.entry_height / s.total_entry_height) / s.elastic_modulus)
            d_sigma_y = mean_elastic_modulus / (1 - s.poissons_ratio ** 2) * d_eps_y + s.poissons_ratio / (1 - s.poissons_ratio) * d_sigma_xm
            d_sigma_z = s.poissons_ratio * (d_sigma_y + d_sigma_xm)
            return np.concatenate([[d_sigma_xm, d_sigma_y, d_sigma_z], dh, dt, d_sigma_x_elastic])

        return np.concatenate([[d_sigma_xm, 0.0, 0.0], dh, dt, d_sigma_x_elastic])

    def _dh_dx(self, active, elastic, h, t, equivalent_strain, kf_layers, dhtot, x, roll_radius):
        s = self.solver
        layer_count = self.layer_count
        mean_elastic_modulus = 1 / np.sum((s.entry_height / s.total_entry_height) / s.elastic_modulus)
        dh = np.zeros(layer_count)
        if not active:
            for i in range(layer_count):
                dh[i] = mean_elastic_modulus / s.elastic_modulus[i] * s.entry_height[i] / s.total_entry_height * dhtot
            return dh
        plastic = sorted(active)
        if len(plastic) == 1:
            share = {plastic[0]: 1.0}
        else:
            share = _hardness_partition(
                plastic, h, s.entry_height, kf_layers,
            )
        for i in elastic:
            dh[i] = mean_elastic_modulus / s.elastic_modulus[i] * s.entry_height[i] / s.total_entry_height * dhtot
        remaining = dhtot - sum(dh[i] for i in elastic)
        for i in plastic:
            dh[i] = share[i] * remaining
        return dh

    def solution_dataframe(self):
        rows_x, rows_p, rows_tau, rows_strain = [], [], [], []
        for segs, label in ((self.entry["segments"], 1), (self.exit["segments"], -1)):
            for active, positions, states in segs:
                for k, x in enumerate(positions):
                    sigma_y = self._sigma_y_of_state(states[:, k], active)
                    rows_x.append(x)
                    rows_p.append(-sigma_y)
                    rows_tau.append(self._interface_shear_stress(x, states[:, k], active))
                    rows_strain.append(self._mean_equivalent_strain(states[:, k]))
        order = np.argsort(rows_x)
        x_arr = np.array(rows_x)[order]
        p_arr = np.array(rows_p)[order]
        tau_arr = np.array(rows_tau)[order]
        strain_arr = np.array(rows_strain)[order]
        return pd.DataFrame(
            {
                "vertical_stress": -p_arr,
                "normal_pressure": p_arr,
                "shear_stress": tau_arr,
                "equivalent_strain": strain_arr,
            },
            index=pd.Index(x_arr, name="x"),
        )

    def solution_dataframe_per_layer(self):
        """Per-layer breakdown of the solution, keyed by layer index: each
        layer's own horizontal stress ``sigma_x`` (the DOF that actually
        diverges between layers - see module docstring), its own
        temperature and equivalent strain, plus the shared ``sigma_y``
        (vertical stress, common to all layers once any has yielded) as a
        reference line. Lets each layer's approach to, and departure from,
        the Mises yield surface be inspected individually - the mechanism
        behind sequential (Schmiedekreuz) yielding - rather than only the
        thickness-aggregated view in ``solution_dataframe``."""
        layer_count = self.layer_count
        rows_x = []
        rows_sigma_x = [[] for _ in range(layer_count)]
        rows_sigma_y = []
        rows_t = [[] for _ in range(layer_count)]
        rows_phi_v = [[] for _ in range(layer_count)]
        rows_active = [[] for _ in range(layer_count)]
        for segs in (self.entry["segments"], self.exit["segments"]):
            for active, positions, states in segs:
                elastic = [i for i in range(layer_count) if i not in active]
                for k, x in enumerate(positions):
                    y = states[:, k]
                    sigma_y, sigma_x_layers = self._decompose_state(y, active, elastic)
                    h = y[3:3 + layer_count]
                    t = y[3 + layer_count:3 + 2 * layer_count]
                    rows_x.append(x)
                    rows_sigma_y.append(sigma_y)
                    for i in range(layer_count):
                        rows_sigma_x[i].append(sigma_x_layers[i])
                        rows_t[i].append(t[i])
                        rows_phi_v[i].append(_equivalent_strain(self.solver.entry_height[i], h[i]))
                        rows_active[i].append(i in active)
        order = np.argsort(rows_x)
        x_arr = np.array(rows_x)[order]
        sigma_y_arr = np.array(rows_sigma_y)[order]
        result = {}
        for i in range(layer_count):
            result[i] = pd.DataFrame(
                {
                    "sigma_x": np.array(rows_sigma_x[i])[order],
                    "sigma_y": sigma_y_arr,
                    "temperature": np.array(rows_t[i])[order],
                    "equivalent_strain": np.array(rows_phi_v[i])[order],
                    "active": np.array(rows_active[i])[order],
                },
                index=pd.Index(x_arr, name="x"),
            )
        return result


def _equivalent_strain(entry_height, h):
    return 2 / np.sqrt(3) * np.log(entry_height / h)


def _equivalent_strain_derivative(entry_height, h, dh):
    return -2 / np.sqrt(3) * dh / h


def _hardness_partition(plastic_layers, h, entry_height, kf_layers):
    """Numerically solves the hardness-based thickness-partition system
    (``LayerThicknessDeltasCalculate`` in the reference implementation):
    plastic layers with lower flow stress ("softer") thin proportionally
    more than "harder" ones, following a power-law relation derived from
    equal-work partitioning between neighboring groups."""
    n = len(plastic_layers)
    if n == 1:
        return {plastic_layers[0]: 1.0}

    def residuals(shares):
        shares = np.abs(shares)
        shares = shares / shares.sum()
        res = [shares.sum() - 1.0]
        for i in range(n - 1):
            upper = plastic_layers[: i + 1]
            lower = plastic_layers[i + 1:]
            h_upper = sum(h[j] for j in upper)
            h_lower = sum(h[j] for j in lower)
            kf_upper = sum(kf_layers[j] * h[j] for j in upper) / h_upper
            kf_lower = sum(kf_layers[j] * h[j] for j in lower) / h_lower
            eps_upper = 1 - h_upper / sum(entry_height[j] for j in upper)
            eps_lower = 1 - h_lower / sum(entry_height[j] for j in lower)
            delta_upper = sum(shares[plastic_layers.index(j)] for j in upper)
            delta_lower = sum(shares[plastic_layers.index(j)] for j in lower)
            if kf_upper > kf_lower:
                harder, softer = (h_upper, kf_upper, eps_upper), (h_lower, kf_lower, eps_lower)
                d_harder, d_softer = delta_upper, delta_lower
            else:
                harder, softer = (h_lower, kf_lower, eps_lower), (h_upper, kf_upper, eps_upper)
                d_harder, d_softer = delta_lower, delta_upper
            h_h, kf_h, eps_h = harder
            h_s, kf_s, eps_s = softer
            exponent = (1 - eps_s) * (1 - eps_h)
            target_ratio = (h_s / h_h) * (kf_h / kf_s * np.sqrt(h_s / h_h)) ** exponent
            actual_ratio = d_softer / max(d_harder, 1e-12)
            res.append(actual_ratio - target_ratio)
        return res

    guess = np.full(n, 1.0 / n)
    sol = root(residuals, guess, method="hybr")
    shares = np.abs(sol.x)
    shares = shares / shares.sum()
    return {plastic_layers[i]: shares[i] for i in range(n)}


def _bracket_from_values(candidates, values):
    for i in range(len(candidates) - 1):
        v0, v1 = values[i], values[i + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        if v0 == 0:
            return candidates[i], candidates[i]
        if v0 * v1 < 0:
            return candidates[i], candidates[i + 1]
    return None


def _integrate_force_torque(segments, sigma_y_func):
    all_positions, all_pressures = [], []
    for active, positions, states in segments:
        for k in range(positions.shape[0]):
            all_positions.append(positions[k])
            all_pressures.append(-sigma_y_func(states[:, k], active))
    all_positions = np.array(all_positions)
    all_pressures = np.array(all_pressures)
    order = np.argsort(all_positions)
    all_positions, all_pressures = all_positions[order], all_pressures[order]
    force = trapezoid(all_pressures, all_positions)
    torque = trapezoid(all_pressures * all_positions, all_positions)
    return force, torque
