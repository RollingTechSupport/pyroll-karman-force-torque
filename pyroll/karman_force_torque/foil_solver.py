"""Elastic-plastic foil-rolling model.

Implements the Fleck/Johnson/Sutcliffe-type model described in Mauk & Overhagen,
"Prozessmodell zum Kaltwalzen von Metallfolien mit keramischen Arbeitswalzen auf
Mehrwalzengerüsten" (2013), and the accompanying MATLAB implementation by
C. Overhagen ("lee_sutcliffe"). Unlike :class:`.karman_solver.KarmanSolver`, this
solver does not assume the roll stays circular in the contact zone: the roll-gap
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
the yield condition ``S + P = k_f / k_fe`` used in the ODEs below.
"""

import logging

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from pyroll.core import RollPass

log = logging.getLogger(__name__)


def _influence_matrix(n: int, step: float) -> np.ndarray:
    """Dimensionless elastic half-space influence-coefficient matrix ``D_ij - D_1j``
    (eq. 34/35 of the report; ``Dij.m`` in the reference implementation).

    All elastic material properties cancel out of the non-dimensional form of the
    problem (verified by substitution into eq. 6), so the only remaining
    parameter is the dimensionless element width ``step`` (the "C" of eq. 35,
    which the reference MATLAB code re-purposes as the grid spacing rather than
    an elastic constant). The additive constant ``D0`` (eq. 35) is arbitrary and
    cancels exactly through the ``D_ij - D_1j`` differencing, so it is omitted.
    """
    idx = np.arange(n)
    k = idx[:, None] - idx[None, :]
    k_row0 = idx[0] - idx[None, :]

    def d(k):
        k1, k2, k3 = k + 1, k - 1, k
        with np.errstate(divide="ignore"):
            lk1 = np.where(k1 == 0, 0.0, np.log(k1.astype(float) ** 2))
            lk2 = np.where(k2 == 0, 0.0, np.log(k2.astype(float) ** 2))
            lk3 = np.where(k3 == 0, 0.0, np.log(k3.astype(float) ** 2))
        return -step / (2 * np.pi) * (k1 ** 2 * lk1 + k2 ** 2 * lk2 - 2 * k3 ** 2 * lk3)

    d_ij = d(k)
    d_1j = d(k_row0)
    return d_ij - d_1j


class FoilRollingSolver:
    """Elastic-plastic solution of the foil-rolling problem for a roll pass.

    Provides the same public attributes as :class:`.karman_solver.KarmanSolver`
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

    # ------------------------------------------------------------------ setup

    def _setup(self):
        rp = self.roll_pass
        roll = rp.roll
        profile = rp.in_profile

        self.radius = roll.working_radius
        self.mu = rp.coulomb_friction_coefficient

        e_r, nu_r = roll.elastic_modulus, roll.poissons_ratio
        e_s, nu_s = profile.elastic_modulus, profile.poissons_ratio
        self.nu_s = nu_s
        self.erp = e_r / (1 - nu_r ** 2)
        self.esp = e_s / (1 - nu_s ** 2)

        self.h0 = profile.equivalent_height
        self.h1 = rp.out_profile.equivalent_height

        total_strain = 2 / np.sqrt(3) * np.log(self.h0 / self.h1)
        self.kf = profile.flow_stress_function(
            strain=total_strain / 2,
            strain_rate=rp.strain_rate,
            temperature=profile.temperature,
        )

        z0 = rp.back_tension
        z1 = rp.front_tension
        self.kfe = self.kf - 0.5 * (z0 + z1)
        if self.kfe <= 0:
            raise ValueError(
                "Combined entry/exit tension exceeds the flow stress; "
                "the foil rolling model requires kfe = kf - 0.5*(back_tension+front_tension) > 0."
            )

        self.x_scale = self.erp / (self.radius * self.kfe)
        self.t_scale = self.erp ** 2 / (self.radius * self.kfe ** 2)

        self.t0 = self.h0 * self.t_scale
        self.t1 = self.h1 * self.t_scale
        self.s0 = z0 / self.kfe
        self.s1 = z1 / self.kfe
        self.kf_ratio = self.kf / self.kfe
        self.u = self.mu * self.erp / self.kfe

        alpha = (1 - nu_s) / (1 - 2 * nu_s)
        self.c1 = alpha / (2 - alpha * (1 - 2 * nu_r) / (1 - nu_r) * self.esp / self.erp)

        # Wide dimensionless arc, generously larger than the classical (Hitchcock)
        # contact-length estimate, matching sutcliffe.m's factor-of-3 margin.
        classical_half_length = np.sqrt((self.h0 - self.h1) * self.radius)
        half_width = 3 * classical_half_length * self.x_scale
        self.grid_x = np.linspace(-half_width, half_width, self.element_count)
        self.grid_step = self.grid_x[1] - self.grid_x[0]
        self.influence_matrix = _influence_matrix(self.element_count, self.grid_step)

        # Undeformed (circular) roll-gap shape, referenced such that the classical
        # (single-radius) entry point matches t0.
        entry_guess = -classical_half_length * self.x_scale
        self.t_min_base = self.t0 - entry_guess ** 2
        self.base_shape = self.t_min_base + self.grid_x ** 2

        self.current_shape = self.base_shape.copy()
        self.entry_guess = entry_guess
        self.neutral_guess = entry_guess / 5

    # ------------------------------------------------------------ zone physics

    def _rhs_elastic(self, x, y, sign, t_of_x, dt_of_x):
        s, p = y
        t = t_of_x(x)
        dt = dt_of_x(x)
        ds = -(s + p) * dt / t - sign * 2 * self.u * p / t
        dp = -self.esp / self.kfe * dt / t + sign * self.nu_s / (1 - self.nu_s) * 2 * self.u * p / t
        return [ds, dp]

    def _rhs_plastic(self, x, y, sign, t_of_x, dt_of_x):
        p = y[0]
        t = t_of_x(x)
        dt = dt_of_x(x)
        q = sign * self.u * p
        dp = self.kf_ratio * dt / t + 2 * q / t
        return [dp]

    def _rhs_sticking(self, x, y, t_of_x, dt_of_x):
        t = t_of_x(x)
        dt = dt_of_x(x)
        dp = -self.c1 * self.esp / self.kfe * dt / t
        return [dp]

    def _sticking_event(self, dt_of_x, direction):
        """Stick/slip switch event: ``|Q_haft| - |Q_slip|`` crosses zero when the
        friction sliding would demand exceeds/undercuts the sticking capacity.

        Using a signed ``direction`` (rather than detecting either-direction
        crossings) is essential here: right after this event fires and a new
        integration segment starts exactly at the crossing point, re-evaluating
        the same zero-crossing quantity without a direction constraint finds a
        spurious immediate root at the starting point itself (the value is ~0
        there by construction), causing the stick/slip zones to ping-pong in
        zero-length steps instead of integrating forward.
        """

        def event(x, y, *args):
            p = y[0] if len(y) == 1 else y[1]
            q_haft = -self.c1 * self.esp / self.kfe / 2 * dt_of_x(x)
            return abs(q_haft) - abs(p * self.u)

        event.terminal = True
        event.direction = direction
        return event

    @staticmethod
    def _yield_event(kf_ratio):
        def event(x, y, *args):
            return y[0] + y[1] - kf_ratio

        event.terminal = True
        event.direction = 0
        return event

    @staticmethod
    def _horizontal_tangent_event(dt_of_x):
        def event(x, y, *args):
            return dt_of_x(x)

        event.terminal = True
        event.direction = 0
        return event

    @staticmethod
    def _pressure_zero_event():
        def event(x, y, *args):
            return y[-1]

        event.terminal = True
        event.direction = 0
        return event

    # -------------------------------------------------------------- zone chain

    def _solve_zone_chain(self, xa, xn, t_of_x, dt_of_x):
        """Integrate all five zones from ``xa`` (entry) through ``xn`` (neutral
        point) to separation, returning the exit-tension residual against the
        prescribed front tension plus the full stress trace for reporting."""
        segments = []

        # 1: elastic entry (sliding)
        sol = solve_ivp(
            self._rhs_elastic, [xa, xn], [self.s0, 0.0], args=(1, t_of_x, dt_of_x),
            events=self._yield_event(self.kf_ratio), dense_output=True, max_step=self.grid_step,
        )
        if sol.t_events[0].size:
            xb = sol.t_events[0][0]
            xs = np.linspace(xa, xb, max(2, int((xb - xa) / self.grid_step) + 1))
        else:
            xb = xn
            xs = np.linspace(xa, xb, max(2, int((xb - xa) / self.grid_step) + 1))
        ys = sol.sol(xs)
        segments.append(("elastic", xs, ys[1], ys[0]))
        p_current = ys[1, -1]
        x_current = xb

        # 2/3: plastic sliding (Nacheilzone), with possible sticking zone(s)
        for _ in range(6):
            if x_current >= xn - 1e-12:
                break
            sol = solve_ivp(
                self._rhs_plastic, [x_current, xn], [p_current], args=(1, t_of_x, dt_of_x),
                events=self._sticking_event(dt_of_x, direction=-1), dense_output=True, max_step=self.grid_step,
            )
            x_end = sol.t_events[0][0] if sol.t_events[0].size else xn
            xs = np.linspace(x_current, x_end, max(2, int((x_end - x_current) / self.grid_step) + 1))
            ys = sol.sol(xs)
            segments.append(("plastic_slip", xs, ys[0], None))
            p_current = ys[0, -1]
            x_current = x_end
            if x_current >= xn - 1e-12:
                break

            sol = solve_ivp(
                self._rhs_sticking, [x_current, xn], [p_current], args=(t_of_x, dt_of_x),
                events=self._sticking_event(dt_of_x, direction=1), dense_output=True, max_step=self.grid_step,
            )
            x_end = sol.t_events[0][0] if sol.t_events[0].size else xn
            xs = np.linspace(x_current, x_end, max(2, int((x_end - x_current) / self.grid_step) + 1))
            ys = sol.sol(xs)
            segments.append(("sticking", xs, ys[0], None))
            p_current = ys[0, -1]
            x_current = x_end

        # 4: plastic sliding (Voreilzone), forward slip, until horizontal tangent
        far_end = self.grid_x[-1]
        sol = solve_ivp(
            self._rhs_plastic, [xn, far_end], [p_current], args=(-1, t_of_x, dt_of_x),
            events=self._horizontal_tangent_event(dt_of_x), dense_output=True, max_step=self.grid_step,
        )
        if not sol.t_events[0].size:
            raise RuntimeError("Foil rolling model: could not locate the horizontal roll-gap tangent (Xc).")
        xc = sol.t_events[0][0]
        xs = np.linspace(xn, xc, max(2, int((xc - xn) / self.grid_step) + 1))
        ys = sol.sol(xs)
        segments.append(("plastic_slip", xs, ys[0], None))
        p_at_xc = ys[0, -1]

        # 5: elastic recovery, until separation (p=0)
        s_at_xc = self.kf_ratio - p_at_xc
        sol = solve_ivp(
            self._rhs_elastic, [xc, far_end], [s_at_xc, p_at_xc], args=(-1, t_of_x, dt_of_x),
            events=self._pressure_zero_event(), dense_output=True, max_step=self.grid_step,
        )
        if not sol.t_events[0].size:
            raise RuntimeError("Foil rolling model: could not locate the exit separation point (Xd).")
        xd = sol.t_events[0][0]
        xs = np.linspace(xc, xd, max(2, int((xd - xc) / self.grid_step) + 1))
        ys = sol.sol(xs)
        segments.append(("elastic", xs, ys[1], ys[0]))
        s_at_xd = ys[0, -1]

        residual = s_at_xd - self.s1
        return residual, xd, segments

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

    def _gap_minimum(self, xa, dt_of_x):
        """Upper bound for the neutral point: the (first, moving rightward from
        Xa) point where the *current* roll-gap shape has a horizontal tangent.
        The neutral point must lie strictly before this - it is not, in general,
        at X=0, which is merely where the undeformed reference parabola happens
        to be centered and drifts arbitrarily as the elastic correction is
        applied, so it must never be used as a search bound."""
        candidates = np.linspace(xa + 1e-6 * abs(xa), self.grid_x[-1] * 0.999, 60)
        values = dt_of_x(candidates)
        for i in range(len(candidates) - 1):
            if values[i] == 0:
                return candidates[i]
            if values[i] * values[i + 1] < 0:
                return brentq(lambda x: float(dt_of_x(x)), candidates[i], candidates[i + 1], xtol=1e-10)
        return self.grid_x[-1] * 0.999

    def _find_neutral_point(self, xa, t_of_x, dt_of_x):
        def residual(xn):
            return self._solve_zone_chain(xa, xn, t_of_x, dt_of_x)[0]

        upper_bound = self._gap_minimum(xa, dt_of_x)

        # Warm-start: the neutral point moves only slightly between successive
        # outer (elastic-flattening) iterations, and T(X) can develop local
        # non-monotonicities once flattening is significant, so a wide scan from
        # the domain edge risks locking onto a spurious root far from the
        # physically continued solution. Try a narrow bracket around the
        # previous iteration's value first, widening it if necessary, before
        # falling back to a full scan of (xa, upper_bound).
        guess = np.clip(self.neutral_guess, xa + 1e-6 * abs(xa), upper_bound - 1e-6 * abs(xa))
        for half_width in (0.05, 0.2, 0.5, 1.0, 2.0):
            lo = max(xa + 1e-6 * abs(xa), guess - half_width)
            hi = min(upper_bound - 1e-6 * abs(xa), guess + half_width)
            if hi <= lo:
                continue
            candidates = np.linspace(lo, hi, 9)
            try:
                values = [residual(xn) for xn in candidates]
            except RuntimeError:
                continue
            bracket = self._bracket_from_candidates(candidates, values)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    return bracket[0]
                return brentq(residual, *bracket, xtol=1e-10)

        candidates = np.linspace(xa + 1e-6 * abs(xa), upper_bound - 1e-6 * abs(xa), 25)
        values = []
        for xn in candidates:
            try:
                values.append(residual(xn))
            except RuntimeError:
                values.append(np.nan)

        bracket = self._bracket_from_candidates(candidates, values)
        if bracket is None:
            raise RuntimeError("Foil rolling model: could not bracket the neutral point (Xn).")
        if bracket[0] == bracket[1]:
            return bracket[0]
        return brentq(residual, *bracket, xtol=1e-10)

    def _find_entry_point(self, t_of_x):
        def residual(xa):
            return t_of_x(xa) - self.t0

        # Warm-start around the previous iteration's entry point first (see the
        # rationale in _find_neutral_point), falling back to a full domain scan.
        guess = np.clip(self.entry_guess, self.grid_x[0] * 0.999, -1e-9)
        for half_width in (0.05, 0.2, 0.5, 1.0, 2.0):
            lo = max(self.grid_x[0] * 0.999, guess - half_width)
            hi = min(-1e-9, guess + half_width)
            if hi <= lo:
                continue
            candidates = np.linspace(lo, hi, 9)
            values = [residual(xa) for xa in candidates]
            bracket = self._bracket_from_candidates(candidates, values)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    return bracket[0]
                return brentq(residual, *bracket, xtol=1e-10)

        candidates = np.linspace(self.grid_x[0] * 0.999, -1e-9, 20)
        values = [residual(xa) for xa in candidates]
        bracket = self._bracket_from_candidates(candidates, values)
        if bracket is None:
            raise RuntimeError("Foil rolling model: could not bracket the entry point (Xa).")
        if bracket[0] == bracket[1]:
            return bracket[0]
        return brentq(residual, *bracket, xtol=1e-10)

    # -------------------------------------------------------------- outer loop

    def _solve_outer_loop(self):
        base_reference = self.base_shape.copy()
        shape = self.current_shape

        for iteration in range(self.max_outer_iterations):
            t_of_x = CubicSpline(self.grid_x, shape)
            dt_of_x = t_of_x.derivative()

            xa = self._find_entry_point(t_of_x)
            xn = self._find_neutral_point(xa, t_of_x, dt_of_x)
            _, xd, segments = self._solve_zone_chain(xa, xn, t_of_x, dt_of_x)

            xs_all = np.concatenate([seg[1] for seg in segments])
            p_all = np.concatenate([seg[2] for seg in segments])
            fill_value = p_all.min()
            pressures = np.interp(self.grid_x, xs_all, p_all, left=fill_value, right=fill_value)

            b = self.influence_matrix @ pressures
            new_shape = base_reference + 2 * b

            t_at_xd = float(np.interp(xd, self.grid_x, new_shape))
            delta = t_at_xd - self.t1
            new_shape = new_shape - delta
            base_reference = base_reference - delta

            relaxed_shape = self.relaxation_factor * new_shape + (1 - self.relaxation_factor) * shape
            change = np.max(np.abs(relaxed_shape - shape))

            shape = relaxed_shape
            self.entry_guess, self.neutral_guess = xa, xn

            if change < self.tolerance:
                log.debug(f"Foil rolling model converged after {iteration + 1} outer iterations.")
                break
        else:
            log.warning(
                f"Foil rolling model did not converge within {self.max_outer_iterations} outer iterations "
                f"(last change {change:.3e})."
            )

        self.current_shape = shape
        self.entry_point_dimless = xa
        self.neutral_point_dimless = xn
        self.exit_point_dimless = xd
        self.final_segments = segments

    # ---------------------------------------------------------------- results

    def _finalize(self):
        t_of_x = CubicSpline(self.grid_x, self.current_shape)
        dt_of_x = t_of_x.derivative()

        xs, p_dimless, sign_of_x, zone_of_x = [], [], [], []
        for zone, xz, pz, _ in self.final_segments:
            xs.append(xz)
            p_dimless.append(pz)
            sign = 1.0 if np.mean(xz) < self.neutral_point_dimless else -1.0
            sign_of_x.append(np.full_like(xz, sign))
            zone_of_x.append([zone] * len(xz))

        xs = np.concatenate(xs)
        p_dimless = np.concatenate(p_dimless)
        signs = np.concatenate(sign_of_x)
        zones = np.concatenate(zone_of_x)

        order = np.argsort(xs)
        xs, p_dimless, signs, zones = xs[order], p_dimless[order], signs[order], zones[order]

        x_physical = xs / self.x_scale
        pressure = p_dimless * self.kfe

        dh_dx = dt_of_x(xs) * self.kfe / self.erp
        shear = np.where(
            zones == "sticking",
            self.c1 * self.esp / 2 * dh_dx,
            signs * self.mu * pressure,
        )
        vertical_stress = -pressure

        h_of_x = t_of_x(xs) / self.t_scale
        equivalent_strain = 2 / np.sqrt(3) * np.log(self.h0 / h_of_x)

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
                "gap_height": self.current_shape / self.t_scale,
                "rigid_gap_height": self.base_shape / self.t_scale,
            },
            index=pd.Index(self.grid_x / self.x_scale, name="x"),
        )

        self.roll_force_per_unit_width = float(trapezoid(pressure, x_physical))
        self.roll_torque_per_unit_width = float(trapezoid(shear, x_physical))

        self.neutral_plane_position = self.neutral_point_dimless / self.x_scale
        self.entry_position = self.entry_point_dimless / self.x_scale
        self.exit_position = self.exit_point_dimless / self.x_scale

        # Kinematics via mass continuity, with the roll surface velocity projected
        # onto the horizontal using the *rigid* roll angle at the neutral point
        # (the report neglects this projection entirely for foil rolling, eq. 41-43,
        # since angles are tiny there; keeping it matches KarmanSolver's convention
        # and is a negligible correction in the foil-rolling regime anyway).
        rotational_frequency = self.roll_pass.roll.rotational_frequency
        neutral_angle = -np.arcsin(np.clip(self.neutral_plane_position / self.radius, -1, 1))
        neutral_velocity = 2 * np.pi * rotational_frequency * self.radius * np.cos(neutral_angle)
        neutral_height = float(t_of_x(self.neutral_point_dimless)) / self.t_scale

        self.entry_velocity = neutral_velocity * neutral_height / self.h0
        self.exit_velocity = neutral_velocity * neutral_height / self.h1
        self.forward_slip = self.exit_velocity / (2 * np.pi * rotational_frequency * self.radius) - 1
