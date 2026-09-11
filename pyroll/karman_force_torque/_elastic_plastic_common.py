"""Shared elastic-plastic zone-chain machinery for
:class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver` and
:class:`.orowan_solver.OrowanSolver`.

Both solvers integrate von Kármán's own horizontal equilibrium (the same
equation, unchanged) with Bay & Wanheim's mixed Coulomb/sticking friction
law, bounded by Hooke's-law elastic entry/exit zones and a shooting search
for the neutral point (matching the boundary tensions) - they differ only
in the algebraic closure relating sigma_x to sigma_y: the plain Mises
relation for ``KarmanMixedFrictionSolver``, Orowan's inhomogeneity-
corrected one for ``OrowanSolver``. This module factors out everything
except that closure.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

from pyroll.core import RollPass


def bay_wanheim_coulomb_from_stiction(stiction_coefficient: float) -> float:
    """Equivalent Coulomb friction coefficient for a given stiction
    coefficient ``m``, per Bay & Wanheim (used when
    ``coulomb_friction_coefficient`` isn't explicitly set)."""
    m = stiction_coefficient
    return m / (1 + np.pi / 2 + np.arccos(m) + np.sqrt(1 - m ** 2))


def _phi_v(h0, h):
    return 2 / np.sqrt(3) * np.log(h0 / h)


class ElasticPlasticSolverBase:
    """Common setup, geometry and result-extraction for a single
    homogeneous cross-section, elastic-plastic, mixed-friction solver.
    Subclasses set ``section_cls`` to the :class:`ElasticPlasticSection`
    subclass implementing their own sigma_x/sigma_y closure, and provide
    the same public attributes as every other solver in this plugin
    (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``)."""

    section_cls = None

    def __init__(self, roll_pass: RollPass, pressure_smoothness: float = 0.1):
        self.roll_pass = roll_pass
        self.pressure_smoothness = pressure_smoothness
        self._setup()
        self._solve()
        self._finalize()

    # ------------------------------------------------------------------ setup

    def _setup(self):
        rp = self.roll_pass
        roll = rp.roll
        profile = rp.in_profile

        self.nominal_radius = roll.working_radius
        self.gap = rp.gap
        self.rotational_frequency = roll.rotational_frequency

        self.mr = rp.friction_stiction_coefficient
        self.mu_r = (
            rp.coulomb_friction_coefficient
            if rp.has_set_or_cached("coulomb_friction_coefficient")
            else bay_wanheim_coulomb_from_stiction(self.mr)
        )

        self.nu_m = profile.poissons_ratio
        self.em = profile.elastic_modulus
        self.h0_tot = profile.equivalent_height
        self.t0 = profile.temperature
        self.flow_stress_function = profile.flow_stress_function

        self.back_tension = rp.back_tension
        self.front_tension = rp.front_tension

    # -------------------------------------------------------------- geometry

    def h_tot(self, x, rw):
        return self.gap + 2 * (rw - np.sqrt(rw ** 2 - x ** 2))

    def dh_tot_dx(self, x, rw):
        return 2 * x / np.sqrt(rw ** 2 - x ** 2)

    def alpha_w(self, x, rw):
        return -np.arcsin(np.clip(x / rw, -1, 1))

    def contact_length(self, rw):
        return np.sqrt(max(rw * (self.h0_tot - self.gap) - (self.h0_tot - self.gap) ** 2 / 4, 1e-30))

    def vw(self, x, rw):
        return 2 * np.pi * self.rotational_frequency * self.nominal_radius * np.cos(self.alpha_w(x, rw))

    def phi_dot_vm(self, rw):
        ld = self.contact_length(rw)
        return abs(2 * np.pi * self.rotational_frequency * rw / ld * np.log(self.h0_tot / self.gap))

    def kf(self, phi_v, phi_v_dot):
        return self.flow_stress_function(strain=phi_v, strain_rate=phi_v_dot, temperature=self.t0)

    # ------------------------------------------------------------------ solve

    def _solve(self):
        self.working_radius = self.nominal_radius
        self.section = self.section_cls(self, self.working_radius)
        self.section.solve()

    # ---------------------------------------------------------------- results

    def _finalize(self):
        section = self.section
        self.roll_force_per_unit_width = section.force_per_width
        self.roll_torque_per_unit_width = section.torque_per_width
        self.neutral_plane_position = section.xn
        self.solution = section.solution_dataframe()
        self.entry_position = section.x0
        self.exit_position = section.x_end
        h_xn = self.h_tot(section.xn, section.rw)
        vxn = self.vw(section.xn, section.rw)
        self.entry_velocity = vxn * h_xn / self.h_tot(section.x0, section.rw)
        self.exit_velocity = vxn * h_xn / self.h_tot(section.x_end, section.rw)


class ElasticPlasticSection:
    """Shared zone-chain/shooting machinery. Subclasses implement
    ``_closure_sigma_y(sigma_x, alpha, kf_val, zone)``, the algebraic
    relation closing von Kármán's horizontal equilibrium (plain Mises for
    ``KarmanMixedFrictionSolver``, Orowan's inhomogeneity-corrected
    relation for ``OrowanSolver``); everything else (elastic zones,
    plastic zones, entry/exit zone chains, the neutral-point shooting
    search, force/torque integration and the reported solution) is
    identical between them.

    ``zone`` is +1 on entry (Vorlauf, pre-neutral-point, material slower
    than the roll) and -1 on exit (Nachlauf, post-neutral-point, material
    faster), consistently determining friction direction throughout rather
    than a velocity-difference sign.
    """

    def __init__(self, solver: ElasticPlasticSolverBase, rw: float):
        self.solver = solver
        self.rw = rw
        self.h0 = solver.h0_tot
        self.ld = solver.contact_length(rw)
        self.x0 = -self.ld

    # ------------------------------------------------------------ constitutive

    def _closure_sigma_y(self, sigma_x, alpha, kf_val, zone):
        raise NotImplementedError

    def _margin(self, sigma_x, sigma_y_elastic, alpha, kf_val, zone):
        """sigma_y(elastic) minus the closure's own predicted sigma_y at
        the same sigma_x - the yield-onset event target (rather than the
        plain Mises criterion for OrowanSolver, so its elastic/plastic
        transition stays continuous with its own closure)."""
        return sigma_y_elastic - self._closure_sigma_y(sigma_x, alpha, kf_val, zone)

    def _kf_at(self, h):
        s = self.solver
        # Clipped at 0: h can momentarily exceed h0 during the exit zone's
        # adaptive-step elastic recovery, before the separation event (at
        # sigma_y=0) actually fires - equivalent strain has no meaning past
        # that point, and letting phi_v go negative feeds flow-stress
        # functions (e.g. Freiberg's power-law form) a negative base raised
        # to a fractional exponent, producing NaN that then poisons the
        # rest of the integration.
        phi_v = max(_phi_v(self.h0, h), 0.0)
        return s.kf(phi_v, self.phi_dot_vm)

    @property
    def phi_dot_vm(self):
        return self.solver.phi_dot_vm(self.rw)

    # ---------------------------------------------------------- shared physics

    def _mixed_friction(self, sigma_y, alpha, kf_val, zone):
        """Bay & Wanheim's mixed Coulomb/sticking friction law, identical
        in form to ``_mixed_friction``/``_mixed_friction_bw`` elsewhere in
        this plugin (layer_solver.py, karman_mixed_friction_solver.py).
        Returns (tau, pn, phi) - phi (the Coulomb-to-sticking transition
        weight, 0 at low pressure, 1 once sticking-limited) is reused by
        ``OrowanSection`` to blend its inhomogeneity function."""
        s = self.solver
        sign = 1.0 if zone > 0 else -1.0
        pn_coulomb = -sigma_y / (1 + s.mu_r * np.tan(alpha) * sign)
        pn_critical = s.mr * kf_val / (s.mu_r * np.sqrt(3))
        phi = np.arctan((pn_coulomb - pn_critical) / (s.pressure_smoothness * pn_critical)) / np.pi + 0.5
        tau = (s.mu_r * pn_coulomb * (1 - phi) + s.mr * kf_val / np.sqrt(3) * phi) * sign
        pn = -sigma_y - tau * np.tan(alpha)
        return tau, pn, phi

    def _elastic_rhs(self, x, y, zone):
        """Elastic compression/recovery (Hooke's law) with Bay & Wanheim
        friction already acting (well-behaved as pressure -> 0, unlike
        assuming sticking friction where contact pressure is still
        essentially zero), same construction as LayerRollingSolver's
        ns=1 pure-elastic branch."""
        s = self.solver
        sigma_x, sigma_y, sigma_z, h, t = y
        rw = self.rw
        htot = s.h_tot(x, rw)
        dhtot = s.dh_tot_dx(x, rw)
        alpha = s.alpha_w(x, rw)
        kf_val = self._kf_at(h)

        tau, pn, _ = self._mixed_friction(sigma_y, alpha, kf_val, zone)
        d_sigma_x = -(sigma_x * dhtot + 2 * tau - 2 * pn * np.tan(alpha)) / htot

        d_eps_y = dhtot / self.h0
        em_star = s.em / (1 - s.nu_m ** 2)
        d_sigma_y = em_star * d_eps_y + s.nu_m / (1 - s.nu_m) * d_sigma_x
        d_sigma_z = s.nu_m * (d_sigma_y + d_sigma_x)
        return [d_sigma_x, d_sigma_y, d_sigma_z, dhtot, 0.0]

    def _plastic_rhs(self, x, y, zone):
        """Von-Kármán's own horizontal equilibrium, integrating sigma_x
        with Bay & Wanheim mixed friction and sigma_y/normal pressure
        derived at each step via the subclass's own closure."""
        s = self.solver
        rw = self.rw
        sigma_x, h, t = y

        htot = s.h_tot(x, rw)
        dhtot = s.dh_tot_dx(x, rw)
        alpha = s.alpha_w(x, rw)
        kf_val = self._kf_at(h)

        sigma_y = self._closure_sigma_y(sigma_x, alpha, kf_val, zone)
        tau, pn, _ = self._mixed_friction(sigma_y, alpha, kf_val, zone)

        d_sigma_x = -(sigma_x * dhtot + 2 * tau - 2 * pn * np.tan(alpha)) / htot
        return [d_sigma_x, dhtot, 0.0]

    # ------------------------------------------------------------------- zones

    def _solve_entry(self, xn):
        s = self.solver
        y0 = [s.back_tension, 0.0, s.back_tension / 2, self.h0, s.t0]

        def elastic_event(x, y):
            sigma_x, sigma_y, sigma_z, h, t = y
            alpha = s.alpha_w(x, self.rw)
            kf_val = self._kf_at(h)
            return self._margin(sigma_x, sigma_y, alpha, kf_val, zone=1)

        elastic_event.terminal = True
        elastic_event.direction = 1

        # The margin can already be >= 0 at x0 itself (yield stress reached
        # essentially at first contact - the elastic zone is vanishingly
        # short whenever kf/E is small, as it is for typical hot-rolling
        # parameters; LayerRollingSolver's own ns=1 case shows the same
        # near-instant transition). scipy's direction=1 event only fires on
        # a negative-to-positive crossing *during* integration, so it does
        # not fire when the sign is already positive at the start - handle
        # that explicitly instead of letting the (very steep, post-yield)
        # elastic-formula slope run uncontrolled all the way to xn.
        if elastic_event(self.x0, y0) >= 0:
            segments = [("elastic", np.array([self.x0]), np.array(y0).reshape(-1, 1))]
            return self._solve_entry_plastic(self.x0, xn, y0[0], y0[3], y0[4], segments)

        sol = solve_ivp(
            lambda x, y: self._elastic_rhs(x, y, zone=1), [self.x0, xn], y0, events=elastic_event,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        has_event = sol.t_events[0].size > 0
        x_yield = sol.t_events[0][0] if has_event else xn
        xs_elastic = np.linspace(self.x0, x_yield, max(2, int((x_yield - self.x0) / (self.ld / 100)) + 1))
        ys_elastic = sol.sol(xs_elastic)

        segments = [("elastic", xs_elastic, ys_elastic)]

        if x_yield >= xn - 1e-12:
            return {"segments": segments, "sigma_x_at_xn": ys_elastic[0, -1], "h_at_xn": ys_elastic[3, -1],
                    "t_at_xn": ys_elastic[4, -1]}

        return self._solve_entry_plastic(x_yield, xn, ys_elastic[0, -1], ys_elastic[3, -1], ys_elastic[4, -1],
                                          segments)

    def _solve_entry_plastic(self, x_yield, xn, sigma_x_yield, h_yield, t_yield, segments):
        y0_plastic = [sigma_x_yield, h_yield, t_yield]
        sol_plastic = solve_ivp(
            lambda x, y: self._plastic_rhs(x, y, zone=1), [x_yield, xn], y0_plastic,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        xs_plastic = np.linspace(x_yield, xn, max(2, int((xn - x_yield) / (self.ld / 100)) + 1))
        ys_plastic = sol_plastic.sol(xs_plastic)
        segments.append(("plastic", xs_plastic, ys_plastic))

        return {
            "segments": segments,
            "sigma_x_at_xn": ys_plastic[0, -1],
            "h_at_xn": ys_plastic[1, -1],
            "t_at_xn": ys_plastic[2, -1],
        }

    def _solve_exit(self, xn, entry):
        s = self.solver
        far_end = self.ld * 1.5

        segments = []
        if xn < -1e-12:
            y0_plastic = [entry["sigma_x_at_xn"], entry["h_at_xn"], entry["t_at_xn"]]
            sol_plastic = solve_ivp(
                lambda x, y: self._plastic_rhs(x, y, zone=-1), [xn, 0.0], y0_plastic,
                dense_output=True, rtol=1e-8, atol=1e-6,
            )
            xs_plastic = np.linspace(xn, 0.0, max(2, int((0.0 - xn) / (self.ld / 100)) + 1))
            ys_plastic = sol_plastic.sol(xs_plastic)
            segments.append(("plastic", xs_plastic, ys_plastic))
            sigma_x_0, h_0, t_0 = ys_plastic[:, -1]
        else:
            sigma_x_0, h_0, t_0 = entry["sigma_x_at_xn"], entry["h_at_xn"], entry["t_at_xn"]

        alpha_0 = s.alpha_w(0.0, self.rw)
        kf_0 = self._kf_at(h_0)
        sigma_y_0 = self._closure_sigma_y(sigma_x_0, alpha_0, kf_0, zone=-1)
        sigma_z_0 = (sigma_x_0 + sigma_y_0) / 2
        y0_elastic = [sigma_x_0, sigma_y_0, sigma_z_0, h_0, t_0]

        def separation_event(x, y):
            return y[1]

        separation_event.terminal = True
        separation_event.direction = 1

        sol_elastic = solve_ivp(
            lambda x, y: self._elastic_rhs(x, y, zone=-1), [0.0, far_end], y0_elastic, events=separation_event,
            dense_output=True, rtol=1e-8, atol=1e-6,
        )
        x_end = sol_elastic.t_events[0][0] if sol_elastic.t_events[0].size else far_end
        xs_elastic = np.linspace(0.0, x_end, max(2, int((x_end - 0.0) / (self.ld / 100)) + 1))
        ys_elastic = sol_elastic.sol(xs_elastic)
        segments.append(("elastic", xs_elastic, ys_elastic))

        return {"segments": segments, "sigma_x_exit": ys_elastic[0, -1], "x_end": x_end}

    # ------------------------------------------------------------------- solve

    def solve(self):
        s = self.solver

        def residual(xn):
            entry = self._solve_entry(xn)
            exit_ = self._solve_exit(xn, entry)
            return exit_["sigma_x_exit"] - s.front_tension

        def safe_residual(xn):
            # Off-root candidates in the coarse scan below can genuinely
            # diverge (e.g. the plastic-zone shooting integrating sigma_x
            # to extreme values for a badly-guessed xn) badly enough that
            # scipy's solve_ivp rejects a non-finite y0 outright rather than
            # producing a merely large (but sign-comparable) value - treated
            # as unusable (nan) here rather than letting that exception
            # abort the whole search.
            try:
                value = residual(xn)
            except (ValueError, RuntimeError):
                return np.nan
            return value if np.isfinite(value) else np.nan

        lo, hi = self.x0 * 0.999, -1e-9
        candidates = np.linspace(lo, hi, 12)
        values = [safe_residual(xn) for xn in candidates]
        bracket = None
        for i in range(len(candidates) - 1):
            if np.isnan(values[i]) or np.isnan(values[i + 1]):
                continue
            if values[i] == 0:
                bracket = (candidates[i], candidates[i])
                break
            if np.sign(values[i]) != np.sign(values[i + 1]):
                bracket = (candidates[i], candidates[i + 1])
                break
        if bracket is None:
            raise RuntimeError(f"{type(self).__name__}: could not bracket the neutral point (Xn).")
        xn = bracket[0] if bracket[0] == bracket[1] else brentq(safe_residual, *bracket, xtol=1e-9)

        self.xn = xn
        self.entry = self._solve_entry(xn)
        self.exit = self._solve_exit(xn, self.entry)
        self.x_end = self.exit["x_end"]

        force, torque = self._integrate_force_torque()
        self.force_per_width = force
        self.torque_per_width = torque
        return xn

    # ---------------------------------------------------------------- results

    def _state_at(self, zone_name, x, y, zone):
        """Returns (sigma_y, h, tau) for a state vector from either zone
        type - elastic states carry sigma_y directly, plastic states
        derive it (and tau) via the closure/friction law."""
        alpha = self.solver.alpha_w(x, self.rw)
        if zone_name == "elastic":
            sigma_x, sigma_y, sigma_z, h, t = y
            kf_val = self._kf_at(h)
            tau, _, _ = self._mixed_friction(sigma_y, alpha, kf_val, zone)
            return sigma_y, h, tau
        sigma_x, h, t = y
        kf_val = self._kf_at(h)
        sigma_y = self._closure_sigma_y(sigma_x, alpha, kf_val, zone)
        tau, _, _ = self._mixed_friction(sigma_y, alpha, kf_val, zone)
        return sigma_y, h, tau

    def _sigma_y_of(self, zone_name, x, y, zone):
        sigma_y, _, _ = self._state_at(zone_name, x, y, zone)
        return sigma_y

    def _integrate_force_torque(self):
        force = 0.0
        torque = 0.0
        for zone_sign, res in ((1, self.entry), (-1, self.exit)):
            for zone_name, xs, ys in res["segments"]:
                sigma_y_vals = np.array([
                    self._sigma_y_of(zone_name, xs[k], ys[:, k], zone_sign) for k in range(xs.shape[0])
                ])
                force += trapezoid(-sigma_y_vals, xs)
                alpha_vals = self.solver.alpha_w(xs, self.rw)
                rw_local = np.sqrt(np.maximum(self.rw ** 2 - xs ** 2, 0.0))
                torque += trapezoid(np.abs(-sigma_y_vals) * rw_local * np.tan(np.abs(alpha_vals)), xs)
        return abs(force), abs(torque)

    def solution_dataframe(self):
        rows_x, rows_p, rows_tau, rows_strain = [], [], [], []
        for zone_sign, res in ((1, self.entry), (-1, self.exit)):
            for zone_name, xs, ys in res["segments"]:
                for k, x in enumerate(xs):
                    sigma_y, h, tau = self._state_at(zone_name, x, ys[:, k], zone_sign)
                    rows_x.append(x)
                    rows_p.append(-sigma_y)
                    rows_tau.append(tau)
                    rows_strain.append(_phi_v(self.h0, h))
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
