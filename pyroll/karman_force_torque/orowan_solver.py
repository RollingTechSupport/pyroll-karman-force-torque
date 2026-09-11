"""Orowan's 1943 slab theory for hot rolling with sticking friction.

Ports Orowan's own model (Orowan, "The Calculation of Roll Pressure in Hot
and Cold Flat Rolling", Proc. IMechE 150, 1943), as reproduced in Overhagen's
2018 dissertation (TU Duisburg-Essen), section 4.4.1.2 - the equations this
module implements are that section's Gl. (4.4/1), (4.4/6) and (4.4/7).

Unlike :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
(a friction-law refinement on top of the ordinary straight-strip von-Kármán
slab, using Bay & Wanheim's mixed Coulomb/sticking law), this solver
implements Orowan's own, geometrically different construction: circular
arcs centered on the roll-gap symmetry line, meeting the roll surfaces at
right angles; a polar coordinate system on each arc; and an inhomogeneity
function omega_O(alpha, a_O) (an integral over the arc's polar angle) that
replaces the ordinary Mises plane-strain offset
sigma_x - sigma_N = -2*kf/sqrt(3) with a friction- and geometry-dependent
one. This is what makes it capture genuinely non-uniform (redundant)
through-thickness shear, rather than assuming uniform deformation across
the slab the way the mixed-friction and layer models both still do (even
the layer model's per-layer split is a coarse discretization, not the same
mechanism).

This solver only implements the **sticking-friction branch**, Orowan's
Gl. (4.4/6) (Haftreibung), applied throughout the contact - appropriate for
hot rolling, where scale/oxide layers and high temperature typically make
sticking friction the realistic assumption across the whole arc, per the
user's own request. Orowan's general theory also has a sliding-friction
branch, Gl. (4.4/5) (Gleitreibung), with a friction-ratio-dependent
a_O = 2*mu*sigma_N/kf in [0, 1] blending between the two; that branch, and
the resulting need to locate where the contact switches from sliding to
sticking, is not implemented here.

Elastic entry and exit zones (Hooke's law, same construction as
:class:`.layer_solver.LayerRollingSolver`'s ``layer_count=1`` case) bound
the Orowan-plastic zone on both sides, exactly as requested - unlike
Orowan's own 1943 paper (and the mixed-friction/layer models here so far),
which are rigid-plastic or use a much simpler elastic treatment. Thermal
coupling is out of scope here (like :class:`.karman_solver.KarmanSolver`
and :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver` - a
separate axis, already covered by the layer model): flow stress is
evaluated at the incoming profile's own scalar temperature throughout.

Important notes on faithfulness to the source, read before trusting this
model beyond qualitative use:

- **Gl. (4.4/7)'s printed integration limit is corrected here.** Overhagen's
  dissertation prints the inhomogeneity integral as
  omega_O(alpha,a_O) = (1/sin(alpha)) * integral_0^{a_O} of
  sqrt(1 - a_O^2*(theta/alpha)^2) * cos(theta) dtheta (upper limit a_O),
  but the same function is later restated with upper limit alpha
  (Gl. 4.4/36, the Yuen/Dixon/Nguyen approximation of the *same* omega_O).
  Integrating to a_O is not even well-defined in general (the integrand's
  square root goes negative once a_O/alpha > 1, which is the common case
  for realistic rolling angles and high a_O), and only the alpha-limit
  choice reproduces the two published reference values in Overhagen's own
  Abb. 4.4/4: omega_O -> 1 as a_O -> 0, and omega_O -> pi/4 =~ 0.785 as
  a_O -> 1, alpha -> 0 (both verified numerically against this
  implementation - see ``tests/test_orowan_solve.py``). This
  implementation integrates to alpha.
- **The flow-stress convention is translated, not copied verbatim.**
  Gl. (4.4/5)/(4.4/6) use a bare kf/sqrt(3) offset (no factor of 2),
  consistent with an older, Tresca-flavoured "plane-strain flow stress"
  convention for kf common in the classical rolling literature (Orowan's
  paper predates the Mises-vs-Tresca precision found in more modern
  treatments) - not the uniaxial kf convention ``pyroll-core``'s
  ``flow_stress_function`` and the rest of this plugin use (where the
  plane-strain relation is sigma_x - sigma_N = -/+ 2*kf/sqrt(3)). To keep
  this solver's plastic zone continuous with its own elastic-zone
  yield-onset criterion (and with the rest of this plugin), this
  implementation substitutes kf_orowan = 2 * kf_pyroll throughout, chosen
  specifically so that Gl. (4.4/6) reduces to the ordinary
  sigma_x - sigma_N = -2*kf/sqrt(3) Mises relation in the frictionless
  limit (which is consistent for the a_O-dependent Gl. 4.4/5, though not
  exercised by this sticking-only solver).
- **The entry/exit sign of the (1/alpha - 1/tan(alpha)) geometric term in
  Gl. (4.4/6) is a best-effort reconstruction.** The source prints it as
  "∓" without the surrounding context needed to pin down which sign
  belongs to the entry (Vorlauf) vs. exit (Nachlauf) zone with certainty
  from this excerpt alone, and Overhagen's own alpha in this equation
  reads as an unsigned, per-zone angle (matching how Gl. 4.4/3 writes
  separate entry/exit formulas with explicit +/- rather than a signed
  coordinate) - so this implementation always uses abs(alpha) and lets the
  explicit zone sign (+1 entry, -1 exit) carry the distinction. This term
  is small relative to the omega_O * kf/sqrt(3) term for typical rolling
  angles; if a discrepancy against a trusted reference is found, this is
  the first place to check.

Given these three points, treat this solver as a careful, documented
reconstruction of Orowan's theory - not a byte-for-byte transcription
verified against the original 1943 paper (which was not available for
comparison) - and validate against any reference data before relying on it
for anything beyond qualitative "how does the inhomogeneity correction
change things" comparisons.
"""

import logging

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq
from scipy.special import jv

from pyroll.core import RollPass

log = logging.getLogger(__name__)


def omega_orowan_sticking(alpha: float) -> float:
    """Orowan's inhomogeneity function omega_O(alpha, a_O=1) (full sticking
    friction, Gl. 4.4/7 with the corrected upper limit alpha - see module
    docstring), evaluated via the closed form

        omega_O(alpha, 1) = pi * J_1(alpha) / (2 * sin(alpha))

    (J_1 the order-1 Bessel function of the first kind), which follows from
    the standard Fourier-Bessel cosine integral
    integral_0^1 sqrt(1-t^2) cos(x*t) dt = pi*J_1(x)/(2*x) substituted into
    Gl. (4.4/7) with a_O=1. Verified numerically against direct quadrature
    of Gl. (4.4/7) to machine precision. Avoids a numerical integration in
    every ODE right-hand-side evaluation. Always evaluated at abs(alpha) -
    see module docstring on the unsigned-angle convention."""
    alpha = abs(alpha)
    if alpha < 1e-6:
        return np.pi / 4
    return np.pi * jv(1, alpha) / (2 * np.sin(alpha))


def _geometric_correction(alpha: float) -> float:
    """The (1/alpha - 1/tan(alpha)) term in Gl. (4.4/6), always evaluated at
    abs(alpha) (see module docstring), via its Taylor series
    (alpha/3 + alpha**3/45 + ...) for small alpha to avoid catastrophic
    cancellation between the two nearly-equal, individually large terms."""
    alpha = abs(alpha)
    if alpha < 1e-3:
        return alpha / 3 + alpha ** 3 / 45
    return 1 / alpha - 1 / np.tan(alpha)


class OrowanSolver:
    """Orowan's (1943) slab theory for hot rolling, sticking friction
    throughout, with elastic entry/exit zones - see module docstring.

    Provides the same public attributes as the other solvers in this
    plugin (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``).
    """

    def __init__(self, roll_pass: RollPass):
        self.roll_pass = roll_pass
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
        self.section = OrowanPassSection(self, self.working_radius)
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


def _phi_v(h0, h):
    return 2 / np.sqrt(3) * np.log(h0 / h)


class OrowanPassSection:
    def __init__(self, solver: OrowanSolver, rw: float):
        self.solver = solver
        self.rw = rw
        self.h0 = solver.h0_tot
        self.ld = solver.contact_length(rw)
        self.x0 = -self.ld

    # ------------------------------------------------------------ constitutive

    def _orowan_sigma_y(self, sigma_x, alpha, kf_val, zone):
        """Orowan's Gl. (4.4/6) (Haftreibung/sticking), solved for sigma_y,
        using kf_orowan = 2*kf_val (see module docstring). ``zone`` is +1 on
        entry (Vorlauf, pre-neutral-point) and -1 on exit (Nachlauf,
        post-neutral-point).

        Gl. (4.4/6) is written sigma_x = sigma_N - kf_orowan/sqrt(3)*[...],
        which only reduces to this codebase's own established Mises
        relation (sigma_x = sigma_y + 2*kf/sqrt(3), used throughout
        layer_solver.py and karman_solver.py) if Orowan's sigma_x and
        sigma_N are both compression-positive (sigma_N = -sigma_y,
        sigma_x_orowan = -sigma_x_mine) - a common convention in the older
        engineering literature this section is translating, unlike this
        codebase's tension-positive sigma_x. Substituting gives
        sigma_x_mine = sigma_y + kf_orowan/sqrt(3)*[...], i.e. solved for
        sigma_y: sigma_y = sigma_x - kf_orowan/sqrt(3)*[...]. (An earlier
        version of this method used sigma_y = -sigma_x - offset, i.e.
        assumed sigma_x itself was already compression-negative like this
        codebase's convention while sigma_N alone got the sign flip - that
        is inconsistent and was caught because it produced a normal
        pressure that *decreased* from entry into the plastic zone instead
        of forming the expected friction hill.)"""
        omega = omega_orowan_sticking(alpha)
        correction = _geometric_correction(alpha)
        sign = 1.0 if zone > 0 else -1.0
        offset = (2 * kf_val / np.sqrt(3)) * (omega - sign * 0.5 * correction)
        return sigma_x - offset

    def _orowan_margin(self, sigma_x, sigma_y_elastic, alpha, kf_val, zone):
        """sigma_y(elastic) minus Orowan's own predicted sigma_y at the same
        sigma_x - the yield-onset event target (chosen instead of the plain
        Mises criterion so the elastic/plastic transition is continuous)."""
        return sigma_y_elastic - self._orowan_sigma_y(sigma_x, alpha, kf_val, zone)

    def _kf_at(self, h):
        s = self.solver
        phi_v = _phi_v(self.h0, h)
        return s.kf(phi_v, self.phi_dot_vm)

    @property
    def phi_dot_vm(self):
        return self.solver.phi_dot_vm(self.rw)

    # ---------------------------------------------------------- shared physics

    def _friction(self, sigma_y, alpha, kf_val, zone):
        """Sticking friction throughout: tau = +/- kf, sign set by zone
        (matching entry material slower than the roll, exit faster) rather
        than a velocity-difference sign (no smooth Coulomb/sticking
        blending is needed for a sticking-only model)."""
        tau = kf_val * (1.0 if zone > 0 else -1.0)
        pn = -sigma_y - tau * np.tan(alpha)
        return tau, pn

    def _elastic_rhs(self, x, y, zone):
        """Elastic compression/recovery (Hooke's law), same construction as
        LayerRollingSolver's ns=1 pure-elastic branch: sigma_x stays fixed
        at its boundary-tension value while sigma_y/sigma_z build up under
        the closing gap. No friction is applied here (sigma_x held
        constant, not sticking): contact pressure is still ~0 through most
        of the elastic zone, and friction cannot exceed what pressure can
        transmit regardless of the "sticking throughout" assumption for the
        plastic zone - applying full sticking (tau=kf) from x0, where
        pressure is genuinely zero, blows the stress up immediately
        (verified: it does, badly). Friction only becomes meaningful once
        real contact pressure develops, i.e. once Orowan's own equations
        take over at yield."""
        s = self.solver
        dhtot = s.dh_tot_dx(x, self.rw)
        d_eps_y = dhtot / self.h0
        em_star = s.em / (1 - s.nu_m ** 2)
        d_sigma_y = em_star * d_eps_y
        d_sigma_z = s.nu_m * d_sigma_y
        return [0.0, d_sigma_y, d_sigma_z, dhtot, 0.0]

    def _plastic_rhs(self, x, y, zone):
        """Standard von-Kármán horizontal equilibrium (Gl. 4.4/1, the same
        equation :class:`.karman_solver.KarmanSolver` and
        :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
        already use - this is von Kármán's own 1925 equation, not Orowan's
        novel contribution), integrating sigma_x with sticking friction and
        sigma_y/normal pressure derived at each step via Orowan's
        Gl. (4.4/6)."""
        s = self.solver
        rw = self.rw
        sigma_x, h, t = y

        htot = s.h_tot(x, rw)
        dhtot = s.dh_tot_dx(x, rw)
        alpha = s.alpha_w(x, rw)
        kf_val = self._kf_at(h)

        sigma_y = self._orowan_sigma_y(sigma_x, alpha, kf_val, zone)
        tau, pn = self._friction(sigma_y, alpha, kf_val, zone)

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
            return self._orowan_margin(sigma_x, sigma_y, alpha, kf_val, zone=1)

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
        sigma_y_0 = self._orowan_sigma_y(sigma_x_0, alpha_0, kf_0, zone=-1)
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

        lo, hi = self.x0 * 0.999, -1e-9
        candidates = np.linspace(lo, hi, 12)
        values = [residual(xn) for xn in candidates]
        bracket = None
        for i in range(len(candidates) - 1):
            if values[i] == 0:
                bracket = (candidates[i], candidates[i])
                break
            if np.sign(values[i]) != np.sign(values[i + 1]):
                bracket = (candidates[i], candidates[i + 1])
                break
        if bracket is None:
            raise RuntimeError("Orowan model: could not bracket the neutral point (Xn).")
        xn = bracket[0] if bracket[0] == bracket[1] else brentq(residual, *bracket, xtol=1e-9)

        self.xn = xn
        self.entry = self._solve_entry(xn)
        self.exit = self._solve_exit(xn, self.entry)
        self.x_end = self.exit["x_end"]

        force, torque = self._integrate_force_torque()
        self.force_per_width = force
        self.torque_per_width = torque
        return xn

    def _sigma_y_of(self, zone_name, x, y, zone):
        if zone_name == "elastic":
            return y[1]
        sigma_x, h, t = y
        alpha = self.solver.alpha_w(x, self.rw)
        kf_val = self._kf_at(h)
        return self._orowan_sigma_y(sigma_x, alpha, kf_val, zone)

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
        rows_x, rows_p = [], []
        for zone_sign, res in ((1, self.entry), (-1, self.exit)):
            for zone_name, xs, ys in res["segments"]:
                for k, x in enumerate(xs):
                    sigma_y = self._sigma_y_of(zone_name, x, ys[:, k], zone_sign)
                    rows_x.append(x)
                    rows_p.append(-sigma_y)
        order = np.argsort(rows_x)
        x_arr = np.array(rows_x)[order]
        p_arr = np.array(rows_p)[order]
        return pd.DataFrame(
            {"vertical_stress": -p_arr, "normal_pressure": p_arr},
            index=pd.Index(x_arr, name="x"),
        )
