"""Orowan's 1943 slab theory for hot rolling, with Bay & Wanheim mixed
friction and elastic entry/exit zones.

Ports Orowan's own model (Orowan, "The Calculation of Roll Pressure in Hot
and Cold Flat Rolling", Proc. IMechE 150, 1943), as reproduced in Overhagen's
2018 dissertation (TU Duisburg-Essen), section 4.4.1.2 - the equations this
module implements are that section's Gl. (4.4/1), (4.4/6) and (4.4/7).

Unlike :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
(a friction-law refinement on top of the ordinary straight-strip von-Kármán
slab), this solver implements Orowan's own, geometrically different
construction: circular arcs centered on the roll-gap symmetry line, meeting
the roll surfaces at right angles; a polar coordinate system on each arc;
and an inhomogeneity function omega_O(alpha, a_O) (an integral over the
arc's polar angle) that replaces the ordinary Mises plane-strain offset
sigma_x - sigma_N = -2*kf/sqrt(3) with a friction- and geometry-dependent
one. This is what makes it capture genuinely non-uniform (redundant)
through-thickness shear, rather than assuming uniform deformation across
the slab the way the mixed-friction and layer models both still do (even
the layer model's per-layer split is a coarse discretization, not the same
mechanism).

This solver always uses Bay & Wanheim's mixed Coulomb/sticking friction law
(the same law :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
and :class:`.layer_solver.LayerRollingSolver` use), rather than Orowan's own
sticking-only Gl. (4.4/6) or his general sliding branch, Gl. (4.4/5) (with
its own friction-ratio parameter a_O = 2*mu*sigma_N/kf in [0, 1] blending
between sliding and sticking *within his own inhomogeneity-function
formalism*). Instead, this keeps Gl. (4.4/6)'s structure (so the geometric
correction term and the flow-stress convention are unchanged) and
generalizes only the inhomogeneity function itself: the sticking-limit
value omega_O(alpha, 1) (see ``omega_orowan_sticking``) is linearly blended
toward the frictionless limit omega_O -> 1 using Bay & Wanheim's own smooth
Coulomb/sticking transition variable Phi (the same Phi computed inside the
mixed friction law, reused rather than duplicated), i.e.
omega_O_eff = (1-Phi) + Phi*omega_O(alpha,1). This is a deliberate,
pragmatic hybrid - not a rederivation of Gl. (4.4/5) - chosen because
omega_O is documented as only weakly dependent on its arguments (see
below), so a linear interpolation between its two known closed-form
endpoints, weighted by the same regime indicator already driving the
friction law, is a reasonable approximation without needing the general,
quadrature-only omega_O(alpha, a_O) for a_O != 0, 1.

Elastic entry and exit zones (Hooke's law, with Bay & Wanheim friction
already acting - see :mod:`._elastic_plastic_common`) bound the
Orowan-plastic zone on both sides, unlike Orowan's own 1943 paper (and the
mixed-friction/layer models here so far), which are rigid-plastic or use a
much simpler elastic treatment. Thermal coupling is out of scope here
(like :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver` - a
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
  limit - relevant here since the mixed friction law approaches that same
  limit as friction drops toward the Coulomb-only regime.
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

from scipy.special import jv
import numpy as np

from pyroll.core import RollPass

from pyroll.karman_force_torque._elastic_plastic_common import (
    ElasticPlasticSolverBase,
    ElasticPlasticSection,
)

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


class OrowanSolver(ElasticPlasticSolverBase):
    """Orowan's (1943) slab theory for hot rolling, with elastic entry/exit
    zones and Bay & Wanheim mixed Coulomb/sticking friction throughout -
    see module docstring for exactly how the two are combined.

    Provides the same public attributes as the other solvers in this
    plugin (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``).
    """

    def __init__(self, roll_pass: RollPass, pressure_smoothness: float = 0.1):
        self.section_cls = OrowanPassSection
        super().__init__(roll_pass=roll_pass, pressure_smoothness=pressure_smoothness)


class OrowanPassSection(ElasticPlasticSection):
    """Orowan's Gl. (4.4/6) inhomogeneity-corrected closure, generalized to
    Bay & Wanheim mixed friction via the omega_O blend described in the
    module docstring."""

    def _closure_sigma_y(self, sigma_x, alpha, kf_val, zone):
        """Solves Orowan's Gl. (4.4/6) for sigma_y, using
        kf_orowan = 2*kf_val (see module docstring). ``zone`` is +1 on
        entry (Vorlauf, pre-neutral-point) and -1 on exit (Nachlauf,
        post-neutral-point).

        Gl. (4.4/6) is written sigma_x = sigma_N - kf_orowan/sqrt(3)*[...],
        which only reduces to this codebase's own established Mises
        relation (sigma_x = sigma_y + 2*kf/sqrt(3), used throughout
        layer_solver.py) if Orowan's sigma_x and sigma_N are both
        compression-positive (sigma_N = -sigma_y, sigma_x_orowan =
        -sigma_x_mine) - a common convention in the older engineering
        literature this section is translating, unlike this codebase's
        tension-positive sigma_x. Substituting gives
        sigma_x_mine = sigma_y + kf_orowan/sqrt(3)*[...], i.e. solved for
        sigma_y: sigma_y = sigma_x - kf_orowan/sqrt(3)*[...]. (An earlier
        version of this method used sigma_y = -sigma_x - offset, i.e.
        assumed sigma_x itself was already compression-negative like this
        codebase's convention while sigma_N alone got the sign flip - that
        is inconsistent and was caught because it produced a normal
        pressure that *decreased* from entry into the plastic zone instead
        of forming the expected friction hill.)

        The sticking inhomogeneity value omega_O(alpha,1) is linearly
        blended toward the frictionless limit 1.0 using Bay & Wanheim's own
        smooth transition variable Phi - which itself depends on sigma_y,
        the very quantity being solved for, so this iterates to a fixed
        point. Phi only depends on sigma_y through pn_coulomb, and omega_O
        varies weakly with its argument (1.0 to ~0.785), so this converges
        in a handful of iterations regardless of starting point."""
        omega_sticking = omega_orowan_sticking(alpha)
        correction = _geometric_correction(alpha)
        sign = 1.0 if zone > 0 else -1.0

        def sigma_y_for(omega):
            offset = (2 * kf_val / np.sqrt(3)) * (omega - sign * 0.5 * correction)
            return sigma_x - offset

        sigma_y = sigma_y_for(omega_sticking)
        for _ in range(10):
            _, _, phi = self._mixed_friction(sigma_y, alpha, kf_val, zone)
            omega_eff = (1 - phi) * 1.0 + phi * omega_sticking
            new_sigma_y = sigma_y_for(omega_eff)
            if abs(new_sigma_y - sigma_y) < 1e-6 * max(abs(kf_val), 1.0):
                sigma_y = new_sigma_y
                break
            sigma_y = new_sigma_y
        return sigma_y
