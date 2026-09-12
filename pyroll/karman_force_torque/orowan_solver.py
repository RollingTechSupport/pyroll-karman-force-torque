"""Orowan's 1943 slab theory for hot rolling, with Bay & Wanheim mixed
friction and elastic entry/exit zones.

Ports Orowan's model (Orowan, "The Calculation of Roll Pressure in Hot and
Cold Flat Rolling", Proc. IMechE 150, 1943), as reproduced in Overhagen's
2018 dissertation (TU Duisburg-Essen), section 4.4.1.2 - the equations here
are that section's Gl. (4.4/1), (4.4/6) and (4.4/7).

Unlike :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver` (a
friction-law refinement on top of the ordinary straight-strip von-Kármán
slab), Orowan's construction is geometrically different: circular arcs
centered on the roll-gap symmetry line, meeting the roll surfaces at right
angles, with a polar coordinate system on each arc. This replaces the
ordinary Mises plane-strain offset sigma_x - sigma_N = -2*flow_stress/sqrt(3)
with an inhomogeneity function omega_O(alpha, a_O) (an integral over the
arc's polar angle) that captures genuinely non-uniform through-thickness
shear, rather than assuming uniform deformation across the slab as the
mixed-friction and layer models do.

This solver always uses Bay & Wanheim's mixed Coulomb/sticking friction law
(the same law :class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
and :class:`.layer_solver.LayerRollingSolver` use), rather than Orowan's own
sticking-only Gl. (4.4/6) or his general sliding branch, Gl. (4.4/5) (with
its own friction-ratio parameter a_O = 2*mu*sigma_N/flow_stress in [0, 1]
blending between sliding and sticking within his own inhomogeneity-function
formalism). Instead, Gl. (4.4/6)'s structure is kept unchanged and only the
inhomogeneity function is generalized: the sticking-limit value
omega_O(alpha, 1) (see ``omega_orowan_sticking``) is linearly blended toward
the frictionless limit omega_O -> 1 using Bay & Wanheim's own smooth
Coulomb/sticking transition weight Phi (the same Phi the friction law
itself computes), i.e. omega_O_eff = (1-Phi) + Phi*omega_O(alpha, 1). This
is a pragmatic hybrid rather than a rederivation of Gl. (4.4/5), justified
by omega_O being only weakly dependent on its arguments (see below).

Elastic entry and exit zones (Hooke's law, with Bay & Wanheim friction
already acting) bound the plastic zone on both sides, unlike Orowan's own
1943 paper. Thermal coupling is out of scope, like
:class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`: flow
stress is evaluated at the incoming profile's own scalar temperature
throughout.

Three interpretive choices were needed to reconstruct Gl. (4.4/6)-(4.4/7)
from Overhagen's account, each affecting faithfulness to Orowan's original
1943 paper (not available for direct comparison):

- **Gl. (4.4/7)'s integration limit.** Overhagen's dissertation prints the
  inhomogeneity integral's upper limit as a_O, but the same function is
  later restated with upper limit alpha (Gl. 4.4/36); integrating to a_O
  is ill-defined in general (the integrand's square root goes negative
  once a_O/alpha > 1), and only the alpha-limit choice reproduces the two
  published reference values in Overhagen's Abb. 4.4/4 (omega_O -> 1 as
  a_O -> 0; omega_O -> pi/4 as a_O -> 1, alpha -> 0 - both checked in
  ``tests/test_orowan_solve.py``). This implementation integrates to alpha.
- **The flow-stress convention.** Gl. (4.4/5)/(4.4/6) use a bare
  flow_stress/sqrt(3) offset (an older, Tresca-flavoured convention),
  rather than this plugin's Mises convention
  (sigma_x - sigma_N = -/+ 2*flow_stress/sqrt(3)). This implementation
  doubles the flow stress throughout Gl. (4.4/6), so it reduces to the
  ordinary Mises relation in the frictionless limit and stays continuous
  with the elastic-zone yield-onset criterion.
- **The entry/exit sign of the (1/alpha - 1/tan(alpha)) geometric term.**
  Overhagen's source prints it as "∓" without enough context to fix which
  sign belongs to which zone; this implementation uses abs(alpha)
  throughout and lets the explicit zone sign carry the distinction. This
  term is small relative to the omega_O term for typical rolling angles.

Treat this solver as a documented reconstruction, not a byte-for-byte
transcription of the 1943 paper - validate against reference data before
relying on it beyond qualitative comparisons.
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


def omega_orowan_sticking(angle: float) -> float:
    """Orowan's inhomogeneity function omega_O(alpha, a_O=1) (full sticking
    friction, Gl. 4.4/7 with the corrected upper limit - see module
    docstring), evaluated via the closed form

        omega_O(alpha, 1) = pi * J_1(alpha) / (2 * sin(alpha))

    (J_1 the order-1 Bessel function of the first kind), which follows from
    the Fourier-Bessel cosine integral
    integral_0^1 sqrt(1-t^2) cos(x*t) dt = pi*J_1(x)/(2*x) substituted into
    Gl. (4.4/7) with a_O=1. Avoids a numerical integration on every ODE
    right-hand-side evaluation. Always evaluated at abs(angle) - see module
    docstring."""
    angle = abs(angle)
    if angle < 1e-6:
        return np.pi / 4
    return np.pi * jv(1, angle) / (2 * np.sin(angle))


def _geometric_correction(angle: float) -> float:
    """The (1/alpha - 1/tan(alpha)) term in Gl. (4.4/6), evaluated at
    abs(angle) via its Taylor series for small angles to avoid catastrophic
    cancellation between two nearly-equal, individually large terms."""
    angle = abs(angle)
    if angle < 1e-3:
        return angle / 3 + angle ** 3 / 45
    return 1 / angle - 1 / np.tan(angle)


class OrowanSolver(ElasticPlasticSolverBase):
    """Orowan's (1943) slab theory for hot rolling, with elastic entry/exit
    zones and Bay & Wanheim mixed Coulomb/sticking friction throughout -
    see module docstring for how the two combine.

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

    def _closure_sigma_y(self, sigma_x, angle, flow_stress, zone_sign):
        """Solves Orowan's Gl. (4.4/6) for sigma_y, doubling flow_stress
        per the module docstring's flow-stress-convention note. Orowan's
        Gl. (4.4/6) is written for a compression-positive sigma_x/sigma_N
        convention, the opposite of this plugin's tension-positive one, so
        translating it here gives sigma_y = sigma_x - offset (not
        sigma_y = -sigma_x - offset).

        The sticking inhomogeneity value omega_O(alpha, 1) is linearly
        blended toward the frictionless limit 1.0 using Bay & Wanheim's
        transition weight Phi, which itself depends on sigma_y - the
        quantity being solved for - so this iterates to a fixed point.
        omega_O varies only weakly with its argument (1.0 to ~0.785), so
        this converges in a handful of iterations regardless of starting
        point."""
        sticking_omega = omega_orowan_sticking(angle)
        correction = _geometric_correction(angle)
        sign = 1.0 if zone_sign > 0 else -1.0

        def sigma_y_for(omega):
            offset = (2 * flow_stress / np.sqrt(3)) * (omega - sign * 0.5 * correction)
            return sigma_x - offset

        sigma_y = sigma_y_for(sticking_omega)
        for _ in range(10):
            _, _, transition_weight = self._mixed_friction(sigma_y, angle, flow_stress, zone_sign)
            effective_omega = (1 - transition_weight) * 1.0 + transition_weight * sticking_omega
            new_sigma_y = sigma_y_for(effective_omega)
            if abs(new_sigma_y - sigma_y) < 1e-6 * max(abs(flow_stress), 1.0):
                sigma_y = new_sigma_y
                break
            sigma_y = new_sigma_y
        return sigma_y
