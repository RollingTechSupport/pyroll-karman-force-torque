"""Von-Kármán slab theory with a mixed Coulomb/sticking friction law and
elastic entry/exit zones.

Generalizes von-Kármán's slab method with Bay & Wanheim's smoothed mixed
Coulomb/sticking friction law (Bay & Wanheim, "Real area of contact and
friction stresses at high pressure sliding contact", Wear 38, 1976 - the
same law :mod:`.layer_solver` and :class:`.orowan_solver.OrowanSolver` use),
instead of pure Coulomb friction throughout. Plain Coulomb friction,
integrated all the way to the roll gap center, demands ever-increasing
shear traction the material cannot actually transmit once a pass is thick
or high-friction enough; the mixed law caps it at the material's sticking
limit instead.

Note this is *not* Orowan's own 1943 slab theory (Orowan, "The Calculation
of Roll Pressure in Hot and Cold Flat Rolling", Proc. IMechE 150, 1943): that
replaces the straight strip-element boundaries with circular arcs and an
inhomogeneity function correcting for genuinely non-uniform through-
thickness deformation, rather than a friction-law refinement on top of the
same strip-element geometry. This solver shares its elastic-plastic
zone-chain machinery with :class:`.orowan_solver.OrowanSolver` (see
:mod:`._elastic_plastic_common`), differing only in the closure relating
sigma_x to sigma_y: the plain Mises relation here, Orowan's
inhomogeneity-corrected one there.

Used for the "medium" dispatch tier (see ``roll_pass.py``), kept as its own
class rather than dispatching to ``LayerRollingSolver`` with a single layer,
since it needs neither thermal coupling nor per-layer resolution.
"""

import logging

import numpy as np

from pyroll.core import RollPass

from pyroll.karman_force_torque._elastic_plastic_common import (
    bay_wanheim_coulomb_from_stiction,
    ElasticPlasticSolverBase,
    ElasticPlasticSection,
)

log = logging.getLogger(__name__)


class KarmanMixedFrictionSolver(ElasticPlasticSolverBase):
    """Elastic-plastic von-Kármán slab-theory solution of a flat roll pass
    with Bay & Wanheim's mixed Coulomb/sticking friction law - see module
    docstring.

    Provides the same public attributes as the other solvers in this
    plugin (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``).
    """

    def __init__(self, roll_pass: RollPass, pressure_smoothness: float = 0.1):
        self.section_cls = KarmanMixedFrictionSection
        super().__init__(roll_pass=roll_pass, pressure_smoothness=pressure_smoothness)


class KarmanMixedFrictionSection(ElasticPlasticSection):
    """Plain Mises closure (sigma_x - sigma_y = 2*flow_stress/sqrt(3)),
    unlike Orowan's inhomogeneity-corrected one."""

    def _closure_sigma_y(self, sigma_x, angle, flow_stress, zone_sign):
        return sigma_x - 2 * flow_stress / np.sqrt(3)
