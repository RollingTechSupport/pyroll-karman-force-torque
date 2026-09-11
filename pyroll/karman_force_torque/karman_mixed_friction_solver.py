"""Von-Kármán slab theory with a mixed Coulomb/sticking friction law and
elastic entry/exit zones.

Generalizes von-Kármán's slab method with a mixed Coulomb/sticking friction
law, following Bay & Wanheim's smoothed formulation of it (Bay & Wanheim,
"Real area of contact and friction stresses at high pressure sliding
contact", Wear 38, 1976 - the same law :mod:`.layer_solver` and
:class:`.orowan_solver.OrowanSolver` use), instead of pure Coulomb friction
throughout. Plain Coulomb friction, integrated all the way to the roll gap
center, produces an unbounded pressure spike once the friction hill would
demand more shear traction than the material can actually transmit; the
mixed law caps it at the material's sticking (shear-yield) limit instead,
which is what actually happens once passes get thick enough (or friction
high enough) for pure Coulomb to become unrealistic.

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
Coulomb/sticking friction) is Bay & Wanheim's, not Orowan's. It shares its
elastic-plastic zone-chain machinery with :class:`.orowan_solver.OrowanSolver`
(see :mod:`._elastic_plastic_common`), differing only in the algebraic
closure relating sigma_x to sigma_y: the plain Mises relation here, versus
Orowan's inhomogeneity-corrected one there.

This solver is used for the "medium" dispatch tier (see ``roll_pass.py``),
in place of ``LayerRollingSolver(layer_count=1)``: it is kept as its own,
thermal-property-free class rather than dispatching to the (thermally
coupled) layer model, for passes where the layer model's thermal coupling
isn't needed or its per-layer sequencing overhead isn't warranted.
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
    docstring (and note there on why this isn't Orowan's own theory).

    Provides the same public attributes as the other solvers in this
    plugin (``roll_force_per_unit_width``, ``roll_torque_per_unit_width``,
    ``entry_velocity``, ``exit_velocity``, ``neutral_plane_position``,
    ``solution``) so that consumers (report plotting, the roll-pass hooks)
    do not need to know which solver actually ran.
    """

    def __init__(self, roll_pass: RollPass, pressure_smoothness: float = 0.1):
        self.section_cls = KarmanMixedFrictionSection
        super().__init__(roll_pass=roll_pass, pressure_smoothness=pressure_smoothness)


class KarmanMixedFrictionSection(ElasticPlasticSection):
    """Plain Mises closure (sigma_x - sigma_y = 2*kf/sqrt(3), zone-
    independent), unlike Orowan's inhomogeneity-corrected one."""

    def _closure_sigma_y(self, sigma_x, alpha, kf_val, zone):
        return sigma_x - 2 * kf_val / np.sqrt(3)
