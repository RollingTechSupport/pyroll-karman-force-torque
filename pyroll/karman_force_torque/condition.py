"""Criteria for switching between the plugin's solvers:
:class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
(elastic-plastic, single homogeneous slab, "medium"),
:class:`.layer_solver.LayerRollingSolver` (elastic-plastic zones, multi-layer,
thick slabs), and :class:`.foil_solver.FoilRollingSolver` (full elastic roll
contour, foil rolling).

``hitchcock_radius_ratio`` (Mauk & Overhagen 2013, p. 12): Hitchcock's
flattened-radius equation

    r'/r = 1 + C * F' / (h0 - h1),  with  C = 16 * (1 - nu_R**2) / (pi * E_R)

(``F'`` = roll force per unit width; the per-unit-width form of eq. 1/2 needs no
strip width) stays "usable" only while ``r'/r < 2``; beyond that a
single-layer or Hitchcock-flattened-roll model is no longer trustworthy and
the full elastic foil-rolling model is required.

``contact_length_over_mean_thickness`` is the classical Ld/Hm ratio deciding
whether through-thickness deformation is homogeneous enough for a
single-layer treatment or needs the multi-layer model.
"""

import math

from pyroll.core import RollPass


def hitchcock_radius_ratio(roll_pass: RollPass, force_per_unit_width: float) -> float:
    roll = roll_pass.roll
    e_r, nu_r = roll.elastic_modulus, roll.poissons_ratio
    c = 16 * (1 - nu_r ** 2) / (math.pi * e_r)

    draft = roll_pass.in_profile.equivalent_height - roll_pass.out_profile.equivalent_height
    return 1 + c * force_per_unit_width / draft


def contact_length_over_mean_thickness(roll_pass: RollPass, working_radius: float) -> float:
    """Classical contact-length-to-mean-thickness ratio (Ld/Hm): the
    criterion for whether through-thickness deformation can be treated as
    homogeneous (single "layer", Ld/Hm large) or whether redundant/
    inhomogeneous deformation through the thickness becomes significant
    enough to need a multi-layer treatment (Ld/Hm small, "thick slab").
    """
    h0 = roll_pass.in_profile.equivalent_height
    h1 = roll_pass.out_profile.equivalent_height
    draft = h0 - h1
    contact_length = math.sqrt(max(working_radius * draft - draft ** 2 / 4, 0.0))
    mean_thickness = (h0 + h1) / 2
    return contact_length / mean_thickness
