"""Criterion for switching from the rigid-roll :class:`.karman_solver.KarmanSolver`
to the elastic-plastic :class:`.foil_solver.FoilRollingSolver`.

Mauk & Overhagen (2013), p. 12: Hitchcock's flattened-radius equation

    r'/r = 1 + C * F' / (h0 - h1),  with  C = 16 * (1 - nu_R**2) / (pi * E_R)

(``F'`` = roll force per unit width; the per-unit-width form of eq. 1/2 needs no
strip width) stays "usable" only while ``r'/r < 2``; beyond that a rigid- or
Hitchcock-flattened-roll model is no longer trustworthy and the full elastic
foil-rolling model is required.
"""

import math

from pyroll.core import RollPass


def hitchcock_radius_ratio(roll_pass: RollPass, force_per_unit_width: float) -> float:
    roll = roll_pass.roll
    e_r, nu_r = roll.elastic_modulus, roll.poissons_ratio
    c = 16 * (1 - nu_r ** 2) / (math.pi * e_r)

    draft = roll_pass.in_profile.equivalent_height - roll_pass.out_profile.equivalent_height
    return 1 + c * force_per_unit_width / draft
