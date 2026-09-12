"""Criterion for switching between the plugin's solvers:
:class:`.karman_mixed_friction_solver.KarmanMixedFrictionSolver`
(elastic-plastic, single homogeneous slab, "medium"),
:class:`.layer_solver.LayerRollingSolver` (elastic-plastic zones, multi-layer,
thick slabs), and :class:`.foil_solver.FoilRollingSolver` (full elastic roll
contour, foil rolling).

``contact_length_over_mean_thickness`` is the classical Ld/Hm ratio deciding
between all three: small (below ``layer_model_ld_hm_limit``) means
through-thickness deformation is inhomogeneous enough to need the
multi-layer model; large (above ``foil_rolling_ld_hm_limit``) means the pass
is thin enough relative to the contact length that elastic roll flattening
can no longer be treated as negligible, requiring the full elastic
foil-rolling model; in between, the single-layer mixed-friction model
applies.
"""

import math

from pyroll.core import RollPass


def contact_length_over_mean_thickness(roll_pass: RollPass, working_radius: float) -> float:
    """Classical contact-length-to-mean-thickness ratio (Ld/Hm): the sole
    criterion this plugin uses to classify a pass as thick (Ld/Hm small,
    needing a multi-layer treatment), medium (single homogeneous layer
    suffices), or foil-thin (Ld/Hm large, needing the full elastic
    roll-gap-contour model) - see ``roll_pass.py``'s ``thick_slab_condition``
    and ``foil_rolling_condition``.
    """
    entry_height = roll_pass.in_profile.equivalent_height
    exit_height = roll_pass.out_profile.equivalent_height
    draft = entry_height - exit_height
    contact_length = math.sqrt(max(working_radius * draft - draft ** 2 / 4, 0.0))
    mean_thickness = (entry_height + exit_height) / 2
    return contact_length / mean_thickness
