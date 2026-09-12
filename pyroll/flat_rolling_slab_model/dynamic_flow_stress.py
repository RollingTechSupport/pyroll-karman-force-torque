"""Troost's (1967) plastokinetic correction to flow stress (see
``docs/docs.tex``, section "Von-Kármán's equilibrium equation", and
``Troost1967`` in ``docs/refs.bib``): folding the strip's inertia
(d'Alembert's principle) into a "dynamic flow stress" k_f* = k_f + this
correction lets every existing static solution carry over unmodified.

Derived from Troost's eqs. (2) and (12):

    T_x(x) = -2 * q_st0 * h0 * (h0/h(x) - 1)      (inertial force, eq. 2)
    k_f*(x) - k_f(x) = -T_x(x)/h(x)                (eq. 12)
                      = 2 * q_st0 * [(h0/h(x))**2 - h0/h(x)]

with q_st0 = 0.5 * density * reference_velocity**2 the "Staudruck"
(stagnation pressure) of the strip at the chosen reference velocity.

The height ratio h0/h(x) is recovered from the equivalent (Mises) strain
already computed at every flow-stress call site in this plugin, since
strain = 2/sqrt(3) * ln(h0/h(x)) - so this needs no extra geometry
plumbing beyond what each solver already passes into its own flow-stress
evaluation.
"""

import numpy as np


def troost_dynamic_flow_stress_correction(strain: float, density: float, reference_velocity: float) -> float:
    height_ratio = np.exp(strain * np.sqrt(3) / 2)
    stagnation_pressure = 0.5 * density * reference_velocity ** 2
    return 2 * stagnation_pressure * (height_ratio ** 2 - height_ratio)
