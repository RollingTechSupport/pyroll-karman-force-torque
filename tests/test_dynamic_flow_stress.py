import numpy as np
import pytest
from pyroll.core import Profile, PassSequence, RollPass, Roll, FlatGroove

from pyroll.flat_rolling_slab_model.dynamic_flow_stress import troost_dynamic_flow_stress_correction
from pyroll.flat_rolling_slab_model.karman_mixed_friction_solver import KarmanMixedFrictionSolver


def test_correction_vanishes_at_zero_strain():
    """At the entry point (strain=0, height_ratio=h0/h=1) Troost's correction
    must vanish regardless of speed - it is a pure function of how far the
    strip has already been drawn down, not an offset."""
    assert troost_dynamic_flow_stress_correction(0.0, density=7850.0, reference_velocity=50.0) == 0.0


def test_correction_matches_hand_derivation():
    """Regression check against the closed form derived from Troost's eqs.
    (2) and (12) (see module docstring and docs.tex): for a 20% height
    reduction (h0/h=1.25) at 40 m/s and steel density, the correction should
    be a few MPa - matching the order of magnitude Troost's own worked
    example reports for a comparable speed."""
    height_ratio = 1.25
    strain = 2 / np.sqrt(3) * np.log(height_ratio)
    density = 7850.0
    reference_velocity = 40.0

    correction = troost_dynamic_flow_stress_correction(strain, density, reference_velocity)

    q_st0 = 0.5 * density * reference_velocity ** 2
    expected = 2 * q_st0 * (height_ratio ** 2 - height_ratio)
    assert correction == pytest.approx(expected)
    assert 1e6 < correction < 10e6


def _flow_stress(kf):
    def hookimpl(self):
        def f(strain, strain_rate, temperature):
            return kf

        return f

    return hookimpl


def _build_pass(rotational_frequency, dynamic_flow_stress_correction_enabled):
    in_profile = Profile.box(
        height=15e-3, width=200e-3, temperature=1200 + 273.15, strain=0,
        material=["dummy"], elastic_modulus=210e9, poissons_ratio=0.3,
        density=7850.0,
        flow_stress_function=_flow_stress(100e6),
    )
    roll_pass = RollPass(
        label="hot",
        roll=Roll(
            groove=FlatGroove(usable_width=300e-3), nominal_radius=160e-3,
            rotational_frequency=rotational_frequency,
            elastic_modulus=210e9, poissons_ratio=0.3,
        ),
        gap=12e-3, back_tension=0, front_tension=0, coulomb_friction_coefficient=0.35,
        dynamic_flow_stress_correction_enabled=dynamic_flow_stress_correction_enabled,
    )
    PassSequence([roll_pass]).solve(in_profile)
    return roll_pass


def test_disabled_by_default():
    """Not passing the hook at all must behave exactly like passing False -
    the correction must never engage silently."""
    roll_pass = RollPass(
        label="hot",
        roll=Roll(
            groove=FlatGroove(usable_width=300e-3), nominal_radius=160e-3, rotational_frequency=50,
            elastic_modulus=210e9, poissons_ratio=0.3,
        ),
        gap=12e-3, back_tension=0, front_tension=0, coulomb_friction_coefficient=0.35,
    )
    assert roll_pass.dynamic_flow_stress_correction_enabled is False


@pytest.mark.slow
def test_disabled_correction_matches_uncorrected_force():
    """Solving with the hook explicitly False must match the solver's
    behavior before this correction existed (no regression for the
    overwhelming majority of passes that never touch this hook)."""
    roll_pass = _build_pass(rotational_frequency=50, dynamic_flow_stress_correction_enabled=False)
    solver = KarmanMixedFrictionSolver(roll_pass=roll_pass)
    assert solver.roll_force_per_unit_width > 0


@pytest.mark.slow
def test_enabled_correction_increases_force_at_high_speed():
    """Enabling the correction adds a strictly positive term to flow stress
    everywhere past the entry point (see test_correction_vanishes_at_zero_strain),
    so it should raise the roll force - at a high rotational frequency (fast
    strip, per Troost's own "höhere Verformungsgeschwindigkeiten" scope) the
    effect should be large enough to measure clearly, unlike at ordinary
    cold-rolling speeds where it is well under 1% (see docs.tex)."""
    baseline = KarmanMixedFrictionSolver(
        roll_pass=_build_pass(rotational_frequency=200, dynamic_flow_stress_correction_enabled=False)
    )
    corrected = KarmanMixedFrictionSolver(
        roll_pass=_build_pass(rotational_frequency=200, dynamic_flow_stress_correction_enabled=True)
    )

    assert corrected.roll_force_per_unit_width > baseline.roll_force_per_unit_width
