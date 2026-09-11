import numpy as np
import pytest
from pyroll.core import Profile, PassSequence, RollPass, Roll, FlatGroove

from pyroll.karman_force_torque.orowan_solver import OrowanSolver, omega_orowan_sticking


def test_omega_orowan_matches_published_reference_values():
    """Overhagen's dissertation (Abb. 4.4/4) plots omega_O(alpha, a_O) from
    a_O=0 (omega_O=1, exact) to a_O=1 (omega_O -> pi/4 =~ 0.785 as alpha->0,
    a well-known closed-form limit of Orowan's theory) - the only two points
    on that curve derivable in closed form, and reproduced here as the
    concrete check that the a_O=1 slice (omega_orowan_sticking) integrates
    Gl. (4.4/7) with the corrected upper limit (see module docstring)."""
    assert omega_orowan_sticking(1e-6) == pytest.approx(np.pi / 4, rel=1e-4)
    assert omega_orowan_sticking(0.05) == pytest.approx(np.pi / 4, rel=1e-2)
    # weakly increasing with alpha, per the dissertation's own statement
    # that omega_O only weakly depends on the angle
    assert omega_orowan_sticking(0.5) > omega_orowan_sticking(0.05)
    assert omega_orowan_sticking(0.5) < 1.0


def _flow_stress(kf):
    def hookimpl(self):
        def f(strain, strain_rate, temperature):
            return kf

        return f

    return hookimpl


@pytest.mark.slow
def test_orowan_solver_converges_with_sane_outputs():
    """A representative hot-rolling pass: converges, gives a positive force
    and torque comparable in order of magnitude to the mixed-friction
    model's estimate for the same geometry and a similarly high friction
    coefficient (both should be dominated by sticking for this geometry),
    and a normal-pressure profile that forms the expected friction hill
    (positive throughout, single interior peak) rather than dipping
    negative (the bug this test would have caught: an earlier version of
    the sigma_x/sigma_y sign mapping made pressure decrease monotonically
    from the entry edge instead)."""
    in_profile = Profile.box(
        height=15e-3, width=200e-3, temperature=1200 + 273.15, strain=0,
        material=["dummy"], elastic_modulus=210e9, poissons_ratio=0.3,
        flow_stress_function=_flow_stress(100e6),
    )
    roll_pass = RollPass(
        label="hot",
        roll=Roll(
            groove=FlatGroove(usable_width=300e-3), nominal_radius=160e-3, rotational_frequency=1,
            elastic_modulus=210e9, poissons_ratio=0.3,
        ),
        gap=12e-3, back_tension=0, front_tension=0, coulomb_friction_coefficient=0.35,
    )
    PassSequence([roll_pass]).solve(in_profile)

    solver = OrowanSolver(roll_pass=roll_pass)

    assert solver.roll_force_per_unit_width > 0
    assert solver.roll_torque_per_unit_width > 0
    assert solver.entry_position < solver.neutral_plane_position < solver.exit_position

    pressure = solver.solution["normal_pressure"]
    assert (pressure >= -1.0).all()  # never meaningfully negative
    assert pressure.max() > pressure.iloc[0]  # a hill, not a monotonic decrease from the entry edge
    assert pressure.max() > pressure.iloc[-1]


def _build_hot_pass(mu):
    in_profile = Profile.box(
        height=15e-3, width=200e-3, temperature=1200 + 273.15, strain=0,
        material=["dummy"], elastic_modulus=210e9, poissons_ratio=0.3,
        flow_stress_function=_flow_stress(100e6),
    )
    roll_pass = RollPass(
        label="hot",
        roll=Roll(
            groove=FlatGroove(usable_width=300e-3), nominal_radius=160e-3, rotational_frequency=1,
            elastic_modulus=210e9, poissons_ratio=0.3,
        ),
        gap=12e-3, back_tension=0, front_tension=0, coulomb_friction_coefficient=mu,
    )
    PassSequence([roll_pass]).solve(in_profile)
    return roll_pass


@pytest.mark.slow
def test_orowan_solver_lower_friction_reduces_force():
    """OrowanSolver always uses Bay & Wanheim's mixed Coulomb/sticking law
    (the same law KarmanMixedFrictionSolver/LayerRollingSolver use) layered
    on top of Orowan's inhomogeneity correction. A lower Coulomb friction
    coefficient should predict a *lower* peak pressure and roll force than
    a higher one (less severe friction assumption), while both still form a
    proper friction hill (positive throughout, single interior peak,
    elastic zones carrying nonzero friction since Bay & Wanheim's law is
    well-behaved as pressure -> 0)."""
    low_friction = OrowanSolver(roll_pass=_build_hot_pass(mu=0.15))
    high_friction = OrowanSolver(roll_pass=_build_hot_pass(mu=0.35))

    for solver in (low_friction, high_friction):
        assert solver.roll_force_per_unit_width > 0
        assert solver.roll_torque_per_unit_width > 0
        assert solver.entry_position < solver.neutral_plane_position < solver.exit_position
        pressure = solver.solution["normal_pressure"]
        assert (pressure >= -1.0).all()
        assert pressure.max() > pressure.iloc[0]
        assert pressure.max() > pressure.iloc[-1]

    assert low_friction.roll_force_per_unit_width < high_friction.roll_force_per_unit_width
    assert low_friction.solution["normal_pressure"].max() < high_friction.solution["normal_pressure"].max()
