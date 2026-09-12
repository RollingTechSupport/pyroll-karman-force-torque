import numpy as np
import pytest
from pyroll.core import Profile, PassSequence, RollPass, Roll, FlatGroove

from pyroll.flat_rolling_slab_model.orowan_solver import OrowanSolver, omega_orowan_sticking, omega_orowan_general


def test_omega_orowan_general_reproduces_overhagens_abb_4_4_4():
    """Overhagen's Abb. 4.4/4 plots omega_O(alpha, a_O) across the whole
    a_O range at two angles (0 and 30 degrees), not just the two corner
    values test_omega_orowan_matches_published_reference_values checks -
    a stronger, whole-curve reproduction of the reconstructed Gl. (4.4/7)
    with its corrected (alpha, not a_O) upper integration limit."""
    for angle in (0.0, np.deg2rad(30)):
        # a_O=0 (frictionless limit) is exactly 1 at any angle.
        assert omega_orowan_general(angle, 0.0) == pytest.approx(1.0)
        # a_O=1 (full sticking) must match the closed-form sticking
        # function this solver actually uses, to machine precision - the
        # general integral and the closed form are two independent
        # derivations of the same a_O=1 slice.
        assert omega_orowan_general(angle, 1.0) == pytest.approx(omega_orowan_sticking(angle), rel=1e-9)
        # monotonically decreasing in a_O, matching the dissertation's own
        # plotted curve shape (omega_O falls from 1 towards ~0.79-0.90 as
        # friction ratio rises from 0 to 1).
        values = [omega_orowan_general(angle, a) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert all(earlier > later for earlier, later in zip(values, values[1:]))

    # at any fixed a_O, omega_O increases weakly with angle - the dissertation's
    # own "depends only weakly on alpha" statement, checked here across the
    # full a_O range rather than only at a_O=1 (already covered above).
    for friction_ratio in (0.25, 0.5, 0.75, 1.0):
        assert omega_orowan_general(np.deg2rad(30), friction_ratio) > omega_orowan_general(0.0, friction_ratio)

    # alpha=0 has its own closed form (l'Hopital on cos(alpha*t) -> 1):
    # omega_O(0, a_O) = 0.5*(sqrt(1-a_O^2) + arcsin(a_O)/a_O) - cross-check
    # the alpha<1e-8 branch against a direct numeric integration at a tiny
    # but nonzero angle, since that branch exists only to avoid a 0/0 in
    # alpha/sin(alpha), not to change the underlying formula.
    tiny_angle = 1e-6
    for friction_ratio in (0.25, 0.5, 0.75, 1.0):
        assert omega_orowan_general(0.0, friction_ratio) == pytest.approx(
            omega_orowan_general(tiny_angle, friction_ratio), rel=1e-4,
        )


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
    and torque, and a normal-pressure profile that forms the expected
    friction hill (positive throughout, single interior peak) rather than
    dipping negative."""
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
