import pytest
from pyroll.core import Profile, PassSequence, RollPass, Roll, FlatGroove

from pyroll.flat_rolling_slab_model.karman_mixed_friction_solver import KarmanMixedFrictionSolver


def _flow_stress(kf):
    def hookimpl(self):
        def f(strain, strain_rate, temperature):
            return kf

        return f

    return hookimpl


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
def test_karman_mixed_friction_solver_converges_with_sane_outputs():
    """A representative hot-rolling pass: converges with elastic entry/exit
    zones now bounding the plastic zone (rather than assuming the whole
    cross-section is plastic from first contact), gives a positive force
    and torque, and a normal-pressure profile that forms the expected
    friction hill (positive throughout, single interior peak) rather than
    dipping negative."""
    roll_pass = _build_hot_pass(mu=0.35)

    solver = KarmanMixedFrictionSolver(roll_pass=roll_pass)

    assert solver.roll_force_per_unit_width > 0
    assert solver.roll_torque_per_unit_width > 0
    assert solver.entry_position < solver.neutral_plane_position < solver.exit_position

    pressure = solver.solution["normal_pressure"]
    assert (pressure >= -1.0).all()
    assert pressure.max() > pressure.iloc[0]
    assert pressure.max() > pressure.iloc[-1]

    assert {"vertical_stress", "normal_pressure", "shear_stress", "equivalent_strain"} <= set(solver.solution.columns)


@pytest.mark.slow
def test_karman_mixed_friction_solver_lower_friction_reduces_force():
    """A lower Coulomb friction coefficient should predict a lower peak
    pressure and roll force than a higher one, same qualitative check as
    OrowanSolver's own friction-sensitivity test - both solvers share the
    same Bay & Wanheim mixed friction law via _elastic_plastic_common.py,
    differing only in their sigma_x/sigma_y closure."""
    low_friction = KarmanMixedFrictionSolver(roll_pass=_build_hot_pass(mu=0.15))
    high_friction = KarmanMixedFrictionSolver(roll_pass=_build_hot_pass(mu=0.35))

    assert low_friction.roll_force_per_unit_width < high_friction.roll_force_per_unit_width
    assert low_friction.solution["normal_pressure"].max() < high_friction.solution["normal_pressure"].max()
