import logging

import pytest
from pyroll.core import Profile, PassSequence, RollPass, Roll, FlatGroove


def _flow_stress(kf):
    def hookimpl(self):
        def f(strain, strain_rate, temperature):
            return kf

        return f

    return hookimpl


def _build_pass(roll_elastic_modulus, hitchcock_limit=None):
    """Reproduces the first pass of the report's Table 5 experimental schedule
    (1.4016 stainless foil, ~23% reduction): h0=0.1mm -> h1=0.077mm, b=380mm,
    entry/exit tensions 214.24/309.91 N/mm^2, steel work rolls d~25.2mm."""
    kf = 900e6

    in_profile = Profile.box(
        height=0.1e-3,
        width=380e-3,
        temperature=293.15,
        strain=0,
        material=["dummy"],
        elastic_modulus=210e9,
        poissons_ratio=0.3,
        flow_stress_function=_flow_stress(kf),
    )

    kwargs = {}
    if hitchcock_limit is not None:
        kwargs["foil_rolling_hitchcock_limit"] = hitchcock_limit

    roll_pass = RollPass(
        label="foil pass",
        roll=Roll(
            groove=FlatGroove(usable_width=380e-3),
            nominal_radius=12.6e-3,
            rotational_frequency=1,
            elastic_modulus=roll_elastic_modulus,
            poissons_ratio=0.3,
        ),
        gap=0.077e-3,
        coulomb_friction_coefficient=0.1,
        back_tension=214.24e6,
        front_tension=309.91e6,
        **kwargs,
    )

    sequence = PassSequence([roll_pass])
    sequence.solve(in_profile)
    return roll_pass


def test_normal_pass_stays_on_rigid_solver(caplog):
    """The existing flat-pass example (large gauge, moderate force) is far from
    the foil-rolling regime, and Roll/Profile don't set elastic properties at
    all - the condition hook must fall back gracefully rather than error."""
    caplog.set_level(logging.INFO, logger="pyroll")

    import pyroll.freiberg_flow_stress
    import pyroll.karman_force_torque
    from pyroll.karman_force_torque.karman_solver import KarmanSolver

    in_profile = Profile.box(
        height=15e-3,
        width=200e-3,
        temperature=1200 + 273.15,
        strain=0,
        material=["C45", "steel"],
    )
    roll_pass = RollPass(
        label="Flat",
        roll=Roll(
            groove=FlatGroove(usable_width=300e-3),
            nominal_radius=160e-3,
            rotational_frequency=1,
        ),
        gap=12e-3,
        coulomb_friction_coefficient=0.35,
        back_tension=0,
        front_tension=0,
    )
    PassSequence([roll_pass]).solve(in_profile)

    assert roll_pass.foil_rolling_condition is False
    assert isinstance(roll_pass.karman_solution, KarmanSolver)


def test_foil_condition_triggers_for_thin_foil():
    """The report's own worked example (Abb. 9, kf=1000 N/mm^2, h0=0.03/h1=0.015mm,
    d=25mm) is deep in Hitchcock-invalid territory (r'/r >> 2)."""
    import pyroll.karman_force_torque

    kf = 1000e6
    in_profile = Profile.box(
        height=0.03e-3, width=300e-3, temperature=293.15, strain=0,
        material=["dummy"], elastic_modulus=210e9, poissons_ratio=0.3,
        flow_stress_function=_flow_stress(kf),
    )
    roll_pass = RollPass(
        label="thin foil",
        roll=Roll(
            groove=FlatGroove(usable_width=300e-3),
            nominal_radius=12.5e-3,
            rotational_frequency=1,
            elastic_modulus=210e9,
            poissons_ratio=0.3,
        ),
        gap=0.015e-3,
        coulomb_friction_coefficient=0.05,
        back_tension=0,
        front_tension=0,
    )
    PassSequence([roll_pass]).solve(in_profile)

    assert roll_pass.foil_rolling_condition is True


@pytest.mark.slow
def test_foil_solver_converges_and_reduces_force_with_stiffer_roll():
    """Steel vs. ceramic (StarCeram N8000, E=310 GPa per the report) work rolls
    on the same pass: the report's headline finding is that the stiffer ceramic
    roll flattens less and needs a lower roll force for the same reduction
    (Abb. 9: 521 kN ceramic vs. 690 kN steel, ~24% lower)."""
    steel = _build_pass(roll_elastic_modulus=210e9, hitchcock_limit=1.0)
    ceramic = _build_pass(roll_elastic_modulus=310e9, hitchcock_limit=1.0)

    assert steel.foil_rolling_condition is True
    assert ceramic.foil_rolling_condition is True

    assert steel.roll_force > 0
    assert ceramic.roll_force > 0
    assert steel.roll.roll_torque > 0
    assert ceramic.roll.roll_torque > 0

    # Ceramic (stiffer) rolls flatten less and should need a lower roll force.
    assert ceramic.roll_force < steel.roll_force

    solution = steel.karman_solution.solution
    assert {"vertical_stress", "normal_pressure", "shear_stress"} <= set(solution.columns)
    # The elastically-flattened contact zone is generally wider than pyroll-core's
    # rigid-geometry entry/exit points, so bound the neutral point against the
    # foil solver's own (self-consistent) contact-zone extent instead.
    foil_solution = steel.karman_solution
    assert foil_solution.entry_position <= steel.roll.neutral_point <= foil_solution.exit_position
