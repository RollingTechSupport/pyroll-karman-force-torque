import logging
import webbrowser
from pathlib import Path

from pyroll.core import Profile, PassSequence, RollPass, Roll, CircularOvalGroove, FlatGroove


def test_solve_oval(tmp_path: Path, caplog):
    """A single oval-groove pass (rather than a chained oval->round schedule
    with a Transport in between): exercises this plugin's dispatch and
    report generation against a non-flat groove and pyroll-core's
    auto-rotation logic, without exercising a chained multi-pass sequence's
    own tension/velocity coupling between stands, which is pyroll-core's
    concern rather than this plugin's."""
    caplog.set_level(logging.INFO, logger="pyroll")

    import pyroll.freiberg_flow_stress
    import pyroll.karman_force_torque

    in_profile = Profile.round(
        diameter=30e-3,
        temperature=1200 + 273.15,
        strain=0,
        material=["C45", "steel"],
        elastic_modulus=210e9,
        poissons_ratio=0.3,
    )

    sequence = PassSequence([
        RollPass(
            label="Oval I",
            roll=Roll(
                groove=CircularOvalGroove(
                    depth=8e-3,
                    r1=6e-3,
                    r2=40e-3
                ),
                nominal_radius=160e-3,
                rotational_frequency=1,
                elastic_modulus=210e9,
                poissons_ratio=0.3,
            ),
            gap=2e-3,
            coulomb_friction_coefficient=0.35,
            back_tension=0,
            front_tension=6e6,

        ),
    ])

    try:
        sequence.solve(in_profile)
    finally:
        print("\nLog:")
        print(caplog.text)

    try:
        import pyroll.report

        report = pyroll.report.report(sequence)

        report_file = tmp_path / "report.html"
        report_file.write_text(report)
        print(report_file)
        webbrowser.open(report_file.as_uri())

    except ImportError:
        pass



def test_solve_flat_flat(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO, logger="pyroll")

    import pyroll.freiberg_flow_stress
    import pyroll.karman_force_torque

    in_profile = Profile.box(
        height=15e-3,
        width=200e-3,
        temperature=1200 + 273.15,
        strain=0,
        material=["C45", "steel"],
        elastic_modulus=210e9,
        poissons_ratio=0.3,
    )

    sequence = PassSequence([
        RollPass(
            label="Oval I",
            roll=Roll(
                groove=FlatGroove(
                    usable_width=300e-3
                ),
                nominal_radius=160e-3,
                rotational_frequency=1,
                elastic_modulus=210e9,
                poissons_ratio=0.3,
            ),
            gap=12e-3,
            coulomb_friction_coefficient=0.35,
            back_tension=0,
            front_tension=0,

        )
    ])

    try:
        sequence.solve(in_profile)
    finally:
        print("\nLog:")
        print(caplog.text)

    try:
        import pyroll.report

        report = pyroll.report.report(sequence)

        report_file = tmp_path / "report.html"
        report_file.write_text(report)
        print(report_file)
        webbrowser.open(report_file.as_uri())

    except ImportError:
        pass