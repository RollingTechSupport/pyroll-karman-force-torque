import scipy.interpolate as inter
from pyroll.report import hookimpl
from pyroll.core import Unit, PassSequence, Transport, CoolingPipe, RollPass

import matplotlib.pyplot as plt

from pyroll.karman_force_torque.foil_solver import FoilRollingSolver


@hookimpl(specname="unit_plot")
def disked_unit_temperature_plot(unit: Unit):
    if isinstance(unit, RollPass):
        solution = unit.karman_solution.solution

        fig: plt.Figure = plt.figure(figsize=(6, 6))
        ax: plt.Axes
        axl: plt.Axes
        ax, axl = fig.subplots(nrows=2, height_ratios=[1, 0.3])
        ax.set_title("Karman Stress and Strain Distribution")
        ax.grid(lw=0.5)

        vertical_stress_interpolation = inter.interp1d(solution.index, solution["vertical_stress"],
                                                         fill_value="extrapolate")

        normal_pressure_interpolation = inter.interp1d(solution.index, solution["normal_pressure"],
                                                         fill_value="extrapolate")

        shear_stress_interpolation = inter.interp1d(solution.index, solution["shear_stress"],
                                                      fill_value="extrapolate")

        vert_stress = ax.plot(solution.index, vertical_stress_interpolation(solution.index),
                               label=r"Vertical Stress $\sigma_{y}$")
        normal_press = ax.plot(solution.index, normal_pressure_interpolation(solution.index),
                                label=r"Normal Pressure $p_{N}$")
        shear_stress = ax.plot(solution.index, shear_stress_interpolation(solution.index),
                                label=r"Shear Stress $\tau$")
        ax.set_xlabel("x")
        ax.set_ylabel("Stress")

        handles = vert_stress + normal_press + shear_stress

        if "equivalent_strain" in solution.columns:
            strain_interpolation = inter.interp1d(solution.index, solution["equivalent_strain"],
                                                   fill_value="extrapolate")
            axr = ax.twinx()
            strain = axr.plot(solution.index, strain_interpolation(solution.index),
                               label=r"Equivalent Strain $\varphi_{v}$", color="black", linestyle="--")
            axr.set_ylabel(r"Equivalent Strain $\varphi_{v}$")
            handles = handles + strain

        axl.axis("off")
        axl.legend(handles=handles, ncols=len(handles), loc="lower center")

        return fig


@hookimpl(specname="unit_plot")
def foil_rolling_contour_plot(unit: Unit):
    if isinstance(unit, RollPass) and isinstance(unit.karman_solution, FoilRollingSolver):
        contour = unit.karman_solution.roll_contour

        fig: plt.Figure = plt.figure(figsize=(6, 6))
        ax: plt.Axes
        axl: plt.Axes
        ax, axl = fig.subplots(nrows=2, height_ratios=[1, 0.3])
        ax.set_title("Foil Rolling - Elastically Flattened Roll-Gap Contour")
        ax.grid(lw=0.5)

        flattened = ax.plot(contour.index, contour["gap_height"], label="Flattened gap height")
        rigid = ax.plot(contour.index, contour["rigid_gap_height"], label="Rigid (circular) gap height",
                         linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("Gap height")

        axl.axis("off")
        axl.legend(handles=flattened + rigid, ncols=2, loc="lower center")

        return fig
