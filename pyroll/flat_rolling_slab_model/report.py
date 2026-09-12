import scipy.interpolate as inter
from pyroll.report import hookimpl
from pyroll.core import Unit, PassSequence, Transport, CoolingPipe, RollPass

import matplotlib.pyplot as plt

from pyroll.flat_rolling_slab_model.foil_solver import FoilRollingSolver
from pyroll.flat_rolling_slab_model.layer_solver import LayerRollingSolver


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

        fig: plt.Figure = plt.figure(figsize=(6, 7))
        ax: plt.Axes
        axd: plt.Axes
        axl: plt.Axes
        ax, axd, axl = fig.subplots(nrows=3, height_ratios=[1, 0.6, 0.3])
        ax.set_title("Foil Rolling - Elastically Flattened Roll-Gap Contour")
        ax.grid(lw=0.5)

        flattened = ax.plot(contour.index, contour["gap_height"], label="Flattened gap height")
        rigid = ax.plot(contour.index, contour["rigid_gap_height"], label="Rigid (circular) gap height",
                         linestyle="--")
        ax.set_ylabel("Gap height")

        # The two curves above can look nearly identical at typical foil
        # r'/r ratios even though the flattening is real and load-bearing -
        # plot their difference on its own axis (a different, much smaller
        # scale) so a small but genuine effect stays visible rather than
        # disappearing into line width.
        deviation = (contour["gap_height"] - contour["rigid_gap_height"])
        axd.grid(lw=0.5)
        axd.axhline(0.0, color="black", lw=0.75)
        axd.plot(contour.index, deviation, color="tab:red")
        axd.set_xlabel("x")
        axd.set_ylabel("Flattened - rigid")
        axd.fill_between(contour.index, deviation, 0.0, color="tab:red", alpha=0.15)

        axl.axis("off")
        axl.legend(handles=flattened + rigid, ncols=2, loc="lower center")

        return fig


def layer_model_per_layer_plots(unit: Unit):
    """One figure per layer for a thick-slab :class:`.layer_solver.LayerRollingSolver`
    pass (``layer_count > 1``): each layer's own horizontal stress ``sigma_x``
    against the shared vertical stress ``sigma_y``, plus that layer's own
    temperature and equivalent strain - lets sequential (Schmiedekreuz)
    yielding be inspected layer-by-layer, which the thickness-aggregated
    plot from :func:`disked_unit_temperature_plot` cannot show. Not wired in
    as a ``unit_plot`` hookimpl (that contract returns one figure per
    registered hookimpl, and the number of layers varies per pass) - call
    directly and handle the returned list of figures.
    """
    if not (isinstance(unit, RollPass) and isinstance(unit.karman_solution, LayerRollingSolver)):
        return []

    solver = unit.karman_solution
    per_layer = solver.section.solution_dataframe_per_layer()
    figures = []
    for i, df in per_layer.items():
        fig: plt.Figure = plt.figure(figsize=(6, 6))
        ax: plt.Axes
        axl: plt.Axes
        ax, axl = fig.subplots(nrows=2, height_ratios=[1, 0.3])
        ax.set_title(f"Layer {i} - Stress, Temperature and Strain")
        ax.grid(lw=0.5)

        sigma_x = ax.plot(df.index, df["sigma_x"], label=r"Horizontal Stress $\sigma_{x,i}$")
        sigma_y = ax.plot(df.index, df["sigma_y"], label=r"Vertical Stress $\sigma_{y}$ (shared)",
                           linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("Stress")
        handles = sigma_x + sigma_y

        axr = ax.twinx()
        temperature = axr.plot(df.index, df["temperature"], label="Temperature $T$", color="red",
                                linestyle=":")
        axr.set_ylabel("Temperature")
        handles = handles + temperature

        axr2 = ax.twinx()
        axr2.spines["right"].set_position(("axes", 1.2))
        strain = axr2.plot(df.index, df["equivalent_strain"], label=r"Equivalent Strain $\varphi_{v,i}$",
                            color="black", linestyle="-.")
        axr2.set_ylabel(r"Equivalent Strain $\varphi_{v,i}$")
        handles = handles + strain

        axl.axis("off")
        axl.legend(handles=handles, ncols=2, loc="center")

        figures.append(fig)
    return figures
