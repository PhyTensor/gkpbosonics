import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""An analysis of photon loss with varying squeezing parameter and finite-energy parameter""")
    return


@app.cell
def _():
    import os
    import sys

    module_path = os.path.abspath(os.path.join("./"))
    if module_path not in sys.path:
        sys.path.append(module_path)
    return


@app.cell
def _():
    from experiments.single_mode_loss_analysis import SingleModeLossAnalysis
    from experiments.double_mode_loss_analysis import DoubleModeLossAnalysis
    return (SingleModeLossAnalysis,)


@app.cell
def _():
    from typing import List

    import numpy as np
    from numpy import pi, sqrt, ndarray

    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colorbar, colors
    import matplotlib.pyplot as plt

    import strawberryfields as sf
    from strawberryfields import Engine, Program, Result
    from strawberryfields.backends import BaseBosonicState
    from strawberryfields.ops import (
        GKP,
        BSgate,
        Coherent,
        LossChannel,
        MeasureP,
        MeasureX,
        MeasureHomodyne,
        Squeezed,
        Xgate,
        Zgate,
        Rgate,
        Dgate,
        S2gate,
        CXgate,
    )

    # set the random seed
    np.random.seed(42)
    return List, ndarray, np, plt


@app.cell
def _():
    basename_sm: str = "single_mode_gkp_loss_"
    basename_dm: str = "double_mode_gkp_loss_"
    return basename_dm, basename_sm


@app.cell
def _(np):
    def linear2db(value):
        return abs(10 * np.log10(value))


    def db2linear(value):
        return 10 ** (value / 10)
    return (linear2db,)


@app.cell
def _(List):
    loss_transmissivities: List[float] = [1.0, 0.93, 0.85, 0.70, 0.60]
    return (loss_transmissivities,)


@app.cell
def _(SingleModeLossAnalysis):
    single_mode_loss_analysis: SingleModeLossAnalysis = SingleModeLossAnalysis()
    # double_mode_loss_analysis: DoubleModeLossAnalysis = DoubleModeLossAnalysis()
    return (single_mode_loss_analysis,)


@app.cell
def _(
    basename_sm: str,
    linear2db,
    loss_transmissivities: "List[float]",
    ndarray,
    plt,
    single_mode_loss_analysis: "SingleModeLossAnalysis",
):
    def single_mode_visualisation():
        figsm, axsm = plt.subplots(figsize=(10, 6))
        plotname: str = "analysis"
        # colorsk = plt.cm.viridis(np.linspace(0, 1, len(loss_transmissivities)))

        for idx, transmissivity in enumerate(loss_transmissivities):
            exps: ndarray = single_mode_loss_analysis.analyse(
                transmissivity=transmissivity
            )

            y_vals: ndarray = exps[:, 2]
            x_vals: ndarray = single_mode_loss_analysis.epsilons
            x_vals_dev: ndarray = linear2db(x_vals)

            axsm.plot(
                x_vals_dev,
                y_vals,
                linestyle="--",
                # color=colorsk[idx],
                label=rf"$\eta = {transmissivity}$",
            )

            # Annotate the line with its transmissivity value
            # label_x = x_vals[len(x_vals)//3]
            # label_y = y_vals[len(y_vals)//3]
            # ax.text(label_x, label_y, fr'$\eta$ = {transmissivity}', fontsize=10,
            # verticalalignment='bottom', horizontalalignment='right')

        # print(expectation_values)
        # secax = ax.secondary_xaxis('top', functions=(linear2db, db2linear))
        # secax.set_xlabel('dB')

        # for i in range(len(loss_transmissivities)):
        #     plt.plot(loss_analysis.epsilons, expectation_values[i][:, 2])

        axsm.set_xlabel(r"$\varepsilon$ [dB]")
        axsm.set_ylabel(r"$\langle Z \rangle$")

        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position('top')

        # secax = ax.secondary_xaxis('bottom', functions=(linear2db, db2linear))
        # secax.set_xlabel(r'$\varepsilon$')

        # ax.invert_xaxis() # plt.gca().invert_xaxis()  # Reverse x-axis
        plt.legend(loc="best", frameon=False)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname=basename_sm + plotname, dpi=300)
        plt.show()
    return (single_mode_visualisation,)


@app.cell
def _(
    basename_dm: str,
    double_mode_loss_analysis,
    linear2db,
    loss_transmissivities: "List[float]",
    ndarray,
    plt,
):
    def double_mode_visualisation():
        figdm, axdm = plt.subplots(figsize=(10, 6))
        plotname: str = "analysis"
        # colorsk = plt.cm.viridis(np.linspace(0, 1, len(loss_transmissivities)))

        for idx, transmissivity in enumerate(loss_transmissivities):
            exps: ndarray = double_mode_loss_analysis.analyse(
                transmissivity=transmissivity
            )

            y_vals: ndarray = exps[:, 2]
            x_vals: ndarray = double_mode_loss_analysis.epsilons
            x_vals_dev: ndarray = linear2db(x_vals)

            axdm.plot(
                x_vals_dev,
                y_vals,
                linestyle="-",
                # color=colorsk[idx],
                label=rf"$\eta = {transmissivity}$",
            )

            # Annotate the line with its transmissivity value
            # label_x = x_vals[len(x_vals)//3]
            # label_y = y_vals[len(y_vals)//3]
            # ax.text(label_x, label_y, fr'$\eta$ = {transmissivity}', fontsize=10,
            # verticalalignment='bottom', horizontalalignment='right')

        # print(expectation_values)
        # secax = ax.secondary_xaxis('top', functions=(linear2db, db2linear))
        # secax.set_xlabel('dB')

        # for i in range(len(loss_transmissivities)):
        #     plt.plot(loss_analysis.epsilons, expectation_values[i][:, 2])

        axdm.set_xlabel(r"$\varepsilon$ [dB]")
        axdm.set_ylabel(r"$\langle Z \rangle$")

        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position('top')

        # secax = ax.secondary_xaxis('bottom', functions=(linear2db, db2linear))
        # secax.set_xlabel(r'$\varepsilon$')

        # ax.invert_xaxis() # plt.gca().invert_xaxis()  # Reverse x-axis
        plt.legend(loc="best", frameon=False)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname=basename_dm + plotname, dpi=300)
        plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Logical Pauli Z expectation values for for the GKP circuit as a function of the GKP squeezing parameter $\epsilon$ for different values of the loss channel transmissivity $\eta$.

    The errors in the absence of losses are due to the finite squeezing of the finite GKP states.
    """
    )
    return


@app.cell
def _(single_mode_visualisation):
    single_mode_visualisation()
    return


@app.cell
def _():
    # double_mode_visualisation()
    return


if __name__ == "__main__":
    app.run()
