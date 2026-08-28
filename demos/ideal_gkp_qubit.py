import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import numpy as np
    from numpy import ndarray

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm, colors, colorbar
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams.update({
        # "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,

        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.top": True,
        "ytick.right": True,

        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    import strawberryfields as sf
    from strawberryfields import hbar, Program, ops, Engine, Result
    from strawberryfields.ops import LossChannel
    from strawberryfields.backends import BaseBosonicState
    return np, plt, sf


@app.cell
def _(np, sf):
    # Set the scale for phase space
    sf.hbar = 1
    scale = np.sqrt(sf.hbar * np.pi)
    return (scale,)


@app.cell
def _(sf):
    # Create a GKP |0> state
    def create_gkp_0():
        prog = sf.Program(1)

        with prog.context as q:
            sf.ops.GKP() | q

        return prog
    return (create_gkp_0,)


@app.cell
def _(np, sf):
    # Create a GKP |1> state
    def create_gkp_1():
        prog = sf.Program(1)

        with prog.context as q:
            sf.ops.GKP(shape="square") | q
            sf.ops.Xgate(np.sqrt(np.pi * sf.hbar)) | q

        return prog
    return (create_gkp_1,)


@app.cell
def _(sf):
    eng = sf.Engine("bosonic")
    shots: int = 1  # 2000  # Number of samples
    return eng, shots


@app.cell
def _(create_gkp_0, create_gkp_1, eng, shots: int):
    circuit_gkp_0 = create_gkp_0()
    circuit_gkp_1 = create_gkp_1()

    results_0 = eng.run(circuit_gkp_0, shots=shots)
    results_1 = eng.run(circuit_gkp_1, shots=shots)

    gkp_0 = results_0.state
    gkp_1 = results_1.state
    return gkp_0, gkp_1


@app.cell
def _(gkp_0, gkp_1, np, plt, scale):
    # Get the phase space coordinates of the delta functions for the two states
    q_coords_0 = gkp_0.means().real[:, 0]
    p_coords_0 = gkp_0.means().real[:, 1]
    q_coords_1 = gkp_1.means().real[:, 0]
    p_coords_1 = gkp_1.means().real[:, 1]

    # Determine whether the delta functions are positively or negatively weighted
    delta_sign_0 = np.sign(gkp_0.weights().real)
    delta_sign_1 = np.sign(gkp_1.weights().real)

    # Plot the locations and signs of the deltas
    # fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(6.6, 3.2),   # ~ PRL column width
            sharex=True,
            sharey=True
        )

    scatter_kwargs = dict(
        s=12,                      # small, dense points
        # cmap="RdBu_r",
        cmap=plt.cm.RdBu,
        vmin=-1.5,
        vmax=1.5,
        linewidths=0.2,
        edgecolors="k",
    )

    sc0 = ax[0].scatter(
        q_coords_0 / scale,
        p_coords_0 / scale,
        c=delta_sign_0,
        # cmap=plt.cm.RdBu,
        # vmin=-1.5,
        # vmax=1.5,
        **scatter_kwargs
    )

    sc1 = ax[1].scatter(
        q_coords_1 / scale,
        p_coords_1 / scale,
        c=delta_sign_0,
        # cmap=plt.cm.RdBu,
        # vmin=-1.5,
        # vmax=1.5,
        **scatter_kwargs
    )

    for i in range(2):
        ax[i].set_xlim(-4.5, 4.5)
        ax[i].set_ylim(-4.5, 4.5)
        ax[i].set_xlabel(r"$q$ ($\sqrt{\pi\hbar}$ )")
        ax[i].set_aspect("equal")
        ax[i].minorticks_on()
        ax[i].tick_params(labelsize=10)

    ax[0].set_title(r"$|0\rangle_{\rm GKP}$", pad=4)
    ax[1].set_title(r"$|1\rangle_{\rm GKP}$", pad=4)

    ax[0].set_ylabel(r"$p$ ($\sqrt{\pi\hbar}$ )")

    # # Single colorbar, shared
    # cbar = fig.colorbar(
    #     sc0,
    #     ax=ax,
    #     fraction=0.046,
    #     pad=0.04
    # )
    # cbar.set_label("Delta sign", fontsize=10)
    # cbar.ax.tick_params(labelsize=9)

    fig.tight_layout(w_pad=1.2)
    plt.savefig("ideal_gkp_state_wigner")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
