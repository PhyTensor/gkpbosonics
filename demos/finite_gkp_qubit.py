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
    from numpy import pi, sqrt, ndarray

    import random

    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colorbar, colors
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt
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
    from strawberryfields import ops, Engine, Program, Result
    from strawberryfields.backends import BaseBosonicState
    from strawberryfields.ops import (
        GKP,
        BSgate,
        Coherent,
        LossChannel,
        MeasureP,
        MeasureX,
        MeasureHomodyne,
        MeasureFock,
        Squeezed,
        Xgate,
        Zgate,
        Rgate,
        Dgate,
        S2gate,
        CXgate,
        Fock,
    )

    # set the random seed
    np.random.seed(42)
    return TwoSlopeNorm, np, plt, sf


@app.cell
def _(TwoSlopeNorm, np, plt, sf):
    # Parameters
    eta = 0.85  # random.random() # np.random.normal(0, 1, 1) # 0.6
    gain = 1 / eta
    r = np.arccosh(np.sqrt(gain))  # squeezing parameter
    print(rf"$\eta$={eta}\t gain={gain}\t r={r}")

    # alpha = 1.0
    # cutoff = 15

    qubit_state: list = [0, 0]
    epsilon: float = 0.0631

    # Grid for Wigner function
    xvec = np.linspace(-5, 5, 200)


    def get_wigner(state, mode=0):
        """Compute Wigner function of a single mode"""
        return state.wigner(mode, xvec, xvec)


    def plot_wigner(W0, W1, title):
        """Plot Wigner function using matplotlib"""
        fig, ax = plt.subplots(
            1,
            2,
            # figsize=(12, 6)
            figsize=(6.6, 3.2),   # APS two-column width
            sharex=True,
            sharey=True
        )

        # Symmetric normalization around zero
        wmax = max(abs(W0).max(), abs(W1).max())
        norm = TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)
        levels = 80  # enough to be smooth, not noisy
    
        cont0 = ax[0].contourf(xvec, xvec, W0, levels=levels, cmap="RdBu_r", norm=norm)
        cont1 = ax[1].contourf(xvec, xvec, W1, levels=levels, cmap="RdBu_r", norm=norm)

        for a in ax:
            a.set_aspect("equal")
            a.set_xlabel(r"$q\;(\sqrt{\pi\hbar})$")
            a.minorticks_on()
            a.tick_params(labelsize=10)
    
        # ax.set_title(title)
        # ax[0].set_xlabel(r"$q$ (units of $\sqrt{\pi\hbar}$)")
        # ax[1].set_xlabel(r"$q$ (units of $\sqrt{\pi\hbar}$)")
        ax[0].set_ylabel(r"$p$ ($\sqrt{\pi\hbar}$)")

        ax[0].set_title(r"$|0\rangle_{\rm gkp}$", pad=4)
        ax[1].set_title(r"$|1\rangle_{\rm gkp}$", pad=4)

        # fig.colorbar(cont1)
        # Single shared colorbar
        cbar = fig.colorbar(
            cont1,
            ax=ax[1],
            fraction=0.046,
            pad=0.04
        )
        cbar.set_label(r"$W(q,p)$", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    
        fig.tight_layout(w_pad=1.4)

        # plt.grid(True, linestyle="--", alpha=0.3)
        plt.savefig("finite_gkp_state_wigner")
        plt.show()


    eng = sf.Engine("bosonic")

    prog0 = sf.Program(1)
    with prog0.context as q:
        sf.ops.GKP(qubit_state, epsilon) | q

    prog1 = sf.Program(1)
    with prog1.context as q:
        sf.ops.GKP(qubit_state, epsilon) | q
        sf.ops.Xgate(np.sqrt(np.pi * sf.hbar)) | q


    state_gkp0 = eng.run(prog0).state
    state_gkp1 = eng.run(prog1).state

    W0_compensated = get_wigner(state_gkp0, mode=0)
    W1_compensated = get_wigner(state_gkp1, mode=0)

    plot_wigner(
        W0_compensated,
        W1_compensated,
        f"After Loss + Amplification (Gain = {gain:.2f})",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
