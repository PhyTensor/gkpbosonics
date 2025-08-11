import marimo

__generated_with = "0.13.15"
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
    import matplotlib.pyplot as plt

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
    return np, plt, sf


@app.cell
def _(np, plt, sf):
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
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        cont0 = ax[0].contourf(xvec, xvec, W0, 100, cmap="RdBu_r")
        cont1 = ax[1].contourf(xvec, xvec, W1, 100, cmap="RdBu_r")
        # ax.set_title(title)
        ax[0].set_xlabel(r"$q$ (units of $\sqrt{\pi\hbar}$)")
        ax[1].set_xlabel(r"$q$ (units of $\sqrt{\pi\hbar}$)")
        ax[0].set_ylabel(r"$p$ (units of $\sqrt{\pi\hbar}$)")

        # ax[0].set_aspect("equal")
        # ax[1].set_aspect("equal")

        ax[0].set_title(r"$|0\rangle_{\rm gkp}$", fontsize=11)
        ax[1].set_title(r"$|1\rangle_{\rm gkp}$", fontsize=11)

        fig.colorbar(cont1)
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.savefig("finite_gkp_state_wigner", dpi=300)
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
