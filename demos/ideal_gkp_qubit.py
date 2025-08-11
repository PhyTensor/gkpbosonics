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
    from numpy import ndarray

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm, colors, colorbar

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
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    ax[0].scatter(
        q_coords_0 / scale,
        p_coords_0 / scale,
        c=delta_sign_0,
        cmap=plt.cm.RdBu,
        vmin=-1.5,
        vmax=1.5,
    )

    ax[1].scatter(
        q_coords_1 / scale,
        p_coords_1 / scale,
        c=delta_sign_0,
        cmap=plt.cm.RdBu,
        vmin=-1.5,
        vmax=1.5,
    )

    for i in range(2):
        ax[i].set_xlim(-4.5, 4.5)
        ax[i].set_ylim(-4.5, 4.5)
        ax[i].set_xlabel(r"$q$ (units of $\sqrt{\pi\hbar}$ )", fontsize=9)
        ax[i].set_aspect("equal")

    ax[0].set_title(r"$|0\rangle_{\rm gkp}$", fontsize=11)
    ax[1].set_title(r"$|1\rangle_{\rm gkp}$", fontsize=11)
    ax[0].set_ylabel(r"$p$ (units of $\sqrt{\pi\hbar}$ )", fontsize=9)

    fig.tight_layout()
    plt.savefig("ideal_gkp_state_wigner", dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
