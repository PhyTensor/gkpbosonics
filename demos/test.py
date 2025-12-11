import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
    return Engine, Program, ndarray, np, ops, plt, sf


@app.cell
def _(mo):
    mo.md(r"""Test 1""")
    return


@app.cell
def _():
    # prog = Program(2)
    # with prog.context as q:
    #     GKP(state=[np.pi/4, np.pi/4], epsilon=0.08) | q[0]
    #     # GKP(state=[np.pi/2, 0], epsilon=0.08) | q[1]
    #     # Fock(2) | q[0]
    #     Fock(4) | q[1]
    #     BSgate() | (q[0], q[1])
    #     # MeasureFock(select=0) | q[0]

    #     MeasureFock() | q[0]
    #     Dgate(q[0].par ** 2) | q[1]
    #     MeasureFock() | q[1]
    return


@app.cell
def _():
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})
    return


@app.cell
def _():
    # result = eng.run(prog)
    return


@app.cell
def _():
    # result.samples
    return


@app.cell
def _(mo):
    mo.md(r"""Test 2""")
    return


@app.cell
def _():
    # # Parameters
    # eta = 0.6                  # Loss transmissivity (e.g., 60%)
    # gain = 1 / eta             # Amplifier gain to compensate loss
    # r = np.arccosh(np.sqrt(gain))  # Two-mode squeezing parameter

    # alpha = 1.0                # Input coherent state amplitude

    # # Cutoff dimension (truncation of Fock space)
    # cutoff = 10

    # # Initialize engine
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # prog = sf.Program(2)  # 2 modes: signal and ancilla (for amplifier)

    # with prog.context as q:
    #     # Prepare initial coherent state in mode 0
    #     sf.ops.Coherent(alpha)       | q[0]

    #     # Apply loss channel to mode 0
    #     sf.ops.LossChannel(eta)      | q[0]

    #     # Prepare ancilla in vacuum (mode 1 is initialized to vacuum by default)

    #     # Apply two-mode squeezing between mode 0 (signal) and 1 (idler)
    #     sf.ops.S2gate(r, 0)          | (q[0], q[1])

    #     # Trace out mode 1 after amplification
    #     # In SF, this is effectively done by not measuring or using mode 1 further

    # state = eng.run(prog).state

    # # Get the expectation values of quadratures on the signal mode
    # x_mean, p_mean = state.quad_expectation(0)

    # print(f"⟨x⟩ = {x_mean:.3f}, ⟨p⟩ = {p_mean:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""Test 3""")
    return


@app.cell
def _():
    # # Parameters
    # eta = np.random.normal(0, 1)  # 0.6
    # gain = 1 / eta
    # r = np.arccosh(np.sqrt(gain))  # squeezing parameter
    # print(rf"$\eta$={eta}\t gain={gain}\t r={r}")

    # alpha = 1.0
    # cutoff = 15

    # # Grid for Wigner function
    # xvec = np.linspace(-5, 5, 200)


    # def get_wigner(state, mode=0):
    #     """Compute Wigner function of a single mode"""
    #     return state.wigner(mode, xvec, xvec)


    # def plot_wigner(W, title):
    #     """Plot Wigner function using matplotlib"""
    #     fig, ax = plt.subplots()
    #     cont = ax.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    #     ax.set_title(title)
    #     ax.set_xlabel("q")
    #     ax.set_ylabel("p")
    #     fig.colorbar(cont)
    #     plt.tight_layout()
    #     plt.grid(True, linestyle="--", alpha=0.3)
    #     plt.show()


    # ### 1. Original state
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # prog = sf.Program(1)

    # with prog.context as q:
    #     sf.ops.Coherent(alpha) | q[0]

    # state = eng.run(prog).state
    # W_orig = get_wigner(state)
    # plot_wigner(W_orig, "Original Coherent State")

    # ### 2. After loss only
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # prog = sf.Program(1)

    # with prog.context as q:
    #     sf.ops.Coherent(alpha) | q[0]
    #     sf.ops.LossChannel(eta) | q[0]

    # state = eng.run(prog).state
    # W_loss = get_wigner(state)
    # plot_wigner(W_loss, f"After Loss (η = {eta})")

    # ### 3. After loss + amplification
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # prog = sf.Program(2)

    # with prog.context as q:
    #     sf.ops.Coherent(alpha) | q[0]
    #     sf.ops.Coherent(0) | q[1]
    #     sf.ops.LossChannel(eta) | q[0]
    #     sf.ops.S2gate(r, 0) | (q[0], q[1])  # Two-mode squeezing = amplifier
    #     # sf.ops.MeasureHomodyne(0, select=0) | q[1]

    # state = eng.run(prog).state
    # W_compensated = get_wigner(state, mode=0)
    # plot_wigner(W_compensated, f"After Loss + Amplification (Gain = {gain:.2f})")
    return


@app.cell
def _(mo):
    mo.md(r"""Test 4""")
    return


@app.cell
def _():
    # # Parameters
    # eta = 0.85  # random.random() # np.random.normal(0, 1, 1) # 0.6
    # gain = 1 / eta
    # r = np.arccosh(np.sqrt(gain))  # squeezing parameter
    # print(rf"$\eta$={eta}\t gain={gain}\t r={r}")

    # # alpha = 1.0
    # # cutoff = 15

    # qubit_state: list = [0, 0]
    # epsilon: float = 0.0631

    # # Grid for Wigner function
    # xvec = np.linspace(-5, 5, 200)


    # def get_wigner(state, mode=0):
    #     """Compute Wigner function of a single mode"""
    #     return state.wigner(mode, xvec, xvec)


    # def plot_wigner(W, title):
    #     """Plot Wigner function using matplotlib"""
    #     fig, ax = plt.subplots()
    #     cont = ax.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    #     ax.set_title(title)
    #     ax.set_xlabel("q")
    #     ax.set_ylabel("p")
    #     fig.colorbar(cont)
    #     plt.tight_layout()
    #     plt.grid(True, linestyle="--", alpha=0.3)
    #     plt.show()


    # ### 1. Original state
    # # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # eng = sf.Engine("bosonic")
    # prog = sf.Program(1)

    # with prog.context as q:
    #     # sf.ops.Coherent(alpha) | q[0]
    #     sf.ops.GKP(qubit_state, epsilon) | q[0]

    # state = eng.run(prog).state
    # W_orig = get_wigner(state)
    # plot_wigner(W_orig, "Original Coherent State")

    # ### 2. After loss only
    # # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # eng = sf.Engine("bosonic")
    # prog = sf.Program(1)

    # with prog.context as q:
    #     # sf.ops.Coherent(alpha)   | q[0]
    #     sf.ops.GKP(qubit_state, epsilon) | q[0]
    #     sf.ops.LossChannel(eta) | q[0]

    # state = eng.run(prog).state
    # W_loss = get_wigner(state)
    # plot_wigner(W_loss, f"After Loss (η = {eta})")

    # ## 3. After loss + amplification
    # # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # eng = sf.Engine("bosonic")
    # prog = sf.Program(2)

    # with prog.context as q:
    #     sf.ops.GKP(qubit_state, epsilon) | q[0]
    #     # sf.ops.GKP([np.pi / 2, 0], epsilon) | q[1]
    #     sf.ops.Coherent(0) | q[1]
    #     # sf.ops.Coherent(alpha)   | q[0]
    #     sf.ops.LossChannel(eta) | q[0]
    #     # sf.ops.BSgate() | (q[1], q[0])
    #     # sf.ops.CXgate(np.pi/2) | (q[1], q[0])
    #     # sf.ops.Xgate(np.pi/2) | q[1]
    #     sf.ops.S2gate(r, 0) | (q[1], q[0])  # Two-mode squeezing = amplifier
    #     # sf.ops.MeasureHomodyne(0, select=0) | q[1]

    # state = eng.run(prog).state
    # W_compensated = get_wigner(state, mode=0)
    # plot_wigner(W_compensated, f"After Loss + Amplification (Gain = {gain:.2f})")
    return


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


    def plot_wigner(W, title):
        """Plot Wigner function using matplotlib"""
        fig, ax = plt.subplots()
        cont = ax.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
        # ax.set_title(title)
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        fig.colorbar(cont)
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()


    ### 1. Original state
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # eng = sf.Engine("bosonic")
    # prog = sf.Program(1)

    # with prog.context as q:
    #     # sf.ops.Coherent(alpha) | q[0]
    #     sf.ops.GKP(qubit_state, epsilon) | q[0]

    # state = eng.run(prog).state
    # W_orig = get_wigner(state)
    # plot_wigner(W_orig, "Original Coherent State")

    # ### 2. After loss only
    # # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    # eng = sf.Engine("bosonic")
    # prog = sf.Program(1)

    # with prog.context as q:
    #     # sf.ops.Coherent(alpha)   | q[0]
    #     sf.ops.GKP(qubit_state, epsilon) | q[0]
    #     sf.ops.LossChannel(eta) | q[0]

    # state = eng.run(prog).state
    # W_loss = get_wigner(state)
    # plot_wigner(W_loss, f"After Loss (η = {eta})")

    ## 3. After loss + amplification
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    eng = sf.Engine("bosonic")
    prog = sf.Program(2)

    with prog.context as q:
        sf.ops.GKP(qubit_state, epsilon) | q[0]
        # sf.ops.GKP([np.pi / 2, 0], epsilon) | q[1]
        sf.ops.Vacuum() | q[1]  #  Coherent(0) | q[1]
        # sf.ops.Coherent(alpha)   | q[0]
        # sf.ops.LossChannel(eta) | q[0]
        # sf.ops.BSgate() | (q[1], q[0])
        sf.ops.CXgate(np.pi / 2) | (q[1], q[0])
        # sf.ops.Xgate(np.pi/2) | q[0]
        # sf.ops.S2gate(r, 0) | (q[1], q[0])  # Two-mode squeezing = amplifier
        sf.ops.MeasureHomodyne(0, select=0) | q[1]

    state_gkp = eng.run(prog).state
    W_compensated = get_wigner(state_gkp, mode=0)
    plot_wigner(W_compensated, f"After Loss + Amplification (Gain = {gain:.2f})")
    return epsilon, state_gkp


@app.cell
def _(ndarray, np, sf, state_gkp):
    # Calculate the quadrature distributions
    # we can directly calculate the expected quadrature distributions
    scale: float = np.sqrt(np.pi * sf.hbar)
    quad_axis: ndarray = np.linspace(-4, 4, 256) * scale

    # Calculate the discretized marginal distribution of the specified mode along the x\cos\phi + p\sin\phi quadrature
    gkp_prob_x: ndarray = state_gkp.marginal(
        mode=0, xvec=quad_axis, phi=0
    )  # This is the q quadrature
    gkp_prob_p: ndarray = state_gkp.marginal(
        mode=0, xvec=quad_axis, phi=np.pi / 2
    )  # This is the p quadrature
    return gkp_prob_p, gkp_prob_x, quad_axis, scale


@app.cell
def _():
    shots: int = 2000  # Number of samples

    # eng = Engine("bosonic")
    return (shots,)


@app.cell
def _(Engine, Program, ops, shots: int):
    # Run the program again, collecting q samples this time
    circuit_gkp_x = Program(1)
    with circuit_gkp_x.context as qx:
        ops.GKP(epsilon=0.0631) | qx
        ops.MeasureX | qx

    eng_x = Engine("bosonic")
    gkp_samples_x = eng_x.run(circuit_gkp_x, shots=shots).samples[:, 0]

    # Run the program again, collecting p samples this time
    circuit_gkp_p = Program(1)
    with circuit_gkp_p.context as qp:
        ops.GKP(epsilon=0.0631) | qp
        ops.MeasureP | qp
    eng_p = Engine("bosonic")
    gkp_samples_p = eng_p.run(circuit_gkp_p, shots=shots).samples[:, 0]
    return gkp_samples_p, gkp_samples_x


@app.cell
def _(np):
    def linear2db(value: float) -> float:
        return -10 * np.log10(value)
    return (linear2db,)


@app.cell
def _(
    epsilon: float,
    gkp_prob_p: "ndarray",
    gkp_prob_x: "ndarray",
    gkp_samples_p,
    gkp_samples_x,
    linear2db,
    plt,
    quad_axis: "ndarray",
    scale: float,
):
    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"$|0^\epsilon\rangle_{GKP}$, $\epsilon=0.0631$ ("
        + str(linear2db(epsilon))
        + " db)",
        fontsize=18,
    )

    axs[0].hist(
        gkp_samples_x / scale,
        bins=100,
        density=True,
        label="Samples",
        color="cornflowerblue",
    )
    axs[0].plot(
        quad_axis / scale, gkp_prob_x * scale, "--", label="Ideal", color="tab:red"
    )
    axs[0].set_xlabel(r"q (units of $\sqrt{\pi\hbar}$)", fontsize=15)
    axs[0].set_ylabel("Pr(q)", fontsize=15)

    axs[1].hist(
        gkp_samples_p / scale,
        bins=100,
        density=True,
        label="Samples",
        color="cornflowerblue",
    )
    axs[1].plot(
        quad_axis / scale, gkp_prob_p * scale, "--", label="Ideal", color="tab:red"
    )
    axs[1].set_xlabel(r"p (units of $\sqrt{\pi\hbar}$)", fontsize=15)
    axs[1].set_ylabel("Pr(p)", fontsize=15)

    axs[1].legend()
    axs[0].tick_params(labelsize=13)
    axs[1].tick_params(labelsize=13)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0].grid(True, linestyle="--", alpha=0.2)
    axs[1].grid(True, linestyle="--", alpha=0.2)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
