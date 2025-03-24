import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Sampling quadratures of non-Gaussian states

        Sampling quadrature measurements (homodyne) in non-Gaussian states.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## GKP Quadratures

        GKP states can be understood as superpositions of multiple squeezed states.

        We are exploring the distributions of its quadratures. The distributions of the position $q$ and momentum $p$ quadratures can be obtained by integrating the Wigner function:

        $$
        Pr(q = x) = \int_{-\infty}^{\infty} dp W(x, p) \\
        Pr(p = y) = \int_{-\infty}^{\infty} dq W(q, y)
        $$

        Even though the Wigner function can be negative since it is a quasiprobability distribution, the quadrature distributions $Pr(q=x)$ and $Pr(p=y)$ are proper probability distributions ie.e. they are non-negative and their integrals equal unity (normalised).
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from numpy import ndarray

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm

    from strawberryfields import hbar, Program, ops, Engine, Result
    from strawberryfields.ops import LossChannel
    from strawberryfields.backends import BaseBosonicState
    return (
        BaseBosonicState,
        Engine,
        LossChannel,
        Program,
        Result,
        cm,
        hbar,
        mpl,
        ndarray,
        np,
        ops,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create a GKP States""")
    return


@app.cell
def _():
    # Finite energy parameter of the state
    epsilon: float = 0.0631
    return (epsilon,)


@app.cell
def _(Program, epsilon, ops):
    def create_gkp_state(num_modes: int) -> Program:
        # Initialise a single mode photonic quantum circuit
        circuit_gkp: Program = Program(num_subsystems=num_modes, name='q')
        # Initialise local quantum program executor engine. Execute program on chosen - Here it is a bosonic backend
        # local backend, making result available via the Result.


        with circuit_gkp.context as q:
            # Prepare a mode in finite GKP state
            ops.GKP(epsilon=epsilon) | q
            # LossChannel(0.85) | q

        return circuit_gkp
    return (create_gkp_state,)


@app.cell
def _(BaseBosonicState, Engine, Program, Result):
    def execute_gkp_circuit(engine_bosonic: Engine, circuit_gkp: Program) -> BaseBosonicState:
        # Execute Program by sending to the backend
        # Returns result of a quantum computation
        result: Result = engine_bosonic.run(program=circuit_gkp)

        # Represents a GKP state as linear combination of Gaussian functions in phase space
        state_gkp: BaseBosonicState = result.state

        return state_gkp
    return (execute_gkp_circuit,)


@app.cell
def _(BaseBosonicState, Engine, create_gkp_state, execute_gkp_circuit):
    engine_bosonic: Engine = Engine("bosonic")
    circuit_gkp = create_gkp_state(num_modes=1)
    state_gkp: BaseBosonicState = execute_gkp_circuit(engine_bosonic, circuit_gkp)
    return circuit_gkp, engine_bosonic, state_gkp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Visualisation of the GKP state.

        To access the quadrature distributions, use the `marginal` method of the `state` object.
        """
    )
    return


@app.cell
def _(hbar, ndarray, np, state_gkp):
    # Calculate the quadrature distributions
    # we can directly calculate the expected quadrature distributions
    scale: float = np.sqrt(np.pi * hbar)
    quad_axis: ndarray = np.linspace(-4, 4, 256) * scale

    # Calculate the discretized marginal distribution of the specified mode along the x\cos\phi + p\sin\phi quadrature
    gkp_prob_x: ndarray = state_gkp.marginal(mode=0, xvec=quad_axis, phi=0)  # This is the q quadrature
    gkp_prob_p: ndarray = state_gkp.marginal(mode=0, xvec=quad_axis, phi=np.pi / 2)  # This is the p quadrature
    return gkp_prob_p, gkp_prob_x, quad_axis, scale


@app.cell
def _(ndarray, quad_axis, state_gkp):
    # Calculate the discretized Wigner function of the specified mode.
    # containing reduced Wigner function values for specified x and p values.
    wigner_gkp: ndarray = state_gkp.wigner(mode=0, xvec=quad_axis, pvec=quad_axis)
    return (wigner_gkp,)


@app.cell
def _(cm, mpl, ndarray, np, plt):
    def wigner_contour_plot(X: ndarray, P: ndarray, Z: ndarray) -> None:
        """
        """
        color_scale: float = np.max(Z.real)
        nrm: mpl.colors.Normalize = mpl.colors.Normalize(-color_scale, color_scale)

        plt.axes().set_aspect("equal")
        plt.contourf(X, P, Z, 120, cmap=cm.RdBu, norm=nrm)
        plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.tight_layout()
        plt.show()
    return (wigner_contour_plot,)


@app.cell
def _(ndarray, np, plt):
    def wigner_3d_plot(X: ndarray, P: ndarray, Z: ndarray) -> None:
        """
        """
        fig = plt.figure(figsize=(10, 6))
        X, P = np.meshgrid(X, P)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
        # plt.axes().set_aspect("equal")
        # fig.set_size_inches(4.8, 5)
        # ax.set_axis_off()
        plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.show()
    return (wigner_3d_plot,)


@app.cell
def _(quad_axis, wigner_contour_plot, wigner_gkp):
    wigner_contour_plot(X=quad_axis, P=quad_axis, Z=wigner_gkp)
    return


@app.cell
def _(quad_axis, wigner_3d_plot, wigner_gkp):
    wigner_3d_plot(X=quad_axis, P=quad_axis, Z=wigner_gkp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding GKP Quadrature Distribution

        We want to generate samples of each given quadrature; we can collect samples of the quadratures to simulate the state being probed with homodyne measurements:

        And then we verify with visualisation that the samples distribute according to the expected distribution.
        """
    )
    return


@app.cell
def _(Engine, Program, ops):
    shots: int = 2000  # Number of samples

    # Run the program again, collecting q samples this time
    circuit_gkp_x = Program(1)
    with circuit_gkp_x.context as qx:
        ops.GKP(epsilon=0.0631) | qx
        ops.MeasureX | qx
    eng = Engine("bosonic")
    gkp_samples_x = eng.run(circuit_gkp_x, shots=shots).samples[:, 0]

    # Run the program again, collecting p samples this time
    circuit_gkp_p = Program(1)
    with circuit_gkp_p.context as qp:
        ops.GKP(epsilon=0.0631) | qp
        ops.MeasureP | qp
    eng = Engine("bosonic")
    gkp_samples_p = eng.run(circuit_gkp_p, shots=shots).samples[:, 0]
    return (
        circuit_gkp_p,
        circuit_gkp_x,
        eng,
        gkp_samples_p,
        gkp_samples_x,
        qp,
        qx,
        shots,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compare Results""")
    return


@app.cell
def _(np):
    def linear2db(value: float) -> float:
        return abs(round(10 * np.log10(value)))
    return (linear2db,)


@app.cell
def _(
    epsilon,
    gkp_prob_p,
    gkp_prob_x,
    gkp_samples_p,
    gkp_samples_x,
    linear2db,
    plt,
    quad_axis,
    scale,
):
    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(r"$|0^\epsilon\rangle_{GKP}$, $\epsilon=0.0631$ ("+ str(linear2db(epsilon)) +" db)", fontsize=18)

    axs[0].hist(gkp_samples_x / scale, bins=100, density=True, label="Samples", color="cornflowerblue")
    axs[0].plot(quad_axis/ scale, gkp_prob_x * scale, "--", label="Ideal", color="tab:red")
    axs[0].set_xlabel(r"q (units of $\sqrt{\pi\hbar}$)", fontsize=15)
    axs[0].set_ylabel("Pr(q)", fontsize=15)

    axs[1].hist(gkp_samples_p / scale, bins=100, density=True, label="Samples", color="cornflowerblue")
    axs[1].plot(quad_axis/ scale, gkp_prob_p * scale, "--", label="Ideal", color="tab:red")
    axs[1].set_xlabel(r"p (units of $\sqrt{\pi\hbar}$)", fontsize=15)
    axs[1].set_ylabel("Pr(p)", fontsize=15)

    axs[1].legend()
    axs[0].tick_params(labelsize=13)
    axs[1].tick_params(labelsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return axs, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The comb of peaks are clearly visible in both the quadratures, as are visible the Gaussian spread of the individual peaks and the Gaussian envelope on the height of all the peaks.""")
    return


if __name__ == "__main__":
    app.run()
