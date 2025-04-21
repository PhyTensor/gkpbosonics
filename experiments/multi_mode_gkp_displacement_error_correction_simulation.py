import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Simulating Single-Mode Gaussian Displacement Noise

        Gaussian displacement noise can be simulated by applying a `Displacement` operation to the GKP qubit, where the displacement amplitude and phase are randomly sampled from a Gaussian distribution.

        One starts by preparing a GKP state and then applying a random displacement in the complex plane, with both the real and imaginary parts of the displacement drawn from a normal distribution with a specified standard deviation. To visualize the effect of this noise, the Wigner function of the noisy state is averaged over multiple simulations.
        """
    )
    return


@app.cell
def _():
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
    return (
        Axes3D,
        BSgate,
        BaseBosonicState,
        CXgate,
        Coherent,
        Dgate,
        Engine,
        GKP,
        LossChannel,
        MeasureHomodyne,
        MeasureP,
        MeasureX,
        Program,
        Result,
        Rgate,
        S2gate,
        Squeezed,
        Xgate,
        Zgate,
        cm,
        colorbar,
        colors,
        mpl,
        ndarray,
        np,
        pi,
        plt,
        sf,
        sqrt,
    )


@app.cell
def _():
    basename: str = "single_mode_gkp_displacement_correction_"
    return (basename,)


@app.cell
def _(np):
    def linear2db(value: float) -> float:
        return abs(round(10 * np.log10(value)))
    return (linear2db,)


@app.cell
def _(basename, cm, mpl, ndarray, np, plt):
    def wigner_contour_plot(X: ndarray, P: ndarray, Z: ndarray) -> None:
        """ """
        plotname: str = "wigner_contour_plot"
        color_scale: float = np.max(Z.real)
        nrm: mpl.colors.Normalize = mpl.colors.Normalize(-color_scale, color_scale)

        plt.axes().set_aspect("equal")
        plt.contourf(X, P, Z, 120, cmap=cm.RdBu, norm=nrm)
        plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.tight_layout()
        plt.savefig(fname=basename + plotname, dpi=300)
        plt.show()
    return (wigner_contour_plot,)


@app.cell
def _(basename, ndarray, np, plt):
    def wigner_3d_plot(X: ndarray, P: ndarray, Z: ndarray) -> None:
        """ """
        plotname: str = "wigner_3d_plot"
        fig = plt.figure(figsize=(10, 6))
        X, P = np.meshgrid(X, P)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
        # plt.axes().set_aspect("equal")
        # fig.set_size_inches(4.8, 5)
        # ax.set_axis_off()
        plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=9)
        plt.savefig(fname=basename + plotname, dpi=300)
        plt.show()
    return (wigner_3d_plot,)


@app.cell
def _(BaseBosonicState, basename, ndarray, np, plt, scale, sf):
    def calculate_and_plot_marginals(state: BaseBosonicState, mode: int) -> None:
        """
        Calculates and plot the q, q-p, and p quadrature marginal distributions for a given circuit mode. These can be used to determine the Pauli  X, Y, and Z outcomes for a GKP qubit.

        Parameters:
            state (object): 'BaseBosonicState' object
            mode (int): index for the circuit mode
        """

        plotname: str = "quad_marginal_distr"

        # Calculate the marginal distributions
        # The rotation angle in phase space is specified by phi
        marginals: ndarray = []
        phis: list = [np.pi / 2, -np.pi / 4, 0]
        quad: ndarray = np.linspace(-5, 5, 400) * scale
        for phi in phis:
            marginals.append(state.marginal(mode, quad, phi=phi))

        # Plot the results
        paulis: list = ["X", "Y", "Z"]
        homodynes: list = ["p", "q-p", "q"]
        expectations: ndarray = np.zeros(3)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i in range(3):
            if i == 1:
                # Rescale the outcomes for Pauli Y
                y_scale = np.sqrt(2 * sf.hbar) / scale
                axs[i].plot(quad * y_scale, marginals[i] / y_scale, "k-")
                axs[i].set_xlim(quad[0] * y_scale, quad[-1] * y_scale)

                # Calculate Pauli expectation value
                # Blue bins are weighted +1, red bins are weighted -1
                bin_weights = 2 * (((quad * y_scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] / y_scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, quad * y_scale)
            else:
                axs[i].plot(quad / scale, marginals[i] * scale, "k-")
                axs[i].set_xlim(quad[0] / scale, quad[-1] / scale)

                # Calculate Pauli expectation value
                # Blue bins are weighted +1, red bins are weighted -1
                bin_weights = 2 * (((quad / scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] * scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, quad / scale)

            # Color the qubit bins blue and red
            for j in range(-10, 10):
                axs[i].axvspan(
                    (2 * j - 0.5), (2 * j + 0.5), alpha=0.2, facecolor="b"
                )
                axs[i].axvspan(
                    (2 * j + 0.5), (2 * j + 1.5), alpha=0.2, facecolor="r"
                )

            axs[i].set_title(
                "Homodyne distribution for Pauli "
                + paulis[i]
                + "\n"
                + r"$\langle$"
                + paulis[i]
                + r"$\rangle$="
                + str(np.around(expectations[i], 2))
            )
            axs[i].set_xlabel(
                homodynes[i] + r" (units of $\sqrt{\pi\hbar}$ )", fontsize=9
            )

        axs[0].set_ylabel("Marginal Distribution", fontsize=9)
        fig.tight_layout()
        plt.savefig(fname=basename + plotname)
        plt.show()
    return (calculate_and_plot_marginals,)


@app.cell
def _(BaseBosonicState, Engine, Program, Result):
    def execute_gkp_circuit(engine: Engine, circuit: Program) -> BaseBosonicState:
        result: Result = engine.run(program=circuit)
        print(f"Result Samples: {result.samples}")

        gkp_state: BaseBosonicState = result.state
        print(f"Result State: {gkp_state}")
        return gkp_state
    return (execute_gkp_circuit,)


@app.cell
def _(CXgate, Dgate, GKP, MeasureHomodyne, Program, Xgate, np, sf):
    def create_gkp_circuit(
        qubit_state: list, epsilon: int, num_modes: int, displacement: tuple) -> Program:
        circuit: Program = Program(num_subsystems=num_modes)

        with circuit.context as q:
            # Target - arbitrary qubit
            GKP(state=qubit_state ,epsilon=epsilon) | q[0]
            # Control
            GKP(state=[np.pi/2, 0], epsilon=epsilon) | q[1]

            # Displacement Error
            Dgate(*displacement) | q[0]

            # Controlled displacement
            CXgate(np.pi/2) | (q[1], q[0])

            Xgate(np.pi/2) | q[1]

            # Rgate(theta=-np.pi/2) | q[1]

            # MeasureHomodyne(0, select=0) | q[1]
            MeasureHomodyne(0, select=np.sqrt(np.pi * sf.hbar)) | q[1]

            # Rgate(theta=-np.pi/2) | q[0]

            # Dgate( q[1].par * (2 * np.sqrt(np.pi)) ) | q[0]

        return circuit
    return (create_gkp_circuit,)


@app.cell
def _(np, sf):
    # Set the scale for phase space
    sf.hbar = 1
    scale: float = np.sqrt(sf.hbar * np.pi)
    return (scale,)


@app.cell
def _():
    noise_std = 0.91  # Standard deviation of displacement noise
    sigma = 2.92  # Standard deviation of Gaussian displacement
    return noise_std, sigma


@app.cell
def _(np):
    # Apply noise: sample displacements for x and p quadratures
    dx = 0.32 # np.random.normal(0, noise_std)
    dp = 0.32 # np.random.normal(0, noise_std)
    print(f"(dx, dp) = ( {dx}, {dp} )")
    # Create a total displacement magnitude (this is simplified)
    alpha = np.sqrt(dx**2 + 1j*dp**2)
    # alpha = (0.5*scale) + (0.5*scale*1j)
    # amplitude = np.abs(alpha)
    # phase = np.angle(alpha)
    # print(f"(dx, dp): ({dx}, {dp}) \t alpha: {alpha} \t amplitude: {amplitude} \t phase: {phase}")
    # Dgate(amplitude) | q

    # alpha = np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma)
    amplitude = np.real(alpha)
    phase = np.imag(alpha)
    print(f"alpha: {alpha}, amplitude: {amplitude}, phase: {phase}")
    return alpha, amplitude, dp, dx, phase


@app.cell
def _(np):
    # Create a GKP |+> state
    # angles theta and phi specify the qubit state
    qubit_state_param: list = [np.pi/4, np.pi/4]
    epsilon: float = 0.08631
    return epsilon, qubit_state_param


@app.cell
def _(Engine):
    engine: Engine = Engine("bosonic")
    return (engine,)


@app.cell
def _(
    Program,
    amplitude,
    create_gkp_circuit,
    epsilon,
    phase,
    qubit_state_param,
):
    circuit: Program = create_gkp_circuit(qubit_state_param, epsilon, 2, (amplitude, phase))
    return (circuit,)


@app.cell
def _(BaseBosonicState, circuit, engine, execute_gkp_circuit):
    gkp_state: BaseBosonicState = execute_gkp_circuit(engine, circuit)
    return (gkp_state,)


@app.cell
def _(calculate_and_plot_marginals, gkp_state):
    calculate_and_plot_marginals(gkp_state, 0)
    return


@app.cell
def _():
    # calculate_and_plot_marginals(gkp_state, 1)gkp_state_noerror: BaseBosonicState = execute_gkp_circuit(engine, circuit_noerror)
    return


@app.cell
def _(mo):
    mo.md(r"""The simulation illustrates that Gaussian displacement noise smears out the sharp, delta-like peaks characteristic of an ideal GKP state's Wigner function . Instead of distinct, localized peaks, the Wigner function of the noisy state resembles a broader Gaussian distribution centered around the origin. This blurring effect indicates that the state has become less well-defined in phase space, increasing the uncertainty in its quadrature values. This increased uncertainty directly translates to a higher probability of measurement errors when attempting to read out the logical state of the qubit. The severity of this effect is directly proportional to the variance of the Gaussian displacement, with larger variances leading to more significant blurring .""")
    return


@app.cell
def _(ndarray, np, scale):
    quad_axis: ndarray = np.linspace(-4, 4, 256) * scale
    return (quad_axis,)


@app.cell
def _(gkp_state, ndarray, np, quad_axis):
    # Calculate the discretized marginal distribution of the specified mode along the x\cos\phi + p\sin\phi quadrature
    gkp_prob_x: ndarray = gkp_state.marginal(
        mode=0, xvec=quad_axis, phi=0
    )  # This is the q quadrature
    gkp_prob_p: ndarray = gkp_state.marginal(
        mode=0, xvec=quad_axis, phi=np.pi / 2
    )  # This is the p quadrature
    return gkp_prob_p, gkp_prob_x


@app.cell
def _():
    # Calculate the discretized Wigner function of the specified mode.
    # containing reduced Wigner function values for specified x and p values.
    # wigner_gkp: ndarray = gkp_state.wigner(mode=0, xvec=quad_axis, pvec=quad_axis)
    return


@app.cell
def _():
    # wigner_contour_plot(X=quad_axis, P=quad_axis, Z=wigner_gkp)
    return


@app.cell
def _():
    # wigner_3d_plot(X=quad_axis, P=quad_axis, Z=wigner_gkp)
    return


@app.cell
def _(Engine, GKP, MeasureP, MeasureX, Program, epsilon, qubit_state_param):
    shots: int = 1024  # Number of samples

    # Run the program again, collecting q samples this time
    circuit_gkp_x = Program(1)
    with circuit_gkp_x.context as qx:
        GKP(state=qubit_state_param, epsilon=epsilon) | qx
        MeasureX | qx
    eng = Engine("bosonic")
    gkp_samples_x = eng.run(circuit_gkp_x, shots=shots).samples[:, 0]

    # Run the program again, collecting p samples this time
    circuit_gkp_p = Program(1)
    with circuit_gkp_p.context as qp:
        GKP(state=qubit_state_param, epsilon=epsilon) | qp
        MeasureP | qp
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


@app.cell
def _(
    basename,
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
    fig.suptitle(
        "Homodyne Distributions (expected - actual)\n"
        + r"$|0^\epsilon\rangle_{GKP}$, $\epsilon=0.0631$ ("
        + str(linear2db(epsilon))
        + " db)",
        fontsize=18,
    )

    axs[0].hist(
        gkp_samples_x / scale,
        bins=100,
        density=True,
        label="Expected (non-lossy)",
        color="cornflowerblue",
    )
    axs[0].plot(
        quad_axis / scale,
        gkp_prob_x * scale,
        "--",
        label="Actual (lossy)",
        color="tab:red",
    )
    axs[0].set_xlabel(r"q (units of $\sqrt{\pi\hbar}$)", fontsize=15)
    axs[0].set_ylabel("Pr(q)", fontsize=15)

    axs[1].hist(
        gkp_samples_p / scale,
        bins=100,
        density=True,
        label="Expected (non-lossy)",
        color="cornflowerblue",
    )
    axs[1].plot(
        quad_axis / scale,
        gkp_prob_p * scale,
        "--",
        label="Actual (lossy)",
        color="tab:red",
    )
    axs[1].set_xlabel(r"p (units of $\sqrt{\pi\hbar}$)", fontsize=15)
    axs[1].set_ylabel("Pr(p)", fontsize=15)

    axs[1].legend()
    axs[0].tick_params(labelsize=13)
    axs[1].tick_params(labelsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(basename + "marginal_distr_comparison", dpi=300)
    plt.show()
    return axs, fig


if __name__ == "__main__":
    app.run()
