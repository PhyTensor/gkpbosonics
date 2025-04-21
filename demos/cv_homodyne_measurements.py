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
        # Studying Realistic Bosonic GKP Qubits

        ## Qubit Pauli measurements: CV homodyne measurements

        Pauli measurements lie as the correspondence between GKP qubit operations and Gaussian resources.

        GKP qubits can be implemented by performing homodyne measurements and processing the outcomes.

        The Pauli Z operator is associated with the position $q$ operator. The Pauli X operator is associated with the momentum $p$ operator. The Pauli Y operator is associated with the $q-p$ operator.

        The outcomes of the homodyne measurements are rounded to the nearest $n\sqrt{\pi\hbar}, n \in \mathbb{Z}$. The **parity** of $n$ is then takes to determine the value of the Pauli measurement. Note that the Pauli Y measurements can be achieved by performing a homodyne measurement along $(q - p)/\sqrt{2}$ and rescaling by $\sqrt{2}$.
        """
    )
    return


@app.cell
def _():
    import strawberryfields as sf
    from strawberryfields import Program, Engine, ops
    from strawberryfields.backends import BaseBosonicState

    import numpy as np
    from numpy import ndarray

    import matplotlib.pyplot as plt
    from matplotlib import colors, colorbar
    return (
        BaseBosonicState,
        Engine,
        Program,
        colorbar,
        colors,
        ndarray,
        np,
        ops,
        plt,
        sf,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Marginal distributions of homodyne outcomes for GKP $|+^\epsilon\rangle_{gkp}$

        This will confirm that Pauli measurements can be performed with homodyne.
        """
    )
    return


@app.cell
def _(np, sf):
    # Set the scale for phase space
    sf.hbar = 1
    scale: float = np.sqrt(sf.hbar * np.pi)
    return (scale,)


@app.cell
def _(Program, ops):
    def create_gkp_circuit(qubit_state: list, epsilon: int, num_modes: int, noise_channel: bool = False, loss_parameter: float = 1.0) -> Program:
        circuit: Program = Program(num_modes)

        with circuit.context as q:
            ops.GKP(state=qubit_state, epsilon=epsilon) | q

            if noise_channel:
                ops.LossChannel(loss_parameter) | q

        return circuit
    return (create_gkp_circuit,)


@app.cell
def _(BaseBosonicState, Engine, Program, Result):
    def execute_gkp_circuit(engine: Engine, circuit: Program) -> BaseBosonicState:
        result: Result = engine.run(program=circuit)
        gkp_state: BaseBosonicState = engine.run(circuit).state
        return gkp_state
    return (execute_gkp_circuit,)


@app.cell
def _(Engine):
    engine: Engine = Engine("bosonic")
    return (engine,)


@app.cell
def _(
    BaseBosonicState,
    Program,
    create_gkp_circuit,
    engine,
    execute_gkp_circuit,
    np,
):
    # Create a GKP |+> state
    # angles theta and phi specify the qubit state
    qubit_state_plus: list = [np.pi / 2, 0]
    epsilon: float = 0.0631

    circuit: Program = create_gkp_circuit(qubit_state_plus, epsilon, 1)
    gkp_state: BaseBosonicState = execute_gkp_circuit(engine, circuit)
    return circuit, epsilon, gkp_state, qubit_state_plus


@app.cell
def _(BaseBosonicState, ndarray, np, plt, scale, sf):
    def calculate_and_plot_marginals(state: BaseBosonicState, mode: int) -> None:
        """
        Calculates and plot the q, q-p, and p quadrature marginal distributions for a given circuit mode. These can be used to determine the Pauli  X, Y, and Z outcomes for a GKP qubit.

        Parameters:
            state (object): 'BaseBosonicState' object
            mode (int): index for the circuit mode
        """

        # Calculate the marginal distributions
        # The rotation angle in phase space is specified by phi
        marginals: ndarray = []
        phis: list = [np.pi/2, -np.pi/4, 0]
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
                axs[i].plot(quad * y_scale, marginals[i] / y_scale, 'k-')
                axs[i].set_xlim(quad[0] * y_scale, quad[-1] * y_scale)

                # Calculate Pauli expectation value
                # Blue bins are weighted +1, red bins are weighted -1
                bin_weights = 2 * (((quad * y_scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] / y_scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, quad * y_scale)
            else:
                axs[i].plot(quad / scale, marginals[i] * scale, 'k-')
                axs[i].set_xlim(quad[0] / scale, quad[-1] / scale)

                # Calculate Pauli expectation value
                # Blue bins are weighted +1, red bins are weighted -1
                bin_weights = 2 * (((quad / scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] * scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, quad / scale)

            # Color the qubit bins blue and red
            for j in range(-10, 10):
                axs[i].axvspan((2 * j - 0.5), (2 * j + 0.5), alpha=0.2, facecolor='b')
                axs[i].axvspan((2 * j + 0.5), (2 * j + 1.5), alpha=0.2, facecolor='r')

            axs[i].set_title("Homodyne data for Pauli " + paulis[i] +
                             "\n" + r'$\langle$'+paulis[i]+r'$\rangle$='+
                             str(np.around(expectations[i],2)))
            axs[i].set_xlabel(homodynes[i] + r' (units of $\sqrt{\pi\hbar}$ )', fontsize=9)

        axs[0].set_ylabel("Marginal distribution", fontsize=9)
        fig.tight_layout()
        plt.show()
    return (calculate_and_plot_marginals,)


@app.cell
def _(calculate_and_plot_marginals, gkp_state):
    calculate_and_plot_marginals(gkp_state, 0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Homodyne outcomes that fall into the blue(red) bins can be interpreted as a Pauli operator eigenvalue measurement with outcome +1(-1). The bins can be used to provide the expectation value of each Pauli operator: we integrate the marginal distributions, each multiplied by an appropriate weighting function that changes the sign of the distribution depending on the Pauli operator eigenvalue associated with the bin.

        It can be seen from the $p$-homodyne measurement that the outcomes that the results are mostly within the blue bins, so there will be a high chance of reading out a +1 eigenvalue from a Pauli X measurement, as would be expected for a standard qubit $|+\rangle$ state.

        For the other two homodyne measurements, the peaks of the marginal distributionns are effectively evenly distributed between +1 and -1 bins, just like Pauli Y and Z measurements on a standard qubit $|+\rangle$ state.
        """
    )
    return


@app.cell
def _(
    BaseBosonicState,
    Program,
    create_gkp_circuit,
    engine,
    epsilon,
    execute_gkp_circuit,
):
    qubit_state_0: list = [0, 0]

    circuit0: Program = create_gkp_circuit(qubit_state_0, epsilon, 1)
    gkp_state_0: BaseBosonicState = execute_gkp_circuit(engine, circuit0)
    return circuit0, gkp_state_0, qubit_state_0


@app.cell
def _(calculate_and_plot_marginals, gkp_state_0):
    calculate_and_plot_marginals(gkp_state_0, 0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Marginal distributions on Loss

        A major source of noise for qubits encoded in CV systems is loss or dissipation. We are going to apply a `LossChannel` operation, and observe the change in the distributions of homodyne outcomes.

        **Photon loss** is modeled as a beam splitter wit loss parameter $\eta$, e.g. $\eta = 0.9$ for $10\%$ loss.
        """
    )
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=1, step=0.01, value=0.85)
    slider
    return (slider,)


@app.cell
def _(mo, slider):
    mo.md(f"Loss parameter value: {slider.value}")
    return


@app.cell
def _(
    BaseBosonicState,
    Program,
    calculate_and_plot_marginals,
    create_gkp_circuit,
    engine,
    epsilon,
    execute_gkp_circuit,
    qubit_state_plus,
    slider,
):
    circuit_lossy: Program = create_gkp_circuit(qubit_state_plus, epsilon, 1, True, slider.value)
    # engine1: Engine = Engine("bosonic")
    gkp_lossy: BaseBosonicState = execute_gkp_circuit(engine, circuit_lossy)

    # gkp_lossy: BaseBosonicState = create_plus_state_with_loss_circuit()
    # calculate and plot marginals
    calculate_and_plot_marginals(gkp_lossy, 0)
    return circuit_lossy, gkp_lossy


@app.cell
def _(mo):
    mo.md(
        r"""
        The peaks in the homodyne distribution get broadened and shifted towards the origin, resulting in outcomes falling outside of the correct bins. This corresponds to **qubit readout errors**.

        **NOTE:** study noise and develop strategies to deal with realistic effects on bosonic qubits.
        """
    )
    return


if __name__ == "__main__":
    app.run()
