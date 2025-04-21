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
        # Single-Qubit Clifford Gates

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

        return circuit
    return (create_gkp_circuit,)


@app.cell
def _(Program, np, ops, sf):
    def create_gkp_circuit_with_bit_phase_flip(qubit_state: list, epsilon: int, num_modes: int, noise_channel: bool = False, loss_parameter: float = 1.0) -> Program:
        circuit: Program = Program(num_modes)

        with circuit.context as q:
            ops.GKP(state=qubit_state, epsilon=epsilon) | q
            ops.Xgate(np.sqrt(np.pi * sf.hbar)) | q
            ops.Zgate(np.sqrt(np.pi * sf.hbar)) | q

        return circuit
    return (create_gkp_circuit_with_bit_phase_flip,)


@app.cell
def _(BaseBosonicState, Engine, Program, Result):
    def execute_gkp_circuit(engine: Engine, circuit: Program) -> BaseBosonicState:
        result: Result = engine.run(program=circuit)
        gkp_state: BaseBosonicState = engine.run(circuit).state
        return gkp_state
    return (execute_gkp_circuit,)


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


app._unparsable_cell(
    r"""
    Pauli X gate corresponds to a q-displacement by $\sqrt{\pi\hbar}$.

    Pauli Z gate corresponds to a $p$-displacement by $\sqrt{\pi\hbar}$.

    Hadamard - Rotation by $\pi/2$.

    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        r"""
        The Pauli X and Z gates for the GKP encoding correspond to displacements in phase space by $\sqrt{\pi\hbar}$ along $q$ and $p$ respectively.

        Here we initialise a GKP qubit with $\theta=\phi=\pi/4$ and plot its marginal distributions, before taking a bit-phase flip.

        Once binned, this data privides the outcome of the Pauli measurements: blue (red) bins correspond to Pauli $+1 (-1)$ eigenvalues.

        """
    )
    return


@app.cell
def _(
    BaseBosonicState,
    Program,
    create_gkp_circuit,
    engine,
    execute_gkp_circuit,
    np,
):
    qubit_state_plus: list = [np.pi / 4, np.pi / 4]
    epsilon: float = 0.0631

    circuit: Program = create_gkp_circuit(qubit_state_plus, epsilon, 1)
    gkp_state: BaseBosonicState = execute_gkp_circuit(engine, circuit)
    return circuit, epsilon, gkp_state, qubit_state_plus


@app.cell
def _(Engine):
    engine: Engine = Engine("bosonic")
    return (engine,)


@app.cell
def _(calculate_and_plot_marginals, gkp_state):
    calculate_and_plot_marginals(gkp_state, 0)
    return


@app.cell
def _(mo):
    mo.md(r"""Circuit with bit-phase flip""")
    return


@app.cell
def _(
    BaseBosonicState,
    Program,
    create_gkp_circuit_with_bit_phase_flip,
    engine,
    epsilon,
    execute_gkp_circuit,
    qubit_state_plus,
):
    circuit_qp: Program = create_gkp_circuit_with_bit_phase_flip(qubit_state_plus, epsilon, 1)
    gkp_state_qp: BaseBosonicState = execute_gkp_circuit(engine, circuit_qp)
    return circuit_qp, gkp_state_qp


@app.cell
def _(calculate_and_plot_marginals, gkp_state_qp):
    calculate_and_plot_marginals(gkp_state_qp, 0)
    return


@app.cell
def _(mo):
    mo.md(r"""We see that for $q$ and $p$ (associated with Pauli X and Pauli Z), regions of the distributions originally in the +1 blue bins get shifted to the -1 red bins and vice versa. The distribution $q-p$ remains unchanged, confirming the gates acts as expected, since the probabilities of +1 and -1 outcomes from Pauli X and Z measurements get swapped by the bit-phase flip, while the Pauli Y is invariant.""")
    return


if __name__ == "__main__":
    app.run()
