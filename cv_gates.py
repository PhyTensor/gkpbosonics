import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Gates

        CVQC uses qumodes that represent bundles of interacting photons. Gaussian and non-Gaussian gates are used to perform computations on qumodes.

        Gaussian and non-Gaussian gates can be descrribed within phase space. Gaussian gates act linearly on modes. These gates can only reach positive quasi-probability distributions and can be classically simulated. However, non-Gaussian gates act nonlinearly, allowing them to be in negative probability distributions and thus cannot be classically simulated.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Single-Qubit Clifford Gates

        The association between GKP single-qubit CLifford gates and qubits is:

        Pauli X - corresponds to $q$-displacement by $\sqrt{\pi\hbar}$

        Pauli Z - corresponds to $p$-displacement by $\sqrt{\pi\hbar}$

        Hadamard - corresponds to Rotation by $\pi/2$

        Phase - corresponds to Quadratic phase gate of strength 1
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

    # Create a GKP |+> state
    # angles theta and phi specify the qubit state
    state_plus: list = [np.pi / 2, 0]
    epsilon: float = 0.0631
    return epsilon, scale, state_plus


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
        homodynes: list = ["p", "q-p", "p"]
        expectations: ndarray = np.zeros(3)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
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
                axs[i].axvspan((2 * j - 0.5), (2 * j + 0.5), alpha=0.2, facecolor='cyan')
                axs[i].axvspan((2 * j + 0.5), (2 * j + 1.5), alpha=0.2, facecolor='pink')

            axs[i].set_title("Homodyne data for Pauli " + paulis[i] +
                             "\n" + r'$\langle$'+paulis[i]+r'$\rangle$='+
                             str(np.around(expectations[i],2)))
            axs[i].set_xlabel(homodynes[i] + r' (units of $\sqrt{\pi\hbar}$ )', fontsize=15)
        axs[0].set_ylabel("Marginal distribution", fontsize=15)
        fig.tight_layout()
        plt.show()
    return (calculate_and_plot_marginals,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Qubit Pauli X and Z gates: CV $q$ and $p$ displacements

        Pauli X and Z gates in GKP encoding correspond to displacements in phase space by $\sqrt{\pi\hbar}$ along $q$ and $p$ respectively.

        Below, we take a GKP qubit with $\theta = \phi = \pi/4$ and plot its marginal distributions along $p$, $q−p$, and $q$ before and after applying a bit-phase flip. Recall that, once binned, this data provides the outcome of the Pauli measurements: blue (red) bins correspond to Pauli +1(-1) eigenvalues.
        """
    )
    return


@app.cell
def _(BaseBosonicState, Engine, Program, epsilon, ops):
    def create_gkp_state(state: list) -> BaseBosonicState:
        circuit: Program = Program(1)

        with circuit.context as q:
            ops.GKP(state=state, epsilon=epsilon) | q

        engine: Engine = Engine("bosonic")
        gkp: BaseBosonicState = engine.run(circuit).state
        return gkp
    return (create_gkp_state,)


@app.cell
def _(BaseBosonicState, create_gkp_state, np):
    # Create GKP state
    state: list = [np.pi/4, np.pi/4]
    gkp0: BaseBosonicState = create_gkp_state(state)
    return gkp0, state


@app.cell
def _(calculate_and_plot_marginals, gkp0):
    # Calcuate and plot marginals
    calculate_and_plot_marginals(gkp0, 0)
    return


@app.cell
def _(BaseBosonicState, Engine, Program, epsilon, ops, scale):
    def create_gkp_state_bitphase_flip(state: list) -> BaseBosonicState:
        circuit: Program = Program(1)

        with circuit.context as q:
            ops.GKP(state=state, epsilon=epsilon) | q
            ops.Xgate(scale) | q
            ops.Zgate(scale) | q

        engine: Engine = Engine("bosonic")
        gkp: BaseBosonicState = engine.run(circuit).state
        return gkp
    return (create_gkp_state_bitphase_flip,)


@app.cell
def _(BaseBosonicState, create_gkp_state_bitphase_flip, state):
    # Create GKP state with applied bit-phase flip
    gkp1: BaseBosonicState = create_gkp_state_bitphase_flip(state)
    return (gkp1,)


@app.cell
def _(calculate_and_plot_marginals, gkp1):
    # Calcuate and plot marginals
    calculate_and_plot_marginals(gkp1, 0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        For $q$ and $p$, regions of the distributions originally in the +1 (blue) bins get shifted to -1 (red) bins and vice versa.

        The distribution of $q-p$ remains unchanged.

        The probabilities of +1 and -1 outcomes from Pauli X and Z measurements get swapped by the bit-phase flip, while Pauli Y in invariant.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Qubit Hadamard gate: CV rotation

        The GKP Hadamard gate can be implemented by a $\pi/2$ rotation in phase space.
        """
    )
    return


@app.cell
def _(BaseBosonicState, Engine, Program, epsilon, ops):
    def create_gkp_state_rotated(state: list, theta) -> BaseBosonicState:
        circuit: Program = Program(1)

        with circuit.context as q:
            ops.GKP(state=state, epsilon=epsilon) | q
            ops.Rgate(theta=theta)

        engine: Engine = Engine("bosonic")
        gkp: BaseBosonicState = engine.run(circuit).state
        return gkp
    return (create_gkp_state_rotated,)


@app.cell
def _(BaseBosonicState, create_gkp_state_rotated, np):
    # Create GKP state with applied rotation
    theta2: float = np.pi/2
    state2: list = [np.pi/2, np.pi/2]
    gkp2: BaseBosonicState = create_gkp_state_rotated(state2, theta2)
    return gkp2, state2, theta2


@app.cell
def _(calculate_and_plot_marginals, gkp2):
    # Calcuate and plot marginals
    calculate_and_plot_marginals(gkp2, 0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Qubit Phase gate: CV Quadratic phase gate

        The GKP phase gate maps to the CV Quadratic gate.

        This is a Gaussian operation, but unlike rotations, the CV phase gate requires a squeezing operation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
