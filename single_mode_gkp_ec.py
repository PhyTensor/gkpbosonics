import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # GKP Error Correction - Single Mode

    ## Key Noise Models
    - Photon Loss -
    - Dephasing - adds random phase shifts, modeled by $e^{i\delta \hat n}$, where $\delta$ is Gaussian noise.
    """
    )
    return


@app.cell
def _():
    import strawberryfields as sf
    from strawberryfields import Program, Engine, Result
    from strawberryfields.backends import BaseBosonicState
    from strawberryfields.program_utils import RegRef
    from strawberryfields.parameters import MeasuredParameter
    from strawberryfields.ops import (GKP, BSgate, Coherent, LossChannel, MeasureP, MeasureX, MeasureHomodyne, Squeezed, Xgate, Zgate, Dgate, CZgate)

    import numpy as np
    from numpy import ndarray, sqrt, pi

    import matplotlib.pyplot as plt
    from matplotlib import colors, colorbar
    return (
        BaseBosonicState,
        Dgate,
        Engine,
        GKP,
        Program,
        Result,
        ndarray,
        np,
        pi,
        plt,
        sf,
        sqrt,
    )


@app.cell
def _(np):
    # set the random seed
    np.random.seed(42)
    return


@app.cell
def _(pi, sf, sqrt):
    # Set the scale for phase space
    sf.hbar = 1
    scale: float = sqrt(sf.hbar * pi)
    return (scale,)


@app.cell
def _():
    # Parameters
    squeezing_param: float = 0.0631       # Finite squeezing parameter
    photon_loss_param: float = 0.85           # 15% photon loss
    shots: int = 1          # Number of trials
    noise_std = 0.81      # Standard deviation of displacement noise
    return noise_std, shots, squeezing_param


@app.cell
def _(BaseBosonicState, Engine, Program, Result, shots: int):
    def execute_gkp_circuit(engine: Engine, circuit: Program) -> BaseBosonicState:
        result: Result = engine.run(program=circuit, shots=shots)
        print(f"Result Samples: {result.samples}")

        gkp_state: BaseBosonicState =result.state
        print(f"Result State: {gkp_state}")
        return gkp_state
    return (execute_gkp_circuit,)


@app.cell
def _(Dgate, GKP, Program, noise_std, np, squeezing_param: float):
    def create_gkp_circuit(qubit_state: list, epsilon: int, num_modes: int, noise_channel: bool = False, loss_parameter: float = 1.0) -> Program:
        # Initialize engine and program
        # Modes: data, ancilla_q, ancilla_p
        circuit: Program = Program(num_subsystems=num_modes)

        with circuit.context as q:
            # Prepare data qubit in |0>_GKP
            GKP(epsilon=squeezing_param) | q

            # Apply noise: sample displacements for x and p quadratures
            dx = np.random.normal(0, noise_std)
            dp = np.random.normal(0, noise_std)
            # Create a total displacement magnitude (this is simplified)
            Dgate(np.sqrt(dx**2 + dp**2)) | q

            # # Syndrome extraction (this example uses a measurement of x)
            # MeasureX | q



            # GKP(epsilon=squeezing_param) | q[1]
            # GKP(state=[np.pi/2, 0], epsilon=epsilon) | q[0]
            # # Coherent(r=r, phi=phi) | q[0]
            # # GKP(state=[0, 0], epsilon=epsilon) | q[1]
            # # GKP(state=[0, 0], epsilon=epsilon) | q[2]

            # Squeezed(-2) | q[1]
            # Squeezed(2) | q[2]

            # # apply gates
            # BS = BSgate(pi/4, pi)
            # BS | (q[0], q[1])
            # BS | (q[1], q[2])

            # Apply photon loss to data
            # LossChannel(photon_loss_param) | q[0]

            # # # Perform homodyne measurements
            # MeasureX | q[0]
            # MeasureP | q[1]

            # # # displacement gates conditioned on the measurements
            # Xgate(sqrt(2) * q[0].par) | q[2]
            # Zgate(-sqrt(2) * q[1].par) | q[2]

            # if noise_channel:
            #     LossChannel(loss_parameter) | q[0]
            #--- Measure q-displacement ---
            # Prepare ancilla for q-measurement
            # Squeezed(-2) | q[1]
            # # GKP(epsilon=squeezing_param) | q[1]
            # CZgate(1) | (q[0], q[1])      # Entangle data and ancilla
            # MeasureHomodyne(0) | q[1]     # Measure q-quadrature
            # # Quantum register reference.
            # m_q: RegRef = circuit.register[0]        # Syndrome result. tuple of all currently valid quantum modes
            # print(f"Quantum modes q: {m_q}")

            # Compute correction for q
            # residual_q = (m_q.par % np.sqrt(np.pi))
            # print(f"Residual_Q q: {residual_q} \t {type(residual_q)} \t {residual_q}")
            # # if residual_q > np.sqrt(np.pi)/2:
            # #     print(f"ok")
            # #     residual_q -= np.sqrt(np.pi)
            # residual_q = 0
            # Xgate(-residual_q) | q[0]     # Apply correction


            #--- Measure p-displacement ---
            # Prepare ancilla for p-measurement
            # Squeezed(2) | q[2]
            # # GKP(epsilon=squeezing_param) | q[2]
            # CZgate(1) | (q[0], q[2])      # Entangle data and ancilla
            # MeasureHomodyne(np.pi/2) | q[2]  # Measure p-quadrature
            # m_p: RegRef = circuit.register[1]        # Syndrome result
            # print(f"Quantum modes p: {m_p}")

            # Compute correction for p
            # residual_p = (m_p.par % np.sqrt(np.pi))
            # print(f"Residual_P p: {residual_p} \t {type(residual_p)} \t {residual_p}")
            # # if residual_p > np.sqrt(np.pi)/2:
            # #     residual_p -= np.sqrt(np.pi)
            # residual_p = 0
            # Zgate(-residual_p) | q[0]     # Apply correction

        return circuit
    return (create_gkp_circuit,)


@app.cell
def _(
    BaseBosonicState,
    Engine,
    Program,
    create_gkp_circuit,
    execute_gkp_circuit,
    squeezing_param: float,
):
    circuit: Program = create_gkp_circuit([], squeezing_param, 1)
    engine: Engine = Engine("bosonic")
    gkp_state: BaseBosonicState = execute_gkp_circuit(engine, circuit)
    return (gkp_state,)


@app.cell
def _(BaseBosonicState, ndarray, np, plt, scale: float, sf):
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
def _(calculate_and_plot_marginals, gkp_state: "BaseBosonicState"):
    calculate_and_plot_marginals(gkp_state, 0)
    return


@app.cell
def _():
    # calculate_and_plot_marginals(gkp_state, 1)
    return


@app.cell
def _():
    # calculate_and_plot_marginals(gkp_state, 2)
    return


if __name__ == "__main__":
    app.run()
