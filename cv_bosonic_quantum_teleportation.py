import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import strawberryfields as sf
    from matplotlib import colorbar, colors
    from numpy import pi, sqrt
    from strawberryfields import Engine, Program, Result
    from strawberryfields.backends import BaseBosonicState
    from strawberryfields.ops import (GKP, BSgate, Coherent, LossChannel,
                                      MeasureP, MeasureX, Squeezed, Xgate,
                                      Zgate, Dgate, S2gate)

    # set the random seed
    np.random.seed(42)
    return (
        BSgate,
        BaseBosonicState,
        Coherent,
        Dgate,
        Engine,
        GKP,
        LossChannel,
        MeasureP,
        MeasureX,
        Program,
        Result,
        S2gate,
        Squeezed,
        Xgate,
        Zgate,
        colorbar,
        colors,
        np,
        pi,
        plt,
        sf,
        sqrt,
    )


@app.cell
def _():
    # from strawberryfields import ops

    # # create a 2-mode quantum program
    # prog = sf.Program(2)

    # # create a free parameter named 'a'
    # a = prog.params('a')

    # # define the program
    # with prog.context as q:
    #     ops.Dgate(a ** 2)    | q[0]  # free parameter
    #     ops.MeasureX         | q[0]  # measure qumode 0, the result is used in the next operation
    #     print(q[0].par)
    #     print(sf.math.sin(q[0].par))
    #     ops.Sgate(1 - sf.math.sin(q[0].par)) | q[1]  # measured parameter
    #     ops.MeasureFock()    | q[1]

    # # initialize the Fock backend
    # # eng = sf.Engine('fock', backend_options={'cutoff_dim': 5})

    # # run the program, with the free parameter 'a' bound to the value 0.9
    # # result = eng.run(prog, args={'a': 0.9})
    return


@app.cell
def _(pi, sf, sqrt):
    # Set the scale for phase space
    sf.hbar = 1
    scale: float = sqrt(sf.hbar * pi)
    gain = 1.0
    return gain, scale


@app.cell
def _(np):
    alpha: complex = 1 + 0.5j
    r: float = np.abs(alpha)
    phi: float = np.angle(alpha)
    print(f"alpha: {alpha} \t r: {r} \t phi: {phi}")
    return alpha, phi, r


@app.cell
def _(BaseBosonicState, Engine, Program, Result):
    def execute_gkp_circuit(engine: Engine, circuit: Program) -> BaseBosonicState:
        result: Result = engine.run(program=circuit)
        print(f"Result Samples: {result.samples}")

        gkp_state: BaseBosonicState =result.state
        print(f"Result State: {gkp_state}")
        return gkp_state
    return (execute_gkp_circuit,)


@app.cell
def _(BSgate, GKP, LossChannel, MeasureP, MeasureX, Program, S2gate, np):
    def create_gkp_circuit(qubit_state: list, epsilon: int, num_modes: int, noise_channel: bool = False, loss_parameter: float = 1.0) -> Program:
        circuit: Program = Program(num_subsystems=num_modes)

        with circuit.context as q:
            GKP(state=[0, 0], epsilon=epsilon) | q[0]
            # Coherent(r=r, phi=phi) | q[0]
            # GKP(state=[0, 0], epsilon=epsilon) | q[1]
            # GKP(state=[0, 0], epsilon=epsilon) | q[2]

            # Squeezed(-2) | q[1]
            # Squeezed(2) | q[2]
            S2gate(2) | (q[1], q[2])

            if noise_channel:
                LossChannel(loss_parameter) | q[0]

            # apply gates
            # BS = BSgate(pi/4, pi)
            # BS | (q[1], q[2])
            # BS | (q[0], q[1])

            BSgate(theta=np.pi/4, phi=0) | (q[0], q[1])

            # Perform homodyne measurements
            MeasureX | q[0]
            MeasureP | q[1]

            # # displacement gates conditioned on the measurements
            # print( sqrt(2) * q[0].par)
            # Xgate(sqrt(2) * q[0].par) | q[2]
            # Zgate(-sqrt(2) * q[1].par) | q[2]

            # if noise_channel:
            #     LossChannel(loss_parameter) | q[0]

            # print(f"Circuit params: {circuit.params()}")
            # x_m = q[0].par  # x measurement from mode 0.
            # p_m = q[1].par  # p measurement from mode 1.
            # print(f"x_m = {x_m} of type {type(x_m)}")
            # print(f"p_m = {p_m} of type {type(p_m)}")

            # amplitude = gain * np.abs(x_m + 1j*p_m)
            # print(f"Amplitude: {amplitude} of type {type(amplitude)}")
            # # phase = np.angle(x_m + 1j*p_m)
            # phase = 1 - sf.math.sin(x_m + p_m)
            # # print(f"Phase: {phase} of type {type(phase)}")
            # print(f"Phase: {phase}")
            # Dgate(amplitude, phase) | q[2]

        print(f"Circuit register par: {circuit.register[0].par}")

        circuit.print()

        print(f"Compilte Info: {circuit.compile_info}")

        return circuit
    return (create_gkp_circuit,)


@app.cell
def _():
    epsilon: float = 0.0631
    return (epsilon,)


@app.cell
def _(Program, create_gkp_circuit, epsilon):
    circuit: Program = create_gkp_circuit([], epsilon, 3)#, True, 0.7)
    return (circuit,)


@app.cell
def _(BaseBosonicState, Engine, circuit, execute_gkp_circuit):
    engine: Engine = Engine("bosonic")
    gkp_state: BaseBosonicState = execute_gkp_circuit(engine, circuit)
    print(gkp_state.is_pure)
    return engine, gkp_state


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
def _():
    # calculate_and_plot_marginals(gkp_state, 0)
    return


@app.cell
def _():
    # calculate_and_plot_marginals(gkp_state, 1)
    return


@app.cell
def _(calculate_and_plot_marginals, gkp_state):
    calculate_and_plot_marginals(gkp_state, 2)
    return


if __name__ == "__main__":
    app.run()
