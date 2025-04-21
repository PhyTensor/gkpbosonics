from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf
from matplotlib import cm, colorbar, colors
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray, pi, sqrt
from strawberryfields import Engine, Program, Result
from strawberryfields.backends import BaseBosonicState, BaseState
from strawberryfields.ops import (GKP, BSgate, Coherent, CXgate, Dgate,
                                  LossChannel, MeasureHomodyne, MeasureP,
                                  MeasureX, Rgate, S2gate, Squeezed, Xgate,
                                  Zgate)

# set the random seed
np.random.seed(42)

class LossAnalysis:
    def __init__(self):
        self.engine: Engine = Engine("bosonic")

        # list of wigner functions
        self.wigners: List[ndarray] = []
        # list of marginal distributions
        # self.marginals: ndarray = np.zeros((0, 4))
        # list of Pauli expectation values
        self.expectation_values: ndarray = np.zeros((0, 3))

        # Set the scale for phase space
        sf.hbar = 1
        self.scale: float = np.sqrt(sf.hbar * np.pi)
        # Choose some values of epsilon
        self.epsilons: ndarray = np.arange(0.01, 0.40, step=0.005) # [0.08, 0.0631] # [0.0631, 0.1, 0.21, 5.1, 15.8489]
        # Choose some values of loss_parameter
        # self.loss_transmissivities: List[float] = [0.95, 0.85 ] # [1.0, 0.99, 0.95, 0.90, 0.85, 0.80, 0.75]
        # Pick a region of phase space to plot
        self.quad_axis: ndarray = np.linspace(-4, 4, 256) * self.scale
        # angles theta and phi specify the qubit state
        self.qubit_state: list = [0, 0]


    @staticmethod
    def linear2db(value: float) -> float:
        return - 10 * np.log10(value)

    @staticmethod
    def db2linear(value: float) -> float:
        return 10 ** (value / 10)

    def create_gkp_circuit(self, qubit_state: list, epsilon: float, num_modes: int, transmissivity: float = 1.0) -> Program:
        circuit: Program = Program(num_subsystems=num_modes)

        with circuit.context as q:
            GKP(state=qubit_state, epsilon=epsilon) | q
            LossChannel(transmissivity) | q

        return circuit

    def execute_gkp_circuit(self, circuit: Program) -> BaseBosonicState:
        result: Result = self.engine.run(program=circuit)
        # print(f"Result Samples: {result.samples}")

        gkp_state: BaseBosonicState = result.state
        # print(f"Result State: {gkp_state}")
        return gkp_state


    def calculate_wigner_functions(self, state: BaseBosonicState, mode: int):
        """Calculate the Wigner function for each state"""
        wigner: ndarray = state.wigner(mode=mode, xvec=self.quad_axis, pvec=self.quad_axis)
        self.wigners.append(wigner)


    def analyse(self, transmissivity: float) -> ndarray:
        marginals: list = []
        expectation_values: ndarray = np.zeros((0, 3))
        for epsilon in self.epsilons:
            circ: Program = self.create_gkp_circuit(self.qubit_state, epsilon, 1, transmissivity)
            gkp_state: BaseBosonicState = self.execute_gkp_circuit(circ)
            print(f"Mean Photon Number, Variance = {gkp_state.mean_photon(0)}")

            # Calculate the marginal distributions
            marginal: list = self.calculate_marginals(gkp_state, 0, epsilon)
            expectation: ndarray = self.calculate_expectation_values(marginal, epsilon)
            expectation_values = np.vstack((expectation_values, expectation))

            marginals.append(marginal)

        # print(f"{transmissivity} Marginals:\n{marginals}\n{expectation_values}")

        return expectation_values


    def calculate_marginals(self, state: BaseBosonicState, mode: int, epsilon: float) -> list:
        """Calculate marginal distributions for each state"""
        # The quadrature angle in phase space is specified by phi
        marginals: list = []
        phis: list = [np.pi/2, -np.pi/4, 0]
        for phi in phis:
            marginals.append(state.marginal(mode, self.quad_axis, phi=phi))

        # print(f"{epsilon} Marginals: {marginals}")
        # self.marginals.append(marginals)
        # marginals_ndarray: ndarray = np.array(marginals)
        # print(f"Marginals ndarray: {marginals_ndarray}")
        # self.marginals = np.vstack((self.marginals, marginals_ndarray))

        # populate the expectation values
        # self.calculate_expectation_values(marginals)
        return marginals


    def calculate_expectation_values(self, marginals: list, epsilon: float) -> ndarray:
        """Calcualate the Pauli expectation values for each state"""
        expectations: ndarray = np.zeros(3)
        for i in range(3):
            if i == 1:
                # rescale the outcomes for Pauli Y
                y_scale = np.sqrt(2 * sf.hbar) / self.scale
                bin_weights = 2 * (((self.quad_axis * y_scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] / y_scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, self.quad_axis * y_scale)
            else:
                bin_weights = 2 * ( ( (self.quad_axis / self.scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] * self.scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, self.quad_axis / self.scale)

        # print(f"{epsilon} Expectation Values: {expectations}")
        # self.expectation_values.append(expectations)
        # self.expectation_values = np.vstack((self.expectation_values, expectations[2]))
        return expectations


def linear2db(value):
    return - 10 * np.log10(value)

def db2linear(value):
    return 10 ** (value / 10)

if __name__ == "__main__":
    loss_analysis: LossAnalysis = LossAnalysis()

    loss_transmissivities: List[float] = [1.0, 0.95, 0.85, 0.70, 0.60, 0.2]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(loss_transmissivities)))

    for idx, transmissivity in enumerate(loss_transmissivities):
        exps: ndarray = loss_analysis.analyse(transmissivity=transmissivity)

        y_vals: ndarray = exps[:, 2]
        x_vals: ndarray = loss_analysis.epsilons
        x_vals_dev: ndarray = linear2db(x_vals)
        ax.plot(x_vals_dev, y_vals, color=colors[idx], label=rf"$\eta = {transmissivity}$")

        # Annotate the line with its transmissivity value
        # label_x = x_vals[len(x_vals)//3]
        # label_y = y_vals[len(y_vals)//3]
        # ax.text(label_x, label_y, fr'$\eta$ = {transmissivity}', fontsize=10,
                # verticalalignment='bottom', horizontalalignment='right')

    # print(expectation_values)
    # secax = ax.secondary_xaxis('top', functions=(linear2db, db2linear))
    # secax.set_xlabel('dB')

    # for i in range(len(loss_transmissivities)):
    #     plt.plot(loss_analysis.epsilons, expectation_values[i][:, 2])
    ax.set_xlabel(r'$\varepsilon$ [dB]')
    ax.set_ylabel(r'$\langle Z \rangle$')

    secax = ax.secondary_xaxis('top', functions=(linear2db, db2linear))
    secax.set_xlabel(r'$\varepsilon$')

    # ax.invert_xaxis() # plt.gca().invert_xaxis()  # Reverse x-axis
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

