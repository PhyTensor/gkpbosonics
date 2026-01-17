from typing import Tuple
import numpy as np
import strawberryfields as sf
from strawberryfields import Engine, Program, Result
from strawberryfields.backends import BaseBosonicState
from strawberryfields.ops import GKP, LossChannel

# Set random seed
np.random.seed(42)

class SingleModeLossAnalysis:
    def __init__(self):
        self.engine: Engine = Engine("bosonic")
        sf.hbar = 1
        self.scale: float = np.sqrt(sf.hbar * np.pi)

        # Simulation parameters
        self.epsilons: np.ndarray = np.arange(0.01, 0.40, step=0.005)
        self.quad_axis: np.ndarray = np.linspace(-4, 4, 256) * self.scale
        self.qubit_state: list = [0, 0]

    def create_gkp_circuit(self, qubit_state: list, epsilon: float, transmissivity: float) -> Program:
        circuit: Program = Program(num_subsystems=1)
        with circuit.context as q:
            GKP(state=qubit_state, epsilon=epsilon) | q
            LossChannel(transmissivity) | q
        return circuit

    def execute_gkp_circuit(self, circuit: Program) -> BaseBosonicState:
        result: Result = self.engine.run(program=circuit)
        return result.state

    def calculate_marginals(self, state: BaseBosonicState, mode: int) -> list:
        # Calculate marginals for Z (0), diagonal (-pi/4), and Y (pi/2)
        # Note: Order matters for your expectation logic
        phis: list = [np.pi/2, -np.pi/4, 0]
        return [state.marginal(mode, self.quad_axis, phi=phi) for phi in phis]

    def calculate_expectation_values(self, marginals: list) -> np.ndarray:
        expectations: np.ndarray = np.zeros(3)
        for i in range(3):
            if i == 1: # Pauli Y scaling
                y_scale = np.sqrt(2 * sf.hbar) / self.scale
                bin_weights = 2 * (((self.quad_axis * y_scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] / y_scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, self.quad_axis * y_scale)
            else: # Pauli X and Z
                bin_weights = 2 * (((self.quad_axis / self.scale - 0.5) // 1) % 2) - 1
                integrand = (marginals[i] * self.scale) * bin_weights
                expectations[i] = np.trapezoid(integrand, self.quad_axis / self.scale)
        return expectations

    def calculate_logical_error_probability(self, marginal: np.ndarray, axis: np.ndarray) -> float:
        """
        Computes logical error probability for ideal GKP decoding.
        Error occurs if |q| > sqrt(pi)/2.
        """
        threshold = self.scale / 2  # sqrt(pi)/2 in physical units

        # normalisation of marginals
        marginal /= np.trapezoid(marginal, axis)

        # Boolean mask for correctable region
        mask = np.abs(axis) <= threshold

        # Probability of being correctable
        p_correctable = np.trapezoid(marginal[mask], axis[mask])

        return 1.0 - p_correctable

    def run_sweep(self, transmissivity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the simulation sweep for a specific transmissivity.
        Returns (epsilons_in_dB, Pauli_Z_expectation_values, logical_error_probability)
        """
        expectation_values: np.ndarray = np.zeros((0, 3))
        logical_errors: np.ndarray = []

        for epsilon in self.epsilons:
            circ: Program = self.create_gkp_circuit(self.qubit_state, epsilon, transmissivity)
            gkp_state: BaseBosonicState = self.execute_gkp_circuit(circ)

            marginals: list = self.calculate_marginals(gkp_state, 0)

            expectation: np.ndarray = self.calculate_expectation_values(marginals)
            expectation_values = np.vstack((expectation_values, expectation))

            # logical error probability
            q_marginal: np.ndarray = marginals[2] # phi = 0 -> position quadrature
            p_logical = self.calculate_logical_error_probability(q_marginal, self.quad_axis)
            logical_errors.append(p_logical)

        # Convert epsilon to dB for plotting x-axis
        epsilons_db = -10 * np.log10(self.epsilons)

        return epsilons_db, expectation_values, np.array(logical_errors)

