import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    from bosonic_circuit_analysis import BosonicCircuitAnalysis
    return BosonicCircuitAnalysis, np


@app.cell
def _(BosonicCircuitAnalysis, np):
    mode = 1
    epsilon=0.08
    plus_state = [np.pi / 2, 0]
    circ_test_meas = BosonicCircuitAnalysis()
    circ_test_meas.create_state(plus_state, mode, epsilon)
    circ_test_meas.execute_circuit()
    circ_test_meas.calculate_marginals(0)
    circ_test_meas.calculate_pauli_expectation_values()
    circ_test_meas.plot_marginal_distributions()
    # circ_test_meas.calculate_wigner_function(0)
    # circ_test_meas.plot_wigner_function()
    # circ_test_meas.plot_3d_wigner_function()
    return epsilon, mode, plus_state


@app.cell
def _(BosonicCircuitAnalysis, epsilon, mode, plus_state):
    circ_noisy_test_meas = BosonicCircuitAnalysis()
    circ_noisy_test_meas.create_lossy_state(plus_state, mode, epsilon, 0.85)
    circ_noisy_test_meas.execute_circuit()
    circ_noisy_test_meas.calculate_marginals(0)
    circ_noisy_test_meas.calculate_pauli_expectation_values()
    circ_noisy_test_meas.plot_marginal_distributions()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Single Qubit Clifford Gates
    """)
    return


@app.cell
def _(BosonicCircuitAnalysis, epsilon, mode, np):
    qp_state: list = [np.pi / 4, np.pi / 4]
    circ_qp_disp = BosonicCircuitAnalysis()
    circ_qp_disp.create_state(qp_state, mode, epsilon)
    circ_qp_disp.execute_circuit()
    circ_qp_disp.calculate_marginals(0)
    circ_qp_disp.calculate_pauli_expectation_values()
    circ_qp_disp.plot_marginal_distributions()
    return


if __name__ == "__main__":
    app.run()
