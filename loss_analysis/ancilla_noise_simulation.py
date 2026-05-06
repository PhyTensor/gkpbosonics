import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Simulating Ancilla-Induced Noise

    This notebook simulates the degradation of a data qubit when coupled to a noisy (finitely-squeezed) ancilla qubit. We prepare the data qubit in the logical $|+\rangle$ state and the ancilla in $|0\rangle$, then couple them with a `CXgate`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import strawberryfields as sf
    from strawberryfields import Program, Engine
    from strawberryfields.ops import GKP, CXgate

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.minor.width": 0.6,
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return CXgate, Engine, GKP, Program, np, plt, sf


@app.cell
def _(np):
    def linear2db(value: float) -> float:
        return -10 * np.log10(2 * value)

    def db2linear(db: float) -> float:
        return 0.5 * (10 ** (-db / 10))
    return (db2linear,)


@app.cell
def _(CXgate, Engine, GKP, Program, db2linear, np, plt, sf):
    def run_simulation():
        sf.hbar = 1
        scale = np.sqrt(sf.hbar * np.pi)
        # CRITICAL FIX: Expanded integration window from [-5, 5] to [-20, 20]. 
        # Highly squeezed GKP states have very wide momentum envelopes that were leaking out of the [-5, 5] box!
        quad_axis = np.linspace(-20, 20, 1000) * scale

        # Fixed Data Qubit Squeezing (approx 12 dB)
        eps_data = db2linear(12.0)

        # Sweep Ancilla Squeezing from 2 dB to 12 dB to clearly see the drop at low squeezing
        ancilla_squeezing_db = np.linspace(2, 16, 10)
        x_expectations = []

        print("Running 2-Mode Ancilla Noise Simulations...")
        eng = Engine("bosonic")

        for anc_db in ancilla_squeezing_db:
            eps_anc = db2linear(anc_db)

            # 2-Mode Circuit: [Data, Ancilla]
            prog = Program(2)
            with prog.context as q:
                # Prepare Data in logical |+> state to measure X fidelity
                GKP(state=[np.pi/2, 0], epsilon=eps_data) | q[0]
                # Prepare Ancilla in logical |+> state (CRITICAL: this protects the data qubit's X quadrature!)
                GKP(state=[np.pi/2, 0], epsilon=eps_anc) | q[1]

                # Couple Data and Ancilla (standard continuous-variable CNOT)
                CXgate(1.0) | (q[0], q[1])

            # Execute
            result = eng.run(prog)
            state = result.state

            # Trace out ancilla and get marginal on data mode's momentum quadrature
            marginal_p = state.marginal(mode=0, xvec=quad_axis, phi=np.pi/2)

            # Integrate to find logical Pauli X expectation
            bin_weights = 2 * (((quad_axis / scale - 0.5) // 1) % 2) - 1
            integrand = (marginal_p * scale) * bin_weights
            exp_x = np.trapezoid(integrand, quad_axis / scale)

            x_expectations.append(exp_x)
            print(f"  -> Ancilla Squeezing: {anc_db:.1f} dB | Data <X>: {exp_x:.4f}")

        # Plotting
        print("Generating Plot...")
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(ancilla_squeezing_db, x_expectations, marker='o', color='tab:red', label=r"Data $\epsilon=12$ dB")

        ax.set_xlabel(r"Ancilla Squeezing [dB]")
        ax.set_ylabel(r"Data Qubit Fidelity $\langle X \rangle$")
        #ax.set_title("Degradation of Data Qubit via Noisy Ancilla Coupling")
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.legend()

        plt.tight_layout()
        #filename = "/home/karoki/Documents/msc_thesis/Thesis/figures/ancilla_noise_analysis.png"
        filename = "ancilla_noise_analysis"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()
        return fig
    return (run_simulation,)


@app.cell
def _(run_simulation):
    run_simulation()
    return


if __name__ == "__main__":
    app.run()
