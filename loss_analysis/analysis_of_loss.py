import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # GKP Code Photon Loss Analysis
    High-fidelity simulation of logical Pauli-Z expectation values under varying squeezing and loss.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib as mpl
    from matplotlib import cm
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams.update({
        # "font.family": "serif",
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
        "ytick.minor.width": 0.6,
        "xtick.top": True,
        "ytick.right": True,

        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    import matplotlib as mpl
    from single_mode_loss_analysis import SingleModeLossAnalysis
    from typing import Dict, List, Tuple
    return SingleModeLossAnalysis, np, plt


@app.cell
def _(np, plt):
    class PublicationPlotter:
        """
        Enforces academic publishing standards (PRL/Nature style).
        Uses Composition to separate style from logic.
        """
        def __init__(self):
            pass

        def plot_sweep(self, data_map: dict, filename: str):
            """
            data_map: Dictionary {transmissivity: (x_vals, y_vals)}
            """
            fig, ax = plt.subplots(figsize=(7, 5)) # Standard single-column width

            # Generate perceptually uniform colors
            transmissivities = sorted(data_map.keys(), reverse=True)
            colors = plt.cm.magma(np.linspace(0.1, 0.8, len(transmissivities)))

            for idx, eta in enumerate(transmissivities):
                x, y = data_map[eta]
                # Combine line and markers for distinctiveness in B&W
                ax.plot(x, y,
                        label=rf"$\eta = {eta:.2f}$",
                        color=colors[idx],
                        linestyle='-' if idx % 2 == 0 else '--')

            # Labels and Limits
            ax.set_xlabel(r"Squeezing parameter $\varepsilon$ [dB]")
            ax.set_ylabel(r"Expectation value $\langle Z \rangle$")

            # Scientific Grid
            # ax.grid(True, which='major', linestyle='-', alpha=0.2, color='gray')
            ax.minorticks_on()

            # Legend Placement
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2)

            fig.tight_layout(w_pad=1.2)
            plt.savefig(filename, bbox_inches='tight')
            plt.show()

        def plot_logical_error(self, data_map: dict, filename: str):
            fig, ax = plt.subplots(figsize=(7, 5))
        
            transmissivities = sorted(data_map.keys(), reverse=True)
            colors = plt.cm.magma(np.linspace(0.1, 0.8, len(transmissivities)))
        
            for idx, eta in enumerate(transmissivities):
                x, p_err = data_map[eta]
                ax.plot(
                    x,
                    p_err,
                    label=rf"$\eta = {eta:.2f}$",
                    color=colors[idx],
                    linestyle='-' if idx % 2 == 0 else '--'
                )
        
            ax.set_xlabel(r"Squeezing parameter $\varepsilon$ [dB]")
            ax.set_ylabel(r"Logical error probability $P_{\mathrm{L}}$")
            ##ax.set_yscale("log")
            ax.minorticks_on()
            ax.legend(loc="upper left", ncol=2)
        
            fig.tight_layout()
            plt.savefig(filename)
            plt.show()

    return (PublicationPlotter,)


@app.cell
def _(PublicationPlotter, SingleModeLossAnalysis):
    def run_experiment():
        # 1. Setup
        analyzer = SingleModeLossAnalysis()
        plotter = PublicationPlotter()

        # 2. Define Param Space
        # Use clean splits for better visualization
        loss_transmissivities = [1.0, 0.96, 0.93, 0.90]#, 0.85, 0.70]

        # 3. Collect Data (Simulation Phase)
        data_store = {}
        data_logical = {}
    
        print("Running Simulations...")
        for eta in loss_transmissivities:
            print(f"  -> Simulating eta={eta}")
            x_vals, exps, p_logical = analyzer.run_sweep(transmissivity=eta)

            # Extract Z expectation (index 2)
            z_vals = exps[:, 2]
            data_store[eta] = (x_vals, z_vals)
            data_logical[eta] = (x_vals, p_logical)

        # 4. Visualize (Plotting Phase)
        print("Generating Figure...")
        #plotter.plot_sweep(data_store, "single_mode_gkp_loss_analysis")
        plotter.plot_logical_error(data_logical, "gkp_logical_error_vs_squeezing")
    return (run_experiment,)


@app.cell
def _(run_experiment):
    run_experiment()
    return


if __name__ == "__main__":
    app.run()
