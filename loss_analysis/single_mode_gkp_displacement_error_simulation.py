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
    # GKP Displacement Error Analysis

    This simulation visualizes the effect of a unitary displacement error $D(\alpha)$ on a finite-energy GKP state.
    Unlike loss (which diffuses/broadens peaks), displacement shifts the lattice in phase space.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    import matplotlib as mpl
    from matplotlib import cm
    import strawberryfields as sf
    from strawberryfields import Engine, Program
    from strawberryfields.ops import GKP, Dgate, MeasureX, MeasureP

    # Physics Constants
    sf.hbar = 1
    SCALE = np.sqrt(sf.hbar * np.pi)
    return Dgate, Engine, GKP, MeasureP, MeasureX, Program, SCALE, mpl, np, plt


@app.cell
def _(Dgate, Engine, GKP, MeasureP, MeasureX, Program, SCALE, np):
    class DisplacementSimulation:
        """
        Manages the GKP state preparation and application of displacement noise.
        """
        def __init__(self, epsilon: float = 0.0631):
            self.epsilon = epsilon
            self.engine = Engine("bosonic")
            # State |H> (Magic state) or |+> is often used.
            # We use |+> (pi/2, 0) for clear X-squeezing visualization.
            self.qubit_params = [np.pi/2, 0]

        def get_state(self, displacement: complex = 0 + 0j):
            """
            Generates a GKP state with a specific displacement alpha.
            """
            prog = Program(1)
            with prog.context as q:
                GKP(state=self.qubit_params, epsilon=self.epsilon) | q
                if displacement != 0:
                    # Dgate takes magnitude and phase
                    mag = np.abs(displacement)
                    phase = np.angle(displacement)
                    Dgate(mag, phase) | q

            result = self.engine.run(prog)
            return result.state

        def get_ideal_samples(self, shots=2000):
            """Monte Carlo samples for the ideal (no displacement) case."""
            # Measure X
            prog_x = Program(1)
            with prog_x.context as q:
                GKP(state=self.qubit_params, epsilon=self.epsilon) | q
                MeasureX | q
            sx = self.engine.run(prog_x, shots=shots).samples[:, 0]

            # Measure P
            prog_p = Program(1)
            with prog_p.context as q:
                GKP(state=self.qubit_params, epsilon=self.epsilon) | q
                MeasureP | q
            sp = self.engine.run(prog_p, shots=shots).samples[:, 0]

            return sx, sp

        def calculate_marginals(self, state, grid_size=500, limit=6):
            """Calculates analytical marginals."""
            quad = np.linspace(-limit, limit, grid_size) * SCALE
            marg_x = state.marginal(mode=0, xvec=quad, phi=0)
            marg_p = state.marginal(mode=0, xvec=quad, phi=np.pi/2)
            return quad, marg_x, marg_p

        def calculate_wigner(self, state, grid_size=200, limit=5):
            xvec = np.linspace(-limit, limit, grid_size) * SCALE
            pvec = np.linspace(-limit, limit, grid_size) * SCALE
            wigner = state.wigner(mode=0, xvec=xvec, pvec=pvec)
            return xvec, pvec, wigner
    return (DisplacementSimulation,)


@app.cell
def _(mpl, np, plt):
    class PublicationPlotter:
        """
        Enforces academic styling for displacement analysis.
        """
        def __init__(self):
            self.configure_matplotlib()
            # Color palette
            self.c_ideal = '#333333'      # Dark Grey (Histogram)
            self.c_error = '#D55E00'      # Vermilion (Displaced Curve)
            self.c_fill  = '#56B4E9'      # Sky Blue (Ideal Fill)

        def configure_matplotlib(self):
            plt.rcParams.update({
                # "font.family": "serif",
                # "font.serif": ["Times New Roman", "Computer Modern Roman"],
                "mathtext.fontset": "cm",
                "font.size": 11,
                "axes.labelsize": 12,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "axes.grid": False,
            })

        def plot_wigner_comparison(self, x, p, w_ideal, w_noisy, filename="wigner_shift.png"):
            """
            Plots contours of both Ideal and Noisy Wigner functions to show the shift.
            """
            fig, ax = plt.subplots(figsize=(5, 5))

            limit = np.max(np.abs(w_ideal))

            # Plot Ideal as grey/neutral contours
            ax.contour(x, p, w_ideal, levels=[0.1, 0.3, 0.5],
                       colors='gray', linewidths=1.0, alpha=0.5, linestyles='--')

            # Plot Noisy as colored density
            norm = mpl.colors.Normalize(-limit, limit)
            c = ax.contourf(x, p, w_noisy, levels=100, cmap="RdBu_r", norm=norm, alpha=0.9)

            # Mark the center shift
            ax.scatter([0], [0], color='k', marker='+', s=50, label='Origin')

            cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel(r"$W(q, p)$", rotation=270, labelpad=15)

            ax.set_xlabel(r"$q$ [$\sqrt{\hbar}$]")
            ax.set_ylabel(r"$p$ [$\sqrt{\hbar}$]")
            ax.set_title(r"Phase Space Displacement")
            ax.set_aspect('equal')

            plt.savefig(filename, bbox_inches='tight')
            plt.show()

        def plot_marginal_shift(self,
                                sx, sp,        # Ideal Samples
                                quad, mx, mp,  # Displaced Analytical
                                scale,
                                shift_val,
                                filename="marginal_shift.png"):
            """
            Visualizes the shift in quadrature peaks.
            """
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            hist_opts = {'bins': 80, 'density': True, 'alpha': 0.3, 'color': self.c_fill, 'label': 'Ideal (Reference)'}
            line_opts = {'color': self.c_error, 'linewidth': 1.5, 'label': r'Displaced $\alpha$'}

            # --- Q Quadrature ---
            axs[0].hist(sx/scale, **hist_opts)
            axs[0].plot(quad/scale, mx*scale, **line_opts)

            # Add annotation for shift
            axs[0].set_xlabel(r"$q$ [$\sqrt{\pi\hbar}$]")
            axs[0].set_ylabel(r"Probability Density")
            axs[0].legend(frameon=False, loc='upper right')
            axs[0].set_title(r"Position Shift ($q$)")

            # Vertical lines to highlight offset
            peak_loc = 0
            axs[0].axvline(peak_loc, color='k', linestyle=':', alpha=0.5)
            axs[0].axvline(peak_loc + shift_val.real * np.sqrt(2), color=self.c_error, linestyle=':', alpha=0.8)

            # --- P Quadrature ---
            axs[1].hist(sp/scale, **hist_opts)
            axs[1].plot(quad/scale, mp*scale, **line_opts)

            axs[1].set_xlabel(r"$p$ [$\sqrt{\pi\hbar}$]")
            axs[1].set_yticks([])
            axs[1].set_title(r"Momentum Shift ($p$)")

            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.show()
    return (PublicationPlotter,)


@app.cell
def _(DisplacementSimulation, PublicationPlotter, SCALE):
    def run_displacement_analysis():
        # 1. Setup
        sim = DisplacementSimulation(epsilon=0.0631)
        plotter = PublicationPlotter()

        # 2. Define Displacement Error
        # We apply a specific displacement to visualize the "Shift" clearly.
        # A shift of ~0.4 is small enough to be an error, large enough to see.
        alpha_error = 0.35 + 0.0j

        print(f"Simulating Displacement Error: alpha = {alpha_error}")

        # 3. Generate Data
        state_ideal = sim.get_state(displacement=0)
        state_error = sim.get_state(displacement=alpha_error)

        # 4. Wigner Comparison (Contour overlay)
        print("Generating Wigner Comparison...")
        x, p, w_ideal = sim.calculate_wigner(state_ideal)
        _, _, w_error = sim.calculate_wigner(state_error)

        plotter.plot_wigner_comparison(x, p, w_ideal, w_error)

        # 5. Marginal Analysis (Histogram vs Curve)
        print("Generating Marginal Shift Analysis...")
        sx, sp = sim.get_ideal_samples()
        quad, mx, mp = sim.calculate_marginals(state_error)

        plotter.plot_marginal_shift(sx, sp, quad, mx, mp, SCALE, alpha_error)

        # 6. Fidelity Check (Brutal honesty: numbers matter)
        fid = state_error.fidelity_vacuum() # Note: This is usually overlap with vacuum, GKP fidelity is specific
        # Actually, let's just output the overlap between the two states if SF supported it easily,
        # but here we'll just note the visual shift.
    return (run_displacement_analysis,)


@app.cell
def _(run_displacement_analysis):
    run_displacement_analysis()
    return


if __name__ == "__main__":
    app.run()
