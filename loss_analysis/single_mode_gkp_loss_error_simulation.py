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
    # Single-Mode GKP Loss Simulation

    Visualizing the degradation of GKP states under photon loss using Wigner functions and Marginal distributions.
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
    from mpl_toolkits.mplot3d import Axes3D
    import strawberryfields as sf
    from strawberryfields import Engine, Program
    from strawberryfields.ops import GKP, Sgate, LossChannel, MeasureX, MeasureP

    # Global Physics Config
    sf.hbar = 1
    SCALE = np.sqrt(sf.hbar * np.pi)
    return (
        Engine,
        GKP,
        LossChannel,
        MeasureP,
        MeasureX,
        Program,
        SCALE,
        Sgate,
        mpl,
        np,
        plt,
    )


@app.cell
def _(Engine, GKP, LossChannel, MeasureP, MeasureX, Program, SCALE, Sgate, np):
    class GKPSimulation:
        """
        Handles the physics: State preparation, Loss simulation, and Sampling.
        """
        def __init__(self, epsilon: float = 0.0631, transmissivity: float = 0.85):
            self.epsilon = epsilon
            self.transmissivity = transmissivity
            self.engine = Engine("bosonic")
            self.qubit_state = [np.pi / 2, 0] # |+> state

        def get_lossy_state(self):
            prog = Program(1)
            with prog.context as q:
                GKP(epsilon=self.epsilon) | q
                Sgate(0.5 * np.pi) | q # Rotate to align with desired basis if needed
                LossChannel(self.transmissivity) | q

            result = self.engine.run(prog)
            return result.state

        def get_ideal_samples(self, shots: int = 2000):
            """Returns samples for X and P from a NON-LOSSY circuit."""
            # X Basis
            prog_x = Program(1)
            with prog_x.context as q:
                GKP(epsilon=self.epsilon) | q
                MeasureX | q
            samples_x = self.engine.run(prog_x, shots=shots).samples[:, 0]

            # P Basis
            prog_p = Program(1)
            with prog_p.context as q:
                GKP(epsilon=self.epsilon) | q
                MeasureP | q
            samples_p = self.engine.run(prog_p, shots=shots).samples[:, 0]

            return samples_x, samples_p

        def calculate_wigner(self, state, grid_size=200, limit=5):
            """Calculates Wigner function on a grid."""
            xvec = np.linspace(-limit, limit, grid_size) * SCALE
            pvec = np.linspace(-limit, limit, grid_size) * SCALE
            wigner = state.wigner(mode=0, xvec=xvec, pvec=pvec)
            return xvec, pvec, wigner

        def calculate_marginals(self, state, grid_size=400, limit=5):
            """Calculates analytical marginals for the lossy state."""
            quad = np.linspace(-limit, limit, grid_size) * SCALE

            # phi=0 -> x quadrature, phi=pi/2 -> p quadrature
            marg_x = state.marginal(mode=0, xvec=quad, phi=0)
            marg_p = state.marginal(mode=0, xvec=quad, phi=np.pi/2)

            return quad, marg_x, marg_p
    return (GKPSimulation,)


@app.cell
def _(mpl, np, plt):
    class PublicationPlotter:
        """
        Standardized Academic Plotting.
        """
        def __init__(self):
            self.configure_matplotlib()

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
            })

        def plot_wigner_2d(self, x, p, z, filename="wigner_2d_contour_plot"):
            """
            Standard Wigner Contour Plot.

            """
            fig, ax = plt.subplots(figsize=(5, 5))

            # Use centered normalization for Wigner (negativity is key)
            limit = np.max(np.abs(z))
            norm = mpl.colors.Normalize(-limit, limit)

            # Contourf is standard for Wigner
            c = ax.contourf(x, p, z, levels=100, cmap="RdBu_r", norm=norm)

            # Minimalist colorbar
            cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel(r"$W(q, p)$", rotation=270, labelpad=15)

            ax.set_xlabel(r"$q$ [$\sqrt{\hbar}$]")
            ax.set_ylabel(r"$p$ [$\sqrt{\hbar}$]")
            ax.set_aspect('equal')
            ax.set_title(r"Wigner Function (Lossy)")

            plt.savefig(filename, bbox_inches='tight')
            plt.show()

        def plot_wigner_3d(self, x, p, z, filename="wigner_3d_plot"):
            """
            3D Surface plot. Note: Usually discouraged in papers unless necessary.
            """
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            X, P = np.meshgrid(x, p)

            # Use a specialized colormap and lighting
            surf = ax.plot_surface(X, P, z, cmap="RdBu_r",
                                  linewidth=0, antialiased=False,
                                  rstride=2, cstride=2, alpha=0.9)

            # Clean up the box
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.set_zlabel(r"$W(q,p)$")

            # Optimize view angle for GKP peaks
            ax.view_init(elev=45, azim=-45)

            plt.savefig(filename, bbox_inches='tight')
            plt.show()

        def plot_marginal_comparison(self,
                                     samples_x, samples_p,
                                     quad_axis, marg_x, marg_p,
                                     scale,
                                     filename="marginals.png"):
            """
            Compares Ideal (Histogram) vs Lossy (Curve).
            """
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            # Plot settings
            hist_kwargs = {'bins': 80, 'density': True, 'alpha': 0.4, 'color': 'gray', 'edgecolor': 'none'}
            line_kwargs = {'color': '#D55E00', 'linewidth': 1.5, 'linestyle': '-'} # Vermilion (Colorblind safe)

            # --- Q Quadrature ---
            axs[0].hist(samples_x/scale, label="Ideal (Monte Carlo)", **hist_kwargs)
            axs[0].plot(quad_axis/scale, marg_x*scale, label="Lossy (Analytical)", **line_kwargs)

            axs[0].set_xlabel(r"$q$ [$\sqrt{\pi\hbar}$]")
            axs[0].set_ylabel(r"Probability Density")
            axs[0].legend(frameon=False, loc='upper right')
            axs[0].set_title(r"Position Quadrature ($q$)")

            # --- P Quadrature ---
            axs[1].hist(samples_p/scale, **hist_kwargs)
            axs[1].plot(quad_axis/scale, marg_p*scale, **line_kwargs)

            axs[1].set_xlabel(r"$p$ [$\sqrt{\pi\hbar}$]")
            axs[1].set_yticks([]) # Remove redundant y-ticks for clean look
            axs[1].set_title(r"Momentum Quadrature ($p$)")

            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.show()
    return (PublicationPlotter,)


@app.cell
def _(GKPSimulation, PublicationPlotter, SCALE):
    def run_simulation_analysis():
        # 1. Initialize
        sim = GKPSimulation(epsilon=0.0631, transmissivity=0.85)
        plotter = PublicationPlotter()

        print("Generating State...")
        lossy_state = sim.get_lossy_state()

        # 2. Wigner Analysis
        print("Calculating Wigner Function...")
        # x, p, wigner = sim.calculate_wigner(lossy_state)
        # plotter.plot_wigner_2d(x, p, wigner)
        # plotter.plot_wigner_3d(x, p, wigner) # Uncomment if 3D is strictly required

        # 3. Marginal Analysis
        print("Comparing Ideal vs Lossy Marginals...")
        # Get ideal samples
        sx, sp = sim.get_ideal_samples(shots=4000)
        # Get lossy curves
        quad, mx, mp = sim.calculate_marginals(lossy_state)

        plotter.plot_marginal_comparison(sx, sp, quad, mx, mp, SCALE)
    return (run_simulation_analysis,)


@app.cell
def _(run_simulation_analysis):
    run_simulation_analysis()
    return


if __name__ == "__main__":
    app.run()
