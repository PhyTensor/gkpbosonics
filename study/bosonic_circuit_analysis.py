import numpy as np
import strawberryfields as sf
from strawberryfields import Engine, Program, Result
from strawberryfields.backends import BaseBosonicState
from strawberryfields.ops import LossChannel, GKP

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors, colorbar
from mpl_toolkits.mplot3d import Axes3D
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

# Set random seed
np.random.seed(42)

class BosonicCircuitAnalysis:
    def __init__(self):
        # Set the scale for phase space
        sf.hbar: float = 1.0
        self.scale: float = np.sqrt(sf.hbar * np.pi)

        # Rescale the outcome for Pauli Y
        self.y_scale = np.sqrt(2 * sf.hbar) / self.scale

        # simulation parameters
        self.epsilon = None
        self.quad_axis: np.ndarray = np.linspace(-5, 5, 400) * self.scale

        # define the simulation engine
        self.engine: Engine = Engine("bosonic")

        self.circuit: Program = None

        self.result: Result = None
        self.state: BaseBosonicState = None
        self.wigner_function: np.ndarray = None
        self.marginals: list = []
        self.p_marginals: np.ndarray = None
        self.qp_marginals: np.ndarray = None
        self.q_marginals: np.ndarray = None
        self.expectation_values: np.ndarray = np.zeros(3)

    def create_state(self, state: list, mode: int, epsilon: float) -> None:
        self.epsilon = epsilon
        self.circuit = Program(num_subsystems=mode)

        with self.circuit.context as q:
            GKP(state=state, epsilon=self.epsilon) | q

    def create_lossy_state(self, state: list, mode: int, epsilon: float, loss_parameter: float) -> None:
        self.epsilon = epsilon
        self.circuit = Program(num_subsystems=mode)

        with self.circuit.context as q:
            GKP(state=state, epsilon=self.epsilon) | q
            LossChannel(loss_parameter) | q

    def execute_circuit(self) -> None:
        self.result = self.engine.run(program=self.circuit)
        self.state = self.result.state

    def calculate_wigner_function(self, mode: int) -> None:
        """
        Calculate the discretized Wigner function of the specified mode.
        """
        self.wigner_function = self.state.wigner(mode, xvec=self.quad_axis, pvec=self.quad_axis)

    def calculate_marginals(self, mode: int) -> None:
        """
        Calculate the p, q-p, and p quadrature marginal distributions for a given circuit mode.
        These can be used to determine the Pauli X, Y, and Z outcomes of the GKP qubit.
        """
        # The rotation angle in phase space is determined by phi
        phis: list = [np.pi/2, -np.pi/4, 0]
        self.marginals = [self.state.marginal(mode, self.quad_axis, phi=phi) for phi in phis]
        self.p_marginals = self.marginals[0]
        self.qp_marginals = self.marginals[1]
        self.q_marginals = self.marginals[2] # phi = 0 -> position quadrature

    def calculate_pauli_expectation_values(self) -> None:
        for i in range(3):
            if i == 1:
                # Blue bins are weighted +1, red bins are weighted -1
                bin_weights = 2 * (((self.quad_axis * self.y_scale - 0.5) // 1) % 2) - 1
                integrand = (self.marginals[i] / self.y_scale) * bin_weights
                self.expectation_values[i] = np.trapezoid(integrand, self.quad_axis * self.y_scale)
            else:
                # Pauli X and Z
                # Blue bins are weighted +1, red bins are weighted -1
                bin_weights = 2 * (((self.quad_axis / self.scale - 0.5) // 1) % 2) - 1
                integrand = (self.marginals[i] * self.scale) * bin_weights
                self.expectation_values[i] = np.trapezoid(integrand, self.quad_axis / self.scale)

    def plot_marginal_distributions(self, savefile: str = None) -> None:
        """
        Plot the p, q-p, and q quadrature marginal distributions for a given circuit mode.
        """
        paulis: list = ["X", "Y", "Z"]
        homodynes: list = ["p", "q-p", "q"]

        fig, axs = plt.subplots(
                nrows=1,
                ncols=3,
                figsize=(6.6, 2.4),   # ~ PRL column width
                sharey=True
                )

        for i in range(3):
            if i == 1:
                axs[i].plot(self.quad_axis * self.y_scale, self.marginals[i] / self.y_scale, 'k-')
                axs[i].set_xlim(self.quad_axis[0] * self.y_scale, self.quad_axis[-1] * self.y_scale)
            else:
                axs[i].plot(self.quad_axis / self.scale, self.marginals[i] * self.scale, 'k-')
                axs[i].set_xlim(self.quad_axis[0] / self.scale, self.quad_axis[-1] / self.scale)

            # Color the qubit bins blue and red
            for j in range(-10, 10):
                axs[i].axvspan((2 * j - 0.5), (2 * j + 0.5), alpha=0.2, facecolor='b')
                axs[i].axvspan((2 * j + 0.5), (2 * j + 1.5), alpha=0.2, facecolor='r')

            axs[i].minorticks_on()

            axs[i].set_xlabel(homodynes[i] + r' ($\sqrt{\pi\hbar}$ )')

            axs[i].text(
                    0.02, 0.95,
                    rf"$\langle {paulis[i]} \rangle = {self.expectation_values[i]:.4f}$",
                    transform=axs[i].transAxes,
                    ha="left",
                    va="top",
                    fontsize=9
                    )

        axs[0].set_ylabel("Marginal Distribution")

        fig.align_ylabels()
        fig.tight_layout(w_pad=1.2)
        if savefile is not None:
            plt.savefig(fname=savefile)
        plt.show()

    def plot_wigner_function(self, savefile: str = None) -> None:
        X: np.ndarray = self.quad_axis
        P: np.ndarray = self.quad_axis
        Z: np.ndarray = self.wigner_function

        fig, ax = plt.subplots(figsize=(6, 5))

        color_scale: float = np.max(Z.real)
        nrm: mpl.colors.Normalize = mpl.colors.Normalize(-color_scale, color_scale)

        contour = plt.contourf(X, P, Z, levels=100, cmap=cm.RdBu, norm=nrm)

        ax.set_aspect("equal")
        ax.set_xlabel(r"$q$ ($\sqrt{\pi\hbar}$)", fontsize=12)
        ax.set_ylabel(r"$p$ ($\sqrt{\pi\hbar}$)", fontsize=12)
        # ax.set_title("Wigner Function", fontsize=13, pad=10)

        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(r"$W(q, p)$", rotation=270, labelpad=15)

        ax.text(
                0.02, 0.95,
                rf"$\epsilon = {self.epsilon}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9
                )

        plt.tight_layout()
        plt.show()

    def plot_3d_wigner_function(self, savefile: str = None) -> None:
        X: np.ndarray = self.quad_axis
        P: np.ndarray = self.quad_axis
        Z: np.ndarray = self.wigner_function

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Meshgrid Safety
        # Ensure inputs are 2D grids. If they are 1D axes, convert them.
        if X.ndim == 1 and P.ndim == 1:
            X, P = np.meshgrid(X, P)

        # 2. Plot Surface
        # cmap="RdBu_r": Red=Positive, Blue=Negative (Standard for Wigner)
        # linewidth=0, antialiased=False: Creates a smooth, solid surface
        surf = ax.plot_surface(
                X, P, Z,
                cmap=cm.RdBu_r,
                linewidth=0,
                antialiased=False,
                rstride=2, cstride=2,  # Optimization for rendering speed vs detail
                alpha=0.9
                )

        # 3. Perspective & Lighting
        # elev=35, azim=-45 provides a clear view of both peaks and negative dips
        ax.view_init(elev=35, azim=-45)

        # 4. Styling
        ax.set_xlabel(r"$q$ ($\sqrt{\pi\hbar}$)", fontsize=11)
        ax.set_ylabel(r"$p$ ($\sqrt{\pi\hbar}$)", fontsize=11)
        ax.set_zlabel(r"$W(q, p)$", fontsize=11)

        # 5. Colorbar
        # Shrink prevents the bar from dominating the plot
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        if savefile is not None:
            plt.savefig(savefile)
        plt.show()

