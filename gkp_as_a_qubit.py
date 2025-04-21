import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # GKP States

        ## GKP States as Qubits

        GKP states can be understood as superpositions of multiple squeezed states. They are examples of non-Gaussian states.

        GKP states are used as a means for encoding a qubit in a photonic mode - forming a GKP qubit. This presents us with a universal set of qubit gates and measurements that can be applied using already familiar Gaussian gates and measurements.
        """
    )
    return


@app.cell
def _():
    # Import relevant libraries
    import strawberryfields as sf
    from strawberryfields import Program, Engine, ops
    from strawberryfields.backends import BaseBosonicState

    import numpy as np
    from numpy import ndarray

    import matplotlib.pyplot as plt
    from matplotlib import colors, colorbar
    return (
        BaseBosonicState,
        Engine,
        Program,
        colorbar,
        colors,
        ndarray,
        np,
        ops,
        plt,
        sf,
    )


@app.cell
def _(np):
    def linear2db(value: float) -> tuple[float, str]:
        ans: float = (round(10 * np.log10(value)))
        return ans, f"({ans} dB)"
    return (linear2db,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ideal GKP Qubits

        For GKP encoding, the Wigner function of the idea GKP qubit $|0\rangle_{gkp}$ state consists of a linear combination of Dirac delta functions centered at half-integer multiples of $\sqrt{\pi\hbar}$ in phase space (ref: Encoding a qubit in an oscillator).

        $$
        W^{0}_{gkp} (q, p) = \sum^{\infty}_{s,t=-\infty} (-1)^{st} \delta(p - \frac{s\sqrt{\pi\hbar}}{2}) \delta(q - t\sqrt{\pi\hbar})
        $$

        The GKP qubit $|1\rangle_{gkp}$ state can be obtained by shifting the $|0\rangle_{gkp}$ state by $\sqrt{\pi\hbar}$ in the position qudrature.

        ### Visualisation of Ideal GKP qubit

        Visulisation of ideal GKP $|0\rangle_{gkp}$ and $|1\rangle_{gkp}$ states in phase space.
        """
    )
    return


@app.cell
def _(Engine, Program, np, ops, plt, sf):
    # Set the scale for phase space
    sf.hbar = 1
    scale: float = np.sqrt(sf.hbar * np.pi)

    ##############################
    # Create a GKP |0> state
    circuit: Program = Program(1)

    with circuit.context as q:
        ops.GKP() | q

    engine: Engine = Engine("bosonic")
    gkp_0 = engine.run(circuit).state

    ###############################
    # Create a GKP |1> state
    circuit: Program = Program(1)

    with circuit.context as q:
        ops.GKP() | q
        ops.Xgate(np.sqrt(np.pi * sf.hbar)) | q

    engine: Engine = Engine("bosonic")
    gkp_1 = engine.run(circuit).state

    # Get the phase space coordinates of the delta functions for the two states
    q_coords_0 = gkp_0.means().real[:, 0]
    p_coords_0 = gkp_0.means().real[:, 1]
    q_coords_1 = gkp_1.means().real[:, 0]
    p_coords_1 = gkp_1.means().real[:, 1]

    # Determine whether the delta functions are positively or negatively weighted
    delta_sign_0 = np.sign(gkp_0.weights().real)
    delta_sign_1 = np.sign(gkp_1.weights().real)

    # Plot the locations and signs of the deltas
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(q_coords_0 / scale,
               p_coords_0 / scale,
               c=delta_sign_0,
               cmap=plt.cm.RdBu, vmin=-1.5, vmax=1.5)
    ax[1].scatter(q_coords_1 / scale,
               p_coords_1 / scale,
               c=delta_sign_0,
               cmap=plt.cm.RdBu, vmin=-1.5, vmax=1.5)
    for i in range(2):
        ax[i].set_xlim(-3.5, 3.5)
        ax[i].set_ylim(-3.5, 3.5)
        ax[i].set_xlabel(r'$q$ (units of $\sqrt{\pi\hbar}$ )', fontsize=9)
        ax[i].set_aspect("equal")

    ax[0].set_title(r'$|0\rangle_{\rm gkp}$ Wigner function', fontsize=11)
    ax[1].set_title(r'$|1\rangle_{\rm gkp}$ Wigner function', fontsize=11)
    ax[0].set_ylabel(r'$p$ (units of $\sqrt{\pi\hbar}$ )', fontsize=9)
    fig.tight_layout()
    plt.show()
    return (
        ax,
        circuit,
        delta_sign_0,
        delta_sign_1,
        engine,
        fig,
        gkp_0,
        gkp_1,
        i,
        p_coords_0,
        p_coords_1,
        q,
        q_coords_0,
        q_coords_1,
        scale,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Finite-energy GKP states

        For the finite-energy GKP states, where their WIgner functions are expressed as linear combinations of Gaussian function; the Gaussian peaks, the variance, the location shift and the damping of the weights all depend on the finite-energy parameter $\epsilon$.

        The weights govern how large the Wigner function gets.

        Visualising how the Wigner function changes as we cary $\epsilon$:
        """
    )
    return


@app.cell
def _(ndarray, np, scale):
    # Choose some values of epsilon
    epsilons: list = [0.0631, 0.1, 0.21, 0.31, 0.4]

    # Pick a region of phase space to plot
    quad: ndarray = np.linspace(-3.5, 3.5, 200) * scale

    wigners = []
    return epsilons, quad, wigners


@app.cell
def _(BaseBosonicState, Engine, Program, epsilons, ops, quad, wigners):
    for epsilon in epsilons:

        # Create a GKP |0> state
        circ: Program = Program(1)

        with circ.context as qq:
            ops.GKP(epsilon=epsilon) | qq

            eng: Engine = Engine("bosonic")
            gkp: BaseBosonicState = eng.run(circ).state

            # Calculate the Wigner function
            wigner = gkp.wigner(mode=0, xvec=quad, pvec=quad)
            wigners.append(wigner)
    return circ, eng, epsilon, gkp, qq, wigner


@app.cell
def _(colorbar, colors, epsilons, linear2db, np, plt, quad, scale, wigners):
    # Plot the results
    fig2, axs = plt.subplots(1, 6, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.05]})
    cmap = plt.cm.RdBu
    cmax = np.real_if_close(np.amax(np.array(wigners)))
    norm = colors.Normalize(vmin=-cmax, vmax=cmax)
    cb1 = colorbar.ColorbarBase(axs[5], cmap=cmap, norm=norm, orientation="vertical")
    for k in range(5):
        axs[k].contourf(
            quad / scale,
            quad / scale,
            wigners[k],
            levels=60,
            cmap=plt.cm.RdBu,
            vmin=-cmax,
            vmax=cmax,
        )
        axs[k].set_title(r'$\epsilon =$'+str(epsilons[k])+linear2db(epsilons[k])[1], fontsize=11)
        axs[k].set_xlabel(r'$q$ (units of $\sqrt{\pi\hbar}$ )', fontsize=9)

    axs[0].set_ylabel(r'$p$ (units of $\sqrt{\pi\hbar}$ )', fontsize=9)
    cb1.set_label("Wigner function", fontsize=9)
    fig2.tight_layout()
    plt.show()
    return axs, cb1, cmap, cmax, fig2, k, norm


@app.cell
def _(mo):
    mo.md(
        r"""
        As $\epsilon$ increases, the variance of each peak increases, the peaks get closer and closer to the origin, and the weights drop exponentially away from the origin.

        In the limit as $\epsilon \rightarrow \infty$, the Fock damping is so strong that we essentially get a vacuum state.
        """
    )
    return


if __name__ == "__main__":
    app.run()
