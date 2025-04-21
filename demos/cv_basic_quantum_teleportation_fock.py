import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Quantum Teleportation

        Also called **state teleportation**.
        """
    )
    return


@app.cell
def _():
    import strawberryfields as sf
    from strawberryfields import Program, Engine, Result
    from strawberryfields.ops import Coherent, Squeezed, MeasureX, MeasureP, Zgate, Xgate, BSgate

    import numpy as np
    from numpy import pi, sqrt

    # set the random seed
    np.random.seed(42)
    return (
        BSgate,
        Coherent,
        Engine,
        MeasureP,
        MeasureX,
        Program,
        Result,
        Squeezed,
        Xgate,
        Zgate,
        np,
        pi,
        sf,
        sqrt,
    )


@app.cell
def _(np):
    alpha: complex = 1 + 0.5j
    r: float = np.abs(alpha)
    phi: float = np.angle(alpha)
    print(f"alpha: {alpha} \t r: {r} \t phi: {phi}")
    return alpha, phi, r


@app.cell
def _(
    BSgate,
    Coherent,
    MeasureP,
    MeasureX,
    Program,
    Squeezed,
    Xgate,
    Zgate,
    phi,
    pi,
    r,
    sqrt,
):
    def circuit_ops() -> Program:
        circuit: Program = Program(num_subsystems=3)
    
        with circuit.context as q:
            # Prepare initial states
            Coherent(r=r, phi=phi) | q[0]
            Squeezed(-2) | q[1]
            Squeezed(2) | q[2]
    
            # apply gates
            BS = BSgate(pi/4, pi)
            BS | (q[1], q[2])
            BS | (q[0], q[1])
    
            # Perform homodyne measurements
            MeasureX | q[0]
            MeasureP | q[1]
    
            # # displacement gates conditioned on the measurements
            Xgate(sqrt(2) * q[0].par) | q[2]
            Zgate(-sqrt(2) * q[1].par) | q[2]

        return circuit
    return (circuit_ops,)


@app.cell
def _(Engine, sf):
    eng: Engine = sf.Engine('fock', backend_options={"cutoff_dim": 15})
    return (eng,)


@app.cell
def _(Program, circuit_ops):
    circuit: Program = circuit_ops()
    return (circuit,)


@app.cell
def _(Result, circuit, eng):
    result: Result = eng.run(circuit, shots=1, modes=None, compile_options={})
    return (result,)


@app.cell
def _(result):
    print(result.samples)
    return


@app.cell
def _(result):
    print(result.state)
    state = result.state
    return (state,)


@app.cell
def _(state):
    print(state.dm().shape)
    return


@app.cell
def _(np, state):
    rho2 = np.einsum('kkllij->ij', state.dm())
    print(rho2.shape)
    return (rho2,)


@app.cell
def _(np, rho2):
    probs = np.real_if_close(np.diagonal(rho2))
    print(probs)
    return (probs,)


@app.cell
def _(probs):
    from matplotlib import pyplot as plt
    plt.bar(range(7), probs[:7])
    plt.xlabel('Fock state')
    plt.ylabel('Marginal probability')
    plt.title('Mode 2')
    plt.show()
    return (plt,)


@app.cell
def _(np, state):
    fock_probs = state.all_fock_probs()
    fock_probs.shape
    np.sum(fock_probs, axis=(0,1))
    return (fock_probs,)


@app.cell
def _(np, plt, state):
    fig = plt.figure()
    X = np.linspace(-5, 5, 100)
    P = np.linspace(-5, 5, 100)
    Z = state.wigner(0, X, P)
    X, P = np.meshgrid(X, P)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
    fig.set_size_inches(4.8, 5)
    ax.set_axis_off()
    plt.show()
    return P, X, Z, ax, fig


if __name__ == "__main__":
    app.run()
