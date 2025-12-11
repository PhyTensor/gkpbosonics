import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import numpy as np
    from numpy import ndarray

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm, colors, colorbar

    import strawberryfields as sf
    from strawberryfields import hbar, Program, ops, Engine, Result
    from strawberryfields.ops import LossChannel
    from strawberryfields.backends import BaseBosonicState
    return BaseBosonicState, Engine, Program, Result, np, sf


@app.cell
def _(np, sf):
    # Set the scale for phase space
    sf.hbar = 1
    scale = np.sqrt(sf.hbar * np.pi)
    return


@app.cell
def _(Program, sf):
    prog: Program = sf.Program(1)

    with prog.context as q:
        sf.ops.GKP([0, 0]) | q

    # prog.draw_circuit("./learn/", True)
    return (prog,)


@app.cell
def _(Engine, sf):
    eng: Engine = sf.Engine("bosonic")
    shots: int = 1
    return eng, shots


@app.cell
def _(Result, eng: "Engine", prog: "Program", shots: int):
    results: Result = eng.run(prog, shots=shots)
    return (results,)


@app.cell
def _(BaseBosonicState, results: "Result"):
    bosonic_state: BaseBosonicState = results.state
    return (bosonic_state,)


@app.cell
def _(bosonic_state: "BaseBosonicState"):
    bosonic_state.means()
    return


@app.cell
def _(bosonic_state: "BaseBosonicState"):
    bosonic_state.covs()
    return


@app.cell
def _(bosonic_state: "BaseBosonicState"):
    bosonic_state.weights()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
