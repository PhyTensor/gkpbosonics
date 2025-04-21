import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Measurements""")
    return


@app.cell
def _():
    import numpy as np

    # set the random seed
    np.random.seed(42)

    import strawberryfields as sf
    from strawberryfields import Engine, Program, Result
    from strawberryfields.ops import Fock, BSgate, MeasureFock
    return BSgate, Engine, Fock, MeasureFock, Program, Result, np, sf


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Consider a Two-Mode Circuit

        Construct two mode circuit, where two incident Fock states $|n\rangle$ and $|m\rangle$ are directed on a beamsplitter, with two photon detectors at the output modes.

        Beamsplitter preserves the photon number of the system; thus, two output states $|n^\prime\rangle$ and $|m^\prime\rangle$ must be such that $n+m=n^\prime+m^\prime$.

        """
    )
    return


@app.cell
def _(BSgate, Fock, MeasureFock, Program):
    def circuit_ops(modes: int) -> Program:
        prog: Program = Program(modes)

        with prog.context as q:
            Fock(2) | q[0]
            Fock(3) | q[1]
            BSgate() | (q[0], q[1]) # 50-50 beamsplitter theta=pi/4, phi=0
            MeasureFock() | q[0]

        return prog
    return (circuit_ops,)


@app.cell
def _(Engine):
    engine_fock: Engine = Engine("fock", backend_options={"cutoff_dim": 6})
    return (engine_fock,)


@app.cell
def _(Program, circuit_ops):
    circ: Program = circuit_ops(2)
    return (circ,)


@app.cell
def _(Result, circ, engine_fock):
    results: Result = engine_fock.run(circ)
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(results):
    # Measured mode gets reset to vacuum state.
    # Measured value of mode q[0] = photon number
    pn0 = results.samples
    pn0
    return (pn0,)


@app.cell
def _():
    return


@app.cell
def _(pn0):
    print(f"Mode 0 has photon number: {pn0[0][0]}")
    return


@app.cell
def _(MeasureFock, engine_fock, sf):
    # executing the backend again, to apply the second Fock measurement
    prog2 = sf.Program(2)
    with prog2.context as q:
        MeasureFock() | q[1]

    results2 = engine_fock.run(prog2)
    return prog2, q, results2


@app.cell
def _(results2):
    pn1 = results2.samples
    pn1
    return (pn1,)


@app.cell
def _(pn1):
    print(f"Mode 1 has photon number: {pn1[0][0]}")
    return


if __name__ == "__main__":
    app.run()
