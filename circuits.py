import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    # import strawberryfields as sf
    # from strawberryfields import ops

    # # create a 3-mode quantum program
    # prog = sf.Program(3)

    # with prog.context as q:
    #     ops.Sgate(0.54) | q[0]
    #     ops.Sgate(0.54) | q[1]
    #     ops.Sgate(0.54) | q[2]
    #     ops.BSgate(0.43, 0.1) | (q[0], q[2])
    #     ops.BSgate(0.43, 0.1) | (q[1], q[2])
    #     ops.MeasureFock() | q

    # # initialize the fock backend with a
    # # Fock cutoff dimension (truncation) of 5
    # eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    return


@app.cell
def _():
    import strawberryfields as sf
    from strawberryfields import ops

    # create a 2-mode quantum program
    prog = sf.Program(2)

    # create a free parameter named 'a'
    a = prog.params('a')

    # define the program
    with prog.context as q:
        ops.Dgate(a ** 2)    | q[0]  # free parameter
        ops.MeasureX         | q[0]  # measure qumode 0, the result is used in the next operation
        ops.Sgate(1 - sf.math.sin(q[0].par)) | q[1]  # measured parameter
        ops.MeasureFock()    | q[1]

    # initialize the Fock backend
    eng = sf.Engine('fock', backend_options={'cutoff_dim': 5})

    # run the program, with the free parameter 'a' bound to the value 0.9
    result = eng.run(prog, args={'a': 0.9})
    return a, eng, ops, prog, q, result, sf


@app.cell
def _(eng, prog):
    result = eng.run(prog)
    return (result,)


@app.cell
def _(result):
    result.samples
    return


@app.cell
def _(result):
    state = result.state
    state
    return (state,)


@app.cell
def _(state):
    state.trace()    # trace of the quantum state
    return


@app.cell
def _(state):
    state.dm().shape # density matrix
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
