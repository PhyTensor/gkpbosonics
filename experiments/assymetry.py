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
    from numpy import pi, sqrt, ndarray

    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colorbar, colors
    import matplotlib.pyplot as plt

    import strawberryfields as sf
    from strawberryfields import Engine, Program, Result
    from strawberryfields.backends import BaseBosonicState
    from strawberryfields.ops import (GKP, Sgate, BSgate, Coherent, LossChannel,
                                      MeasureP, MeasureX, Squeezed, Xgate,
                                      Zgate, Dgate, S2gate)

    # set the random seed
    np.random.seed(42)
    return cm, mpl, np, pi, plt, sf


@app.cell
def _(np, sf):
    # Calculate the quadrature distributions
    scale1 = np.sqrt(sf.hbar)
    quad_axis= np.linspace(-4, 4, 256) * scale1
    # cat_prob_x = cat.marginal(0, quad_axis)  # This is the q quadrature
    # cat_prob_p = cat.marginal(0, quad_axis, phi=np.pi / 2)  # This is the p quadrature
    return (quad_axis,)


@app.cell
def _(cm, mpl, np, pi, plt, quad_axis, sf):
    # Create GKP state
    epsilon = 0.1
    prog_gkp = sf.Program(1)
    with prog_gkp.context as q:
        sf.ops.GKP(epsilon=epsilon) | q
        sf.ops.Sgate(1.05, 0.25*pi) | q
        sf.ops.Rgate(-0.125*pi) | q
        sf.ops.LossChannel(0.85) | q

    eng = sf.Engine("bosonic")
    gkp = eng.run(prog_gkp, shots=1).state

    Wgkp = gkp.wigner(mode=0, xvec=quad_axis, pvec=quad_axis)
    scale = np.max(Wgkp.real)
    nrm = mpl.colors.Normalize(-scale, scale)
    plt.axes().set_aspect("equal")
    plt.contourf(quad_axis, quad_axis, Wgkp, 60, cmap=cm.RdBu, norm=nrm)
    plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=15)
    plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=15)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
