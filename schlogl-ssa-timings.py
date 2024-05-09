import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import os
import sys
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm


def define_system():

    S = ["S"]
    emptySet = []

    c1 = 3e-7
    c2 = 1e-4
    c3 = 1e-3
    c4 = 3.5

    N1 = 1e5
    N2 = 2e5

    c1N12 = c1 * N1 / 2
    c26 = c2 / 6
    c3N2 = c3 * N2

    r1 = cu.Reaction(2 * S, 3 * S, c1)
    r2 = cu.Reaction(3 * S, 2 * S, c2)
    r3 = cu.Reaction(emptySet, S, c3)
    r4 = cu.Reaction(S, emptySet, c4)
    rs = cu.ReactionSet(r1, r2, r3, r4)

    V = rs.stochioimetric_matrix()
    return rs, V, (c1, c2, c3, c4, N1, N2, c1N12, c26, c3N2)


@jit(nopython=True)
def prop_fun(props: np.array, X: np.array, css: np.array):
    c1, c2, c3, c4, N1, N2, c1N12, c26, c3N2 = css

    S = X[0]
    props[0] = c1N12 * S * (S - 1)
    props[1] = c26 * S * (S - 1) * (S - 2)
    props[2] = c3N2
    props[3] = c4 * S


@jit(nopython=True)
def simulate_schlogl_ssa_final_time(T: float, V: np.array, css: np.array):
    props = np.zeros((4,))
    X = np.array([250.0])
    t = 0.0

    while t < T:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)

    return X


def simulate_schlogl_ssa(T: float, V: np.array, css: np.array):
    props = np.zeros((4,))
    X = np.array([250.0])
    t = 0.0

    X_hist = []
    t_hist = []

    i = 0
    while t < T:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


def simulate_schlogl_ssa_fixed_times(
    T: float, V: np.array, css: np.array, save_at_ts: np.array
):
    props = np.zeros((4,))
    X = np.array([250.0])
    t = 0.0

    X_hist = np.zeros((save_at_ts.size,))

    i = 0
    while t < T:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)

        while i < save_at_ts.size and save_at_ts[i] <= t:
            X_hist[i] = X
            i += 1
            print(i)

    return X_hist, save_at_ts


def time_paths(Npaths: int, T: float, css: np.array, V : np.array):
    low_times = np.zeros((Npaths,))
    high_times = np.zeros((Npaths,))

    simulate_schlogl_ssa_final_time(T, V, css)

    i, j = 0, 0
    while i < Npaths or j < Npaths:
        t0 = time.time()
        X = simulate_schlogl_ssa_final_time(T, V, css)
        t1 = time.time()
        if X < 300 and i < Npaths:
            low_times[i] = t1 - t0
            i += 1
        elif j < Npaths:
            high_times[j] = t1 - t0
            j += 1

    return low_times, high_times


def simulate_Npaths(Npaths: int, T: float, V: np.array, css: np.array):
    X_paths = []
    t_paths = []

    for _ in range(Npaths):
        Xpath, tpath = simulate_schlogl_ssa(T, V, css)
        X_paths.append(Xpath)
        t_paths.append(tpath)

    return X_paths, t_paths


def generate_npath_fig(show=True):

    plt.rcParams.update({"font.size": 14})

    npaths = 10
    X_paths, t_paths = simulate_Npaths(npaths, 50, V, css)
    for i in range(npaths):
        plt.plot(t_paths[i], X_paths[i])
    plt.xlabel("$t$")
    plt.tight_layout()
    plt.savefig("schlogl-paths.pdf", format="pdf")
    if show:
        plt.show()


# TODO a routine for simulating a lot of paths at time $T$ for different choices of time steps and ISS -- hopefully not too long
# new interval sets : (35,45), (45, 170), (200,210),(200,300)
def generate_Npaths_final_time(N: int, T: float, V: np.array, css: tuple):

    dat = np.zeros((N,1), dtype=np.short)
    for n in tqdm(range(N)):
        dat[n] = simulate_schlogl_ssa_final_time(T, V, css)

    return dat


def xp1():
    T = 50
    rs, V, css = define_system()
    Npaths = 500
    low, high = time_paths(Npaths, T, css, V)

    fname=f"dat/schlogl/times-ssa"
    np.save(fname+"-low", low)
    np.save(fname+"-high", high)



# TODO finish
def xp2():
    T = 1e6
    rs, V, css = define_system()
    Npaths = 1


def xp3():
    generate_npath_fig()


def xp4(gen_dat=True):

    fname = "dat/final-times-ssa.npy"

    rs, V, css = define_system()
    T = 50
    N = int(1e6)
    dat = (
        generate_Npaths_final_time(N, T, V, css)
        if gen_dat
        else np.load(fname)
    )
    np.save(fname, dat) if gen_dat else None

    return dat


if __name__ == "__main__":
    xp1()
    #dat = xp4(gen_dat=True)
