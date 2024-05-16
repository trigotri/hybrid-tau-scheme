import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import matplotlib.pyplot as plt
from numba import jit
import tqdm


def define_system():

    DNA, mRNA, P, DNA0, emptySet = ["DNA"], ["mRNA"], ["P"], ["DNA0"], []

    V = 1e2
    g1, g2, g3, g4, g5, g6 = 1e-2, 0.5, 0.1, 1e-2, 5e-3, 0.2

    c1 = V * g1
    c2 = g2
    c3 = g3 / V
    c4 = g4
    c5 = g5
    c6 = g6

    r1 = cu.Reaction(DNA, DNA + mRNA, c1)
    r2 = cu.Reaction(mRNA, mRNA + P, c2)
    r3 = cu.Reaction(DNA + P, DNA0, c3)
    r4 = cu.Reaction(DNA0, DNA + P, c4)
    r5 = cu.Reaction(mRNA, emptySet, c5)
    r6 = cu.Reaction(P, emptySet, c6)

    rs = cu.ReactionSet(r1, r2, r3, r4, r5, r6)
    V = rs.stochioimetric_matrix()

    return rs, V, (c1, c2, c3, c4, c5, c6)


@jit(nopython=True)
def prop_fun(props: np.array, X: np.array, css: np.array):
    c1, c2, c3, c4, c5, c6 = css

    DNA, DNA0, P, mRNA= X
    props[0] = c1 * DNA
    props[1] = c2 * mRNA
    props[2] = c3 * DNA * P
    props[3] = c4 * DNA0
    props[4] = c5 * mRNA
    props[5] = c6 * P


@jit(nopython=True)
def simulate_ags_ssa_final_time(T: float, V: np.array, css: np.array):
    props = np.zeros((6,))
    X = np.array([1.0, 0.0, 0.0, 0.0])
    t = 0.0
    while t < T:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)

    return X


def simulate_ags_ssa(T: float, V: np.array, css: np.array):
    props = np.zeros((6,))
    X = np.array([1.0, 0.0, 0.0, 0.0])
    t = 0.0

    X_hist = []
    t_hist = []

    while t < T:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


def simulate_ags_ssa_fixed_times(
    T: float, V: np.array, css: np.array, save_at_ts: np.array
):
    props = np.zeros((6,))
    X = np.array([1.0, 0.0, 0.0, 0.0])
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

    times = np.zeros((Npaths,))
    simulate_ags_ssa_final_time(T, V, css)

    for i in tqdm.tqdm(range(Npaths)):
        t0 = time.time()
        simulate_ags_ssa_final_time(T, V, css)
        t1 = time.time()

        times[i] = t1 - t0

    return times

def xp1():
    T = 1e3
    rs, V, css = define_system()
    Npaths = 100
    times = time_paths(Npaths, T, css, V)

    np.save("dat/ags-ssa-times", times)

def xp2(show=False):

    plt.rcParams.update({"font.size": 14})

    T = 1e3
    rs, V, css = define_system()
    X_hist, t_hist = simulate_ags_ssa(T, V, css)

    plt.figure(figsize=(10,6))
    plt.fill_between(t_hist, 5, 10, alpha=.4, label='IS 1')
    plt.fill_between(t_hist, 10, 15, alpha=.4, label='IS 2')
    plt.fill_between(t_hist, 15, 20,  alpha=.4, label='IS 3')

    plt.plot(t_hist, X_hist, label=["DNA", "DNA0", "P", "mRNA"])
    plt.grid()
    plt.legend()
    plt.xlabel("$t$")
    plt.tight_layout()
    plt.savefig("dat/ags-IS-zones.pdf")
    plt.show() if show else None


def xp3():

    T = 1e3
    rs, V, css = define_system()
    Npaths = 1000

    simulate_ags_ssa_final_time(T, V, css)

    dat = np.zeros((Npaths, 4))
    for k in tqdm.tqdm(range(Npaths)):
        dat[k] = simulate_ags_ssa_final_time(T, V, css)

    np.save(f"dat/ags/ags-ssa-states", dat)


if __name__ == "__main__":
    # rs, V, css = define_system() 
    # xp1()
    xp3()
