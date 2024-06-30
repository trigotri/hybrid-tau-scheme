import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import matplotlib.pyplot as plt
from numba import jit
import tqdm

def hcle_params():

    ISs = ((5, 10), (10, 15), (15, 20))
    TSSs = ((2e-1, 1.5e-1), (1, 0.4))

    return ISs, TSSs


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

    DNA, DNA0, P, mRNA = X
    props[0] = c1 * DNA
    props[1] = c2 * mRNA
    props[2] = c3 * DNA * P
    props[3] = c4 * DNA0
    props[4] = c5 * mRNA
    props[5] = c6 * P


@jit(nopython=True)
def fbeta(x: float, I1: float, I2: float):
    if x <= I1:
        return 1.0
    elif x >= I2:
        return 0.0
    else:
        return (I2 - x) / (I2 - I1)


@jit(nopython=True)
def oneMinFBeta(x: float, I1: float, I2: float):
    return 1.0 - fbeta(x, I1, I2)


@jit(nopython=True)
def compute_betas(betas: np.array, X: np.array, I1: float, I2: float):

    DNA, DNA0, P, mRNA = X
    DNAb = oneMinFBeta(DNA, I1, I2)
    DNA0b = oneMinFBeta(DNA0, I1, I2)
    Pb = oneMinFBeta(P, I1, I2)
    mRNAb = oneMinFBeta(mRNA, I1, I2)

    betas[0] = 1.0 - DNAb * mRNAb
    betas[1] = 1.0 - mRNAb * Pb
    betas[2] = 1.0 - DNAb * Pb * DNA0b
    betas[3] = betas[2]  # same reaction in reverse
    betas[4] = 1.0 - mRNAb
    betas[5] = 1.0 - Pb


@jit(nopython=True)
def simulate_ags_hcle_final_time(
    T: float, V: np.array, css: np.array, IS: tuple[float], TSS: tuple[float]
):

    props = np.zeros((6,))
    props_Xrint = np.zeros((6,))
    betas = np.zeros((6,))
    X = np.array([1.0, 0.0, 0.0, 0.0])
    t = 0.0
    while t < T:
        prop_fun(props, X, css)
        prop_fun(props_Xrint, np.rint(X), css)
        compute_betas(betas, X, IS[0], IS[1])
        X, t = cs.hybrid1_cle_step_numba(X, t, V, props, props_Xrint, betas, TSS[0], TSS[1])

    return X


def simulate_ags_hcle(T: float, V: np.array, css: np.array, IS: tuple, TSS: tuple):
    props = np.zeros((6,))
    props_Xrint = np.zeros((6,))
    betas = np.zeros((6,))
    X = np.array([1.0, 0.0, 0.0, 0.0])
    t = 0.0

    X_hist = []
    t_hist = []

    while t < T:
        prop_fun(props, X, css)
        prop_fun(props_Xrint, np.rint(X), css)
        compute_betas(betas, X, IS[0], IS[1])
        X, t = cs.hybrid1_cle_step_numba(X, t, V, props, props_Xrint, betas, TSS[0], TSS[1])
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


def time_paths(
    Npaths: int, T: float, css: np.array, V: np.array, IS: tuple, TSS: tuple
):

    times = np.zeros((Npaths,))
    simulate_ags_hcle_final_time(T, V, css, IS, TSS)

    for i in tqdm.tqdm(range(Npaths)):
        t0 = time.time()
        simulate_ags_hcle_final_time(T, V, css, IS, TSS)
        t1 = time.time()

        times[i] = t1 - t0

    return times


def xp1():

    ISs, TSSs = hcle_params()

    T = 1e3
    rs, V, css = define_system()
    Npaths = 10000

    for i, IS in enumerate(ISs):
        for j, TSS in enumerate(TSSs):
            times = time_paths(Npaths, T, css, V, IS, TSS)
            np.save(f"dat/ags/ags-hcle-times-IS{i}-TS{j}", times)


def xp2():

    ISs, TSSs = hcle_params()

    T = 1e3
    rs, V, css = define_system()
    for i, IS in enumerate(ISs):
        for j, TSS in enumerate(TSSs):

            X_hist, t_hist = simulate_ags_hcle(T, V, css, IS, TSS)

            Xhist = np.array(X_hist)
            plt.plot(t_hist, Xhist)
            plt.show()



def xp3():

    ISs, TSSs = hcle_params()

    T = 1e3
    rs, V, css = define_system()
    Npaths = 1000

    simulate_ags_hcle_final_time(T, V, css, ISs[0], TSSs[0])

    for i, IS in enumerate(ISs):
        for j, TSS in enumerate(TSSs):

            dat = np.zeros((Npaths, 4))
            for k in tqdm.tqdm(range(Npaths)):
                dat[k] = simulate_ags_hcle_final_time(T, V, css, IS, TSS)

            np.save(f"dat/ags/ags-hcle-states-IS{i}-TS{j}", dat)


if __name__ == "__main__":
    rs,V,css = define_system()
    xp1()
    #xp2()
    xp3()
