import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import matplotlib.pyplot as plt
from numba import jit
import tqdm
from scipy.special import factorial


def define_system(K: int, c1: float):

    S, emptySet = ["S"], []

    r1 = cu.Reaction(emptySet, S, c1)
    r2 = cu.Reaction(S, emptySet, 1.0)

    rs = cu.ReactionSet(r1, r2)
    V = rs.stochioimetric_matrix()

    return rs, V, (K, c1)


@jit(nopython=True)
def prop_fun(props: np.array, X: np.array, css: tuple):
    K, c1 = css

    S = X[0]
    props[0] = c1 if S < K else 0.0
    props[1] = S


@jit(nopython=True)
def fbeta(x: float, I1: float, I2: float):
    if x <= I1:
        return 1.0
    elif x >= I2:
        return 0.0
    else:
        return (I2 - x) / (I2 - I1)


@jit(nopython=True)
def compute_betas(betas: np.array, X: np.array, I1: float, I2: float):
    S = X[0]
    betas[0] = fbeta(S, I1, I2)
    betas[1] = betas[0]


def simulate_bd_ht(
    T: float,
    V: np.array,
    css: tuple,
    I1: float,
    I2: float,
    delta_t: float,
    Delta_t: float,
):

    props = np.zeros((2,))
    betas = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    X_hist = []
    t_hist = []

    while t < T:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X, t = cs.hybrid1_tau_step_numba(X, t, V, props, betas, delta_t, Delta_t)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


def simulate_bd_ssa(T: float, V: np.array, css: tuple):
    props = np.zeros((2,))
    K, c1 = css

    X = np.array([K], dtype=float)
    t = 0.0

    X_hist = []
    t_hist = []

    while t < T:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


def simulate_extinction_time_ssa_path(V: np.array, css: tuple):
    props = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    X_hist = []
    t_hist = []

    while X[0] > 0.0:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


def simulate_extinction_time_ht_path(
    V: np.array, css: tuple, I1: float, I2: float, delta_t: float, Delta_t: float
):

    props = np.zeros((2,))
    betas = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    X_hist = []
    t_hist = []

    while X[0] > 0.0:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X, t = cs.hybrid1_tau_step_numba(X, t, V, props, betas, delta_t, Delta_t)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


def simulate_extinction_time_hcle_path(
    V: np.array, css: tuple, I1: float, I2: float, delta_t: float, Delta_t: float
):

    props = np.zeros((2,))
    betas = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    X_hist = []
    t_hist = []

    while X[0] > 0.0:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X, t = cs.hybrid1_cle_step_numba(X, t, V, props, betas, delta_t, Delta_t)
        X_hist.append(X)
        t_hist.append(t)

    return X_hist, t_hist


@jit(nopython=True)
def simulate_extinction_time_ssa(V: np.array, css: tuple):
    props = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    while X[0] > 0:
        prop_fun(props, X, css)
        X, t = cs.ssa_prop_step(X, t, V, props)

    return t


@jit(nopython=True)
def simulate_extinction_time_tau(V: np.array, css: tuple, delta_t: float):
    props = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    while X[0] > 0:
        prop_fun(props, X, css)
        X, t = cs.tau_leap_step_prop(X, t, V, props, delta_t)

    return t


@jit(nopython=True)
def simulate_extinction_time_cle(V: np.array, css: tuple, delta_t: float):
    props = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    while X[0] > 0:
        prop_fun(props, X, css)
        X, t = cs.cle_step_numba(X, t, V, props, delta_t)

    return t


@jit(nopython=True)
def simulate_extinction_time_ht(
    V: np.array, css: tuple, I1: float, I2: float, delta_t: float, Delta_t: float
):

    props = np.zeros((2,))
    betas = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    i = 0
    while X[0] > 0.0:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X, t = cs.hybrid1_tau_step_numba(X, t, V, props, betas, delta_t, Delta_t)

    return t


@jit(nopython=True)
def simulate_extinction_time_hcle(
    V: np.array, css: tuple, I1: float, I2: float, delta_t: float, Delta_t: float
):

    props = np.zeros((2,))
    betas = np.zeros((2,))
    K, c1 = css

    X = np.array([float(c1)])
    t = 0.0

    i = 0
    while X[0] > 0.0:
        prop_fun(props, X, css)
        compute_betas(betas, X, I1, I2)
        X, t = cs.hybrid1_cle_step_numba(X, t, V, props, betas, delta_t, Delta_t)

    return t


def compute_extinction_times_npaths_ssa(npaths: int, css: np.array, V: np.array):

    extinction_times = np.zeros((npaths,))
    simulate_extinction_time_ssa(V, css)

    t0 = time.time()
    for i in tqdm.tqdm(range(npaths)):
        extinction_times[i] = simulate_extinction_time_ssa(V, css)
    t1 = time.time()

    return extinction_times, t1 - t0


def compute_extinction_times_npaths_tau(
    npaths: int, css: np.array, V: np.array, delta_t: float
):

    extinction_times = np.zeros((npaths,))
    simulate_extinction_time_tau(V, css, delta_t)

    t0 = time.time()
    for i in tqdm.tqdm(range(npaths)):
        extinction_times[i] = simulate_extinction_time_tau(V, css, delta_t)
    t1 = time.time()

    return extinction_times, t1 - t0


def compute_extinction_times_npaths_cle(
    npaths: int, css: np.array, V: np.array, delta_t: float
):

    extinction_times = np.zeros((npaths,))
    simulate_extinction_time_cle(V, css, delta_t)

    t0 = time.time()
    for i in tqdm.tqdm(range(npaths)):
        extinction_times[i] = simulate_extinction_time_cle(V, css, delta_t)
    t1 = time.time()

    return extinction_times, t1 - t0


def compute_extinction_times_npaths_htau(
    npaths: int,
    css: np.array,
    V: np.array,
    I1: float,
    I2: float,
    delta_t: float,
    Delta_t: float,
):

    extinction_times = np.zeros((npaths,))
    simulate_extinction_time_ht(V, css, I1, I2, delta_t, Delta_t)

    t0 = time.time()
    for i in tqdm.tqdm(range(npaths)):
        extinction_times[i] = simulate_extinction_time_ht(
            V, css, I1, I2, delta_t, Delta_t
        )
    t1 = time.time()

    return extinction_times, t1 - t0


def compute_extinction_times_npaths_hcle(
    npaths: int,
    css: np.array,
    V: np.array,
    I1: float,
    I2: float,
    delta_t: float,
    Delta_t: float,
):

    extinction_times = np.zeros((npaths,))
    simulate_extinction_time_hcle(V, css, I1, I2, delta_t, Delta_t)

    t0 = time.time()
    for i in tqdm.tqdm(range(npaths)):
        extinction_times[i] = simulate_extinction_time_hcle(
            V, css, I1, I2, delta_t, Delta_t
        )
    t1 = time.time()

    return extinction_times, t1 - t0


def xp1():
    K = 50
    npaths = 1000
    c1s = np.arange(4, 10)

    delta_t, Delta_t = 5e-2, 5e-2
    Is = [(2, 4), (4, 6), (6, 8)]

    METs_ssa = np.zeros_like(c1s)
    METs_ht = np.zeros((len(c1s), len(Is)))
    for i, c1 in enumerate(c1s):
        rs, V, css = define_system(K, c1)
        ET = compute_extinction_times_npaths_ssa(npaths, css, V)
        METs_ssa[i] = ET.mean()

        for j, I in enumerate(Is):
            I1, I2 = I
            ET = compute_extinction_times_npaths_htau(
                npaths, css, V, I1, I2, delta_t, Delta_t
            )
            METs_ht[i, j] = ET.mean()

    plt.plot(c1s, METs_ssa, label="SSA")
    plt.plot(c1s, METs_ht, label=[str(I) for I in Is])
    plt.legend()
    plt.grid()
    plt.xlabel("$c_1$")
    plt.ylabel("MET")
    plt.show()

    np.save("dat/met-ssa-times", METs_ssa)
    np.save("dat/met-ht-times", METs_ht)


def xp2(show=False):

    plt.rcParams.update({"font.size": 16})

    T = 1e3
    rs, V, css = define_system()
    X_hist, t_hist = simulate_met_ssa(T, V, css)

    plt.figure(figsize=(10, 6))
    plt.fill_between(t_hist, 5, 10, alpha=0.4, label="IS 1")
    plt.fill_between(t_hist, 10, 15, alpha=0.4, label="IS 2")
    plt.fill_between(t_hist, 15, 20, alpha=0.4, label="IS 3")

    plt.plot(t_hist, X_hist, label=["DNA", "DNA0", "P", "mRNA"])
    plt.grid()
    plt.legend()
    plt.xlabel("$t$")
    plt.tight_layout()
    plt.savefig("dat/met-IS-zones.pdf")
    plt.show() if show else None


# not working as of yet
def MET_CME(n: int, K: int, c1: float):
    met = 0
    for m in range(1, n + 1):
        for j in range(K - m + 2):
            met += factorial(m - 1) / (factorial(j + m - 1) * c1**j)

    return met


if __name__ == "__main__":
    K = 50
    npaths = 100000

    delta_t, Delta_t = 1e-1, 1e-1
    c1 = 10
    I1 = 5
    I2 = 7

    compute_data = False

    names_algs = ["SSA", "H-tau", "Tau-L", "H-CLE", "CLE"]
    labels_algs = ["SSA", "H $\\tau$", "$\\tau$-leap", "H CLE", "CLE"]

    rs, V, css = define_system(K, c1)
    if compute_data:
        ET_ssa, t_ssa = compute_extinction_times_npaths_ssa(npaths, css, V)
        ET_tau, t_tau = compute_extinction_times_npaths_tau(npaths, css, V, delta_t)
        ET_ht, t_ht = compute_extinction_times_npaths_htau(
            npaths, css, V, I1, I2, delta_t, Delta_t
        )
        ET_hcle, t_hcle = compute_extinction_times_npaths_hcle(
            npaths, css, V, I1, I2, delta_t, Delta_t
        )
        ET_cle, t_cle = compute_extinction_times_npaths_cle(npaths, css, V, delta_t)

        for ET, t, name in zip(
            [ET_ssa, ET_ht, ET_tau, ET_hcle, ET_cle],
            [t_ssa, t_ht, t_tau, t_hcle, t_cle],
            names_algs,
        ):
            np.save(f"./dat/{name}-ET", ET)
            np.save(f"./dat/{name}-t", np.array([t]))

    ETs = []
    ts = []
    for name in names_algs:
        ETs.append(np.load(f"./dat/{name}-ET.npy"))
        ts.append(np.load(f"./dat/{name}-t.npy"))

    ET_ssa = ETs[0]
    ET_tau = ETs[2]
    ET_ht = ETs[1]
    ET_hcle = ETs[3]
    ET_cle = ETs[4]

    t_ssa = ts[0][0]
    t_tau = ts[2][0]
    t_ht = ts[1][0]
    t_hcle = ts[3][0]
    t_cle = ts[4][0]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.rcParams.update({"font.size" : 16})

    plt.figure()
    plt.hist(
        np.log(ET_ssa),
        bins=1000,
        histtype="step",
        range=[-np.log(10), 4.7 * np.log(10)],
        density=True,
        cumulative=True,
    )
    plt.hist(
        np.log(ET_ht),
        bins=1000,
        histtype="step",
        range=[-np.log(10), 4.7 * np.log(10)],
        density=True,
        cumulative=True,
    )
    plt.hist(
        np.log(ET_tau),
        bins=1000,
        histtype="step",
        range=[-np.log(10), 4.7 * np.log(10)],
        density=True,
        cumulative=True,
    )
    plt.hist(
        np.log(ET_hcle),
        bins=1000,
        histtype="step",
        range=[-np.log(10), 4.7 * np.log(10)],
        density=True,
        cumulative=True,
    )
    plt.hist(
        np.log(ET_cle),
        bins=1000,
        histtype="step",
        range=[-np.log(10), 4.7 * np.log(10)],
        density=True,
        cumulative=True,
    )
    K = 5
    xticks = [k * np.log(10) for k in range(-1, K)]
    xticks_labels = ["$10^{" + str(k) + "}$" for k in range(-1, K)]
    plt.xticks(xticks, labels=xticks_labels)
    plt.xlabel("Extinction time CDF")
    plt.xlim([-1.01 * np.log(10), 4.69 * np.log(10)])
    plt.legend(labels_algs, loc="upper left")
    plt.tight_layout()
    plt.savefig("./dat/met-densities.pdf", format="pdf")
    # plt.show()

    # start time-scale at 10^{-1}, tau step --> done
    # for consistency, have added corrections to Fig 3 & 4
    # add CLE lines --> done, but not logical wrt H-CLE & SSA

    indxs = [k for k in range(0, npaths, 1)]
    plt.figure()
    plt.scatter(np.sort(ET_ssa), np.sort(ET_ht), label="H $\\tau$")
    plt.scatter(np.sort(ET_ssa), np.sort(ET_tau), label="$\\tau$-leap")
    plt.scatter(np.sort(ET_ssa), np.sort(ET_hcle), label="H CLE")
    plt.scatter(np.sort(ET_ssa), np.sort(ET_cle), label="CLE")
    plt.plot(np.sort(ET_ssa), np.sort(ET_ssa), label="SSA")
    #plt.title("QQ-plot")
    plt.xlabel("SSA empirical quantiles")
    plt.ylabel("Empirical quantiles")
    #plt.xticks(range(5000, 30001, 5000), labels=[f"${k}$" for k in range(5, 31, 5)])
    plt.legend()
    plt.tight_layout()
    plt.savefig("./dat/met-qq.png", format="png")
    # plt.show()

    if True:
        print("Alg.\tMean\t\tStd\t\tAvg sim. time\tRatio SSA")
        for ET, t, na in zip(
            [ET_ssa, ET_ht, ET_tau, ET_hcle, ET_cle],
            [t_ssa, t_ht, t_tau, t_hcle, t_cle],
            ["SSA", "H-tau", "Tau-L", "H-CLE", "CLE"],
        ):
            print(f"{na}\t{ET.mean():5e}\t{ET.std():5e}\t{t/npaths:5e}\t{t_ssa/t:<5g}")
