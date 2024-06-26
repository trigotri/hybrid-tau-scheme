import numpy as np
import chem_utils as cu
import chem_step as cs
import time
from numba import jit
import matplotlib.pyplot as plt
import tqdm

# Produces files of format X{k}-{proc}.npy and corresponding time index
# using SSA for Lotka-Volterra system from paper.

## set reactions
def define_system():
    A, B = ["A"], ["B"]
    k1, k2, k3 = 2.0, 2e-3, 2.0

    r1 = cu.Reaction(A, 2 * A, k1)
    r2 = cu.Reaction(A + B, 2 * B, k2)
    r3 = cu.Reaction(B, [], k3)
    rs = cu.ReactionSet(r1, r2, r3)

    V = rs.stochioimetric_matrix()

    return rs, V, (k1, k2, k3)


@jit(nopython=True)
def prop_fun(props: np.array, X: np.array, ks: np.array):
    k1, k2, k3 = ks
    A, B = X[0], X[1]
    props[0] = k1 * A
    props[1] = k2 * A * B
    props[2] = k3 * B


@jit(nopython=True)
def simulate_lv_cle(
    T: float, V: np.array, ks: np.array, save_at_ts: np.array, delta_t: float
):

    props = np.zeros((3,))
    X = np.array([50.0, 60.0])
    t = 0.0

    X_hist = np.zeros((save_at_ts.size, 2))

    i = 0
    while t < T:
        prop_fun(props, X, ks)
        X, t = cs.cle_step_nb_refl(X, t, V, props, delta_t)

        while save_at_ts[i] <= t and i < save_at_ts.size:
            X_hist[i] = X
            i += 1

    return X_hist, save_at_ts


def single_launch():
    rs, V, ks = define_system()
    T = 6.0
    save_at_ts = np.linspace(0, T, 1000)
    I1A, I2A, I1B, I2B = 5, 10, 5, 10
    delta_t, Delta_t = 1e-2, 1e-3
    X_hist, t_hist = simulate_lv_cle(T, V, ks, save_at_ts, delta_t)

    plt.plot(t_hist, X_hist, "-o")
    plt.show()


def record_times_cle(T : float, V : np.array, ks : np.array, save_at_ts : np.array, delta_t : float):

    props = np.zeros((3,))
    betas = np.zeros((3,))

    X = np.array([50.,60.])
    t = 0.

    prop_fun(props, X, ks)
    cs.cle_step_nb_refl(X, t, V, props, delta_t)

    times = np.zeros(save_at_ts.size)

    t0 = time.time()
    i = 0
    while t < T:
        prop_fun(props, X, ks)
        X,t = cs.cle_step_nb_refl(X, t, V, props, delta_t)

        while i < save_at_ts.size and save_at_ts[i] <= t :
           times[i] = time.time() - t0
           i+=1

    return times


def time_lv():

    N = int(1e4)
    rs,V,ks = define_system()
    T = 6.
    numsteps = 50

    save_at_ts = np.linspace(1,T,numsteps, endpoint=True)

    delta_t = 3e-2

    IS = (20,25)

    time_dat = np.zeros((N,save_at_ts.size))
    for i in tqdm.tqdm(range(N)):
        time_dat[i] = record_times_cle(T, V, ks, save_at_ts, delta_t)

    np.save(f'dat/cle-times', time_dat)
    np.save(f'dat/cle-save-times', save_at_ts)

    return time_dat



def MC_xp():
    N = 10000
    rs, V, ks = define_system()
    I1A, I2A, I1B, I2B = 5, 10, 5, 10
    delta_t = 1e-2
    T = 6.0
    numsteps = 7
    # save_at_ts = np.linspace(0,T,numsteps)
    save_at_ts = np.arange(numsteps)
    dat = np.zeros((N, numsteps, 2))
    for i in range(N):
        # if (i+1) % 1000 == 0:
        print(f"Iteration {i+1}")
        X_hist, t_hist = simulate_lv_cle(T, V, ks, save_at_ts, delta_t)

        dat[i] = X_hist

    return dat, save_at_ts


def save_MC_dat(dat: np.array):
    np.save(f"dat/cle-{dat.shape[0]}-{dat.shape[1]}", dat)


if __name__ == "__main__":
    # single_launch()
    #dat, save_at_ts = MC_xp()
    #save_MC_dat(dat)
    time_lv()
