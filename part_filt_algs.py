import numpy as np
import chem_step as cs
import part_filt_utils as pfu
from numba import jit

rng = np.random.default_rng()






#@jit(nopython=True)
def run_pf_ssa(N : int, V : np.array, logc : np.array, noisy_data : np.array) -> float:
    J=50

    c1, c2, c3, c4, c5 = np.exp(logc)

    def prop_func(X : np.array) -> np.ndarray:
        return np.array([c1, c2, c3*X[0], c4*X[1], c5*X[0]*X[1]])

    particle_states = np.zeros((N,2))
    particle_times = np.zeros((N,))

    weights = np.zeros((N,J))

    for j_ in range(1,J+1):
        for n_ in range(N):

            X_ = particle_states[n_]
            t_ = particle_times[n_]

            while t_ < j_:
                X_state, t_state = X_, t_
                X_,t_ = cs.ssa_step(X_, t_, V, prop_func)

            particle_states[n_] = X_state
            particle_times[n_] = t_state

            # weights
            weights[n_, j_-1] = pfu.pi_data(noisy_data[j_,:], X_state)

        sumweights = np.sum(weights[:,j_-1])

        if sumweights < 1e-150:
            return -np.inf

        probs = weights[:,j_-1] / sumweights if sumweights > 0 else 1/N * np.ones((N,))  # do safe division

        indxs = rng.choice(N, size=N, replace=True, p=probs)

        particle_states = particle_states[indxs]
        particle_times = particle_times[indxs]

    mean = np.mean(weights, axis=0)

    log_pihat_y =  np.sum(np.log(mean)) if not (mean == 0.).any() else -np.inf
    return log_pihat_y


@jit(nopython=True)
def run_pf_ssa_prop(N : int, V : np.array, logc : np.array, noisy_data : np.array) -> float:
    J=50

    c1, c2, c3, c4, c5 = np.exp(logc)

    particle_states = np.zeros((N,2))
    particle_times = np.zeros((N,))

    weights = np.zeros((N,J))

    for j_ in range(1,J+1):

        for n_ in range(N):

            X_ = particle_states[n_]
            t_ = particle_times[n_]

            while t_ < j_:
                X_state, t_state = X_, t_
                a = np.array([c1, c2, c3*X_[0], c4*X_[1], c5*X_[0]*X_[1]])
                X_,t_ = cs.ssa_prop_step(X_, t_, V, a)

            particle_states[n_] = X_state
            particle_times[n_] = t_state

            weights[n_, j_-1] = pfu.pi_data2(noisy_data[j_], X_state)

        sumweights = np.sum(weights[:,j_-1])

        if sumweights < 1e-150:
            return -np.inf


        probs = weights[:,j_-1] / sumweights if sumweights > 0 else 1/N * np.ones((N,))  # do safe division

        # working around the lack of np.random.choice with arbitrary weights...
        samples = np.random.rand(N).reshape((N,1))
        cumprobs = np.cumsum(probs).reshape((1,N))
        arr = samples < cumprobs
        indxs = np.argmax(arr, axis=1)

        particle_states = particle_states[indxs]
        particle_times = particle_times[indxs]

    mean = np.zeros((J,))
    for j in range(J):
        mean[j] = np.mean(weights[:,j])

    log_pihat_y =  np.sum(np.log(mean)) if not (mean == 0.).any() else -np.inf
    return log_pihat_y



def run_pf_hybrid1(N : int, V : np.array, logc : np.array, noisy_data : np.array, delta_t : float, Delta_t : float, beta_f) -> float:
    J=50

    c1, c2, c3, c4, c5 = np.exp(logc)
    prop_func = lambda X : np.array([c1, c2, c3*X[0], c4*X[1], c5*X[0]*X[1]])

    particle_states = np.zeros((N,2))
    particle_times = np.zeros((N,))

    weights = np.zeros((N,J))

    for j_ in range(1,J+1):
        for n_ in range(N):

            X_ = particle_states[n_]
            t_ = particle_times[n_]

            while t_ < j_:
                X_state, t_state = X_, t_
                X_,t_ = cs.hybrid1_step(X_, t_, V, prop_func, beta_f, delta_t, Delta_t)

            particle_states[n_] = X_state
            particle_times[n_] = t_state

            # weights
            weights[n_, j_-1] = pfu.pi_data(noisy_data[j_,:], X_state)
            #print(weights[n_,:j_-1])
        sumweights = np.sum(weights[:,j_-1])

        ## cut short if any of the sumweights is (extremely close to) 0.
        if sumweights < 1e-150: 
            return -np.inf

        probs = weights[:,j_-1] / sumweights if sumweights > 0 else 1/N * np.ones((N,))  # do safe division
        indxs = rng.choice(N, size=N, replace=True, p=probs)
        particle_states = particle_states[indxs]
        particle_times = particle_times[indxs]

    mean = np.mean(weights, axis=0)

    log_pihat_y =  np.sum(np.log(mean)) if not (mean == 0.).any() else -np.inf
    return log_pihat_y

def run_pf_hybrid_tau(N : int, V : np.array, logc : np.array, noisy_data : np.array, delta_t : float, Delta_t : float, beta_f) -> float:
    J=50

    c1, c2, c3, c4, c5 = np.exp(logc)
    prop_func = lambda X : np.array([c1, c2, c3*X[0], c4*X[1], c5*X[0]*X[1]])

    particle_states = np.zeros((N,2))
    particle_times = np.zeros((N,))

    weights = np.zeros((N,J))

    for j_ in range(1,J+1):
        for n_ in range(N):

            X_ = particle_states[n_]
            t_ = particle_times[n_]

            while t_ < j_:
                X_state, t_state = X_, t_
                X_,t_ = cs.hybrid1_tau_step(X_, t_, V, prop_func, beta_f, delta_t, Delta_t)
                #cs.hybrid1_step(X_, t_, V, prop_func, beta_f, delta_t, Delta_t)

            particle_states[n_] = X_state
            particle_times[n_] = t_state

            # weights
            weights[n_, j_-1] = pfu.pi_data(noisy_data[j_,:], X_state)
            #print(weights[n_,:j_-1])

        sumweights = np.sum(weights[:,j_-1])

        ## cut short if any of the sumweights is (extremely close to) 0.
        if sumweights < 1e-150:
            return -np.inf

        probs = weights[:,j_-1] / sumweights if sumweights > 0 else 1/N * np.ones((N,))  # do safe division
        indxs = rng.choice(N, size=N, replace=True, p=probs)
        particle_states = particle_states[indxs]
        particle_times = particle_times[indxs]

    mean = np.mean(weights, axis=0)

    log_pihat_y =  np.sum(np.log(mean)) if not (mean == 0.).any() else -np.inf
    return log_pihat_y


@jit(nopython=True)
def run_pf_hybridtau_prop(N : int, V : np.array, logc : np.array, noisy_data : np.array, I1 : float, I2 : float, delta_t : float, Delta_t : float) -> float:
    J=50

    def fbeta(x : float, I1 : float, I2 : float):
        if x <= I1:
            return 1.
        elif x >= I2:
            return 0.
        else:
            return (I2-x)/(I2-I1)

    c1, c2, c3, c4, c5 = np.exp(logc)

    particle_states = np.zeros((N,2))
    particle_times = np.zeros((N,))

    weights = np.zeros((N,J))

    for j_ in range(1,J+1):

        for n_ in range(N):

            X_ = particle_states[n_]
            t_ = particle_times[n_]

            while t_ < j_:
                X_state, t_state = X_, t_
                a = np.array([c1, c2, c3*X_[0], c4*X_[1], c5*X_[0]*X_[1]])

                beta = np.zeros((5,))
                beta[0] = 1. - (1. - fbeta(X_[0],I1,I2)) # first production reaction, only depends on X1
                beta[1] = 1. - (1. - fbeta(X_[1],I1,I2)) # second production reaction, only depends on X2
                beta[2] = 1. - (1. - fbeta(X_[0],I1,I2)) # first decay reaction, only depends on X1
                beta[3] = 1. - (1. - fbeta(X_[1],I1,I2)) # second decay reaction, only depends on X2
                beta[4] = 1. - (1. - fbeta(X_[1],I1,I2))*(1. - fbeta(X_[0],I1,I2)) # dimerization

                X_,t_ = cs.hybrid1_tau_step_numba(X_, t_, V, a, beta, delta_t, Delta_t)


            particle_states[n_] = X_state
            particle_times[n_] = t_state

            # weights
            weights[n_, j_-1] = pfu.pi_data2(noisy_data[j_], X_state)

        sumweights = np.sum(weights[:,j_-1])

        ## cut short if any of the sumweights is (extremely close to) 0.
        if sumweights < 1e-150: 
            return -np.inf

        probs = weights[:,j_-1] / sumweights if sumweights > 0 else 1/N * np.ones((N,))  # do safe division

        # working around the lack of np.random.choice with arbitrary weights...
        samples = np.random.rand(N).reshape((N,1))
        cumprobs = np.cumsum(probs).reshape((1,N))
        arr = samples < cumprobs
        indxs = np.argmax(arr, axis=1)

        #indxs = np.random.choice(np.arange(N), N, True, probs)
        particle_states = particle_states[indxs]
        particle_times = particle_times[indxs]

    #mean = np.mean(weights, axis=0)
    mean = np.zeros((J,))
    for j in range(J):
        mean[j] = np.mean(weights[:,j])

    log_pihat_y =  np.sum(np.log(mean)) if not (mean == 0.).any() else -np.inf
    return log_pihat_y


def select_new_c(logc : np.array, logc_star : np.array, log_pihat_y_c : float, log_pihat_y_cstar : float):

    if np.isinf(log_pihat_y_cstar):
        return logc, log_pihat_y_c, 0

    p = np.exp(np.minimum(0., log_pihat_y_cstar - log_pihat_y_c))
    if rng.binomial(n=1, p=p):
        return logc_star, log_pihat_y_cstar, 1
    return logc, log_pihat_y_c, 0
