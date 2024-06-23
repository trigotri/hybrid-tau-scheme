import numpy as np
from numba import jit

rng = np.random.default_rng()

def ssa_step(X, t, V, propensity_f):
    a = propensity_f(X)

    asum = np.sum(a)

    xi1, xi2 = rng.uniform(size=(2,))

    j = np.argmax(xi1*asum<np.cumsum(a))
    tau = -np.log(xi2)/asum

    X = X + V[:,j]
    new_t = t + tau

    return X, new_t

def ssa_safe_step(X, t, V, propensity_f):
    a = propensity_f(X)

    asum = np.sum(a)
    if asum <= 0.:
        return X, t+1

    xi1, xi2 = rng.uniform(size=(2,))

    j = np.argmax(xi1*asum<np.cumsum(a))
    tau = -np.log(xi2)/asum

    X = X + V[:,j]
    new_t = t + tau

    return X, new_t

@jit(nopython=True)
def ssa_prop_step(X : np.array, t : float, V : np.array, a : np.array):
    asum = np.sum(a)
    xi1, xi2 = np.random.rand(2)

    j = np.argmax(xi1<np.cumsum(a/asum))
    tau = np.log(1/xi2)/asum

    new_X = X + V[:,j]
    new_t = t + tau

    return new_X, new_t

def tau_leap_step(X,t,V,propensity_f,tau):
    a = propensity_f(X)
    pjs = rng.poisson(a*tau)
    X = X + V @ pjs
    new_t = t+tau

    return X, new_t

@jit(nopython=True)
def tau_leap_step_prop(X : np.array,t : float,V : np.array, a : np.array, tau :float):

    pjs = np.zeros((len(a),)) ### this wordy expansion to make it numba-compatible
    atau = a * tau
    for i in range(len(a)):
        pjs[i] = np.random.poisson(atau[i])
    X = X + V @ pjs
    new_t = t+tau
    return X, new_t

def cle_step(X,t, V, propensity_f,tau):
    a = propensity_f(X)

    zjs = rng.normal(size=V.shape[1])
    tau_a = tau*a
    d = tau_a + np.sqrt(tau_a)*zjs
    new_X = X + V @ d
    new_t = t+tau

    return new_X, new_t

@jit(nopython=True)
def cle_step_nb_refl(X : np.array, t : float, V : np.array, props : np.array, tau : float):

    m = V.shape[1]

    new_t = t+tau

    # because Numba doesn't let you do it easily
    zjs = np.zeros((m,))
    for j in range(m):
        zjs[j] = np.random.randn()

    tau_props = tau*props
    d = tau_props + np.sqrt(tau_props)*zjs
    new_X = X + V @ d

    return np.abs(new_X), new_t

@jit(nopython=True)
def cle_step_numba(X : np.array, t : float, V : np.array, props : np.array, tau : float):

    m = V.shape[1]

    new_t = t+tau

    # because Numba doesn't let you do it easily
    zjs = np.zeros((m,))
    for j in range(m):
        zjs[j] = np.random.randn()

    tau_props = tau*props
    d = tau_props + np.sqrt(tau_props)*zjs
    new_X = X + V @ d

    return new_X, new_t


def cle_wt_step(X,t,V,propensity_f, dt):
    '''
    Using weak trapezoidal approximation for CLE.
    '''
    Lambda = propensity_f(X)

    Xi = rng.normal(size=V.shape[1])
    Xi_prime = rng.normal(size=V.shape[1])
    dt2 = dt/2

    X_star = X + dt2 * V @ Lambda + np.sqrt(dt2) * V @ np.multiply(np.sqrt(Lambda), Xi)
    H = 2*propensity_f(X_star) - Lambda
    X_new = X_star + dt2 * V @ H + np.sqrt(dt2) * V @ np.multiply(np.sqrt(H * (H > 0)), Xi_prime)

    return X_new, t+dt


################### HYBRID METHODS ###################

def hybrid1_step(X,t,V,propensity_f, beta_f, delta_t, Delta_t):
    beta = beta_f(X)

    # CLE
    if np.amax(beta) == 0:
        new_X, new_t = cle_wt_step(X, t, V, propensity_f, delta_t)

    # SSA -- for stability reasons, consider using ssa_prop_step
    elif np.amin(beta) == 1:
        prop_func_prime = lambda X : propensity_f(np.rint(X))
        new_X, new_t = ssa_step(X, t, V, prop_func_prime)

    # jump-diffusion
    else:
        prop_func_pp = lambda X: (1-beta)*propensity_f(X)

        a_prime = beta * propensity_f(np.rint(X))
        a0_prime = np.sum(a_prime)

        xi1, xi2 = rng.uniform(size=(2,))
        tau = -np.log(xi2)/a0_prime if a0_prime > 0 else Delta_t

        if tau < Delta_t:
            j = np.argmax(xi1*a0_prime<np.cumsum(a_prime))
            new_X, new_t = cle_wt_step(X,t,V,prop_func_pp, tau)
            new_X = new_X + V[:,j]

        else:
            new_X, new_t = cle_wt_step(X,t,V,prop_func_pp, Delta_t)


    return new_X, new_t

def hybrid1_tau_step(X,t,V,propensity_f, beta_f, delta_t, Delta_t):
    beta = beta_f(X)

    # Tau leap
    if np.amax(beta) == 0:
        new_X, new_t = tau_leap_step(X, t, V, propensity_f, delta_t)

    # SSA
    elif np.amin(beta) == 1:
        #prop_func_prime = lambda X : propensity_f(np.rint(X))
        new_X, new_t = ssa_step(X, t, V, propensity_f)

    # jump-diffusion
    else:
        prop_func_pp = lambda X: (1-beta)*propensity_f(X)

        #a_rint = propensity_f(np.rint(X))
        #a_prime = beta * a_rint
        a_prime = beta * propensity_f(X)
        a0_prime = np.sum(a_prime)

        xi1, xi2 = np.random.uniform(size=(2,))
        tau = -np.log(xi2)/a0_prime if a0_prime > 0 else Delta_t

        if tau < Delta_t:
            j = np.argmax(xi1<np.cumsum(a_prime/a0_prime))
            new_X, new_t = tau_leap_step(X,t,V,prop_func_pp, tau)
            new_X = new_X + V[:,j]

        else:
            new_X, new_t = tau_leap_step(X,t,V,prop_func_pp, Delta_t)

    return new_X, new_t

@jit(nopython=True)
def hybrid1_tau_step_numba(X,t,V, props, beta, delta_t, Delta_t, verbose=False):
    '''
    First running version of the Hybrid-tau (numba-compatible).
    Props are the already-computed propensities.
    '''

    # Tau leap
    if np.amax(beta) == 0:
        new_X, new_t = tau_leap_step_prop(X, t, V, props, delta_t)

    # SSA
    elif np.amin(beta) == 1:
        new_X, new_t = ssa_prop_step(X, t, V, props)

    # jump-tau-leap
    else:

        props_ssa = beta * props ## need to make sure this guy makes what it claims it does
        a0_prime = np.sum(props_ssa)

        xi1, xi2 = np.random.rand(2)
        tau = -np.log(xi2)/a0_prime if a0_prime > 0 else Delta_t
        #print(tau) if verbose else None

        props_tau =  (1.-beta)*props
        if tau < Delta_t:
            #print("blocking") if verbose else None
            j = np.argmax(xi1<np.cumsum(props_ssa/a0_prime))
            new_X, new_t = tau_leap_step_prop(X, t, V, props_tau, tau)
            new_X = new_X + V[:,j]

        else:
            new_X, new_t = tau_leap_step_prop(X, t, V, props_tau, Delta_t)
            #print("skipping") if verbose else None

    return new_X, new_t


@jit(nopython=True)
def hybrid1_cle_step_numba(X,t,V, props, beta, delta_t, Delta_t):
    '''
    First running version of the Hybrid-CLE (numba-compatible).
    Props are the already-computed propensities.
    '''
    props_rint = np.rint(props)

    # Tau leap
    if np.amax(beta) == 0:
        new_X, new_t = cle_step_numba(X, t, V, props, delta_t)

    # SSA
    elif np.amin(beta) == 1:
        new_X, new_t = ssa_prop_step(X, t, V, props_rint)

    # jump-tau-leap
    else:

        props_ssa = beta * props_rint ## need to make sure this guy makes what it claims it does
        a0_prime = np.sum(props_ssa)

        xi1, xi2 = np.random.rand(2)
        tau = -np.log(xi2)/a0_prime if a0_prime > 0 else Delta_t

        props_cle =  (1.-beta)*props
        if tau < Delta_t:
            j = np.argmax(xi1<np.cumsum(props_ssa/a0_prime))
            new_X, new_t = cle_step_numba(X, t, V, props_cle, tau) # if somwething goes wrong, it is here...
            new_X = new_X + V[:,j]

        else:
            new_X, new_t = cle_step_numba(X, t, V, props_cle, Delta_t)

    return new_X, new_t


def hybrid_alg1_method(X,t,V,propensity_f, beta_f, delta_t, Delta_t):
    beta = beta_f(X)
    method=None

    # CLE
    if np.amax(beta) == 0:
        new_X, new_t = cle_wt_step(X, t, V, propensity_f, delta_t)
        method=[0]

    # SSA -- for stability reasons, consider using ssa_prop_step
    elif np.amin(beta) == 1:
        prop_func_prime = lambda X : propensity_f(np.rint(X))
        new_X, new_t = ssa_step(X, t, V, prop_func_prime)
        method = [1]

    # jump-diffusion
    else:
        prop_func_pp = lambda X: (1-beta)*propensity_f(X)

        a_prime = beta * propensity_f(np.rint(X))
        a0_prime = np.sum(a_prime)

        #print('beta ', beta)
        #print('a\' ', a_prime)

        xi1, xi2 = rng.uniform(size=(2,))
        tau = -np.log(xi2)/a0_prime if a0_prime > 0 else Delta_t

        #print('tau ', tau)

        if tau < Delta_t:
            j = np.argmax(xi1*a0_prime<np.cumsum(a_prime))
            new_X, new_t = cle_wt_step(X,t,V,prop_func_pp, tau)
            new_X = new_X + V[:,j]
            method=(2,j)

        else:
            new_X, new_t = cle_wt_step(X,t,V,prop_func_pp, Delta_t)
            method = [3]


    return new_X, new_t, method


def test_hybrid1_tau_step(X,t,V,propensity_f, beta_f, delta_t, Delta_t):
    beta = beta_f(X)

    # Tau leap
    if np.amax(beta) == 0:
        method='tau'
        new_X, new_t = tau_leap_step(X, t, V, propensity_f, delta_t)

    # SSA
    elif np.amin(beta) == 1:
        method='SSA'
        #prop_func_prime = lambda X : propensity_f(np.rint(X))
        new_X, new_t = ssa_step(X, t, V, propensity_f)

    # jump-diffusion
    else:
        prop_func_pp = lambda X: (1-beta)*propensity_f(X)

        #a_rint = propensity_f(np.rint(X))
        #a_prime = beta * a_rint
        a_prime = beta * propensity_f(X)
        a0_prime = np.sum(a_prime)

        xi1, xi2 = np.random.uniform(size=(2,))
        tau = -np.log(xi2)/a0_prime

        if tau < Delta_t:
            method='jump diff. tau-ssa'
            j = np.argmax(xi1<np.cumsum(a_prime/a0_prime))
            new_X, new_t = tau_leap_step(X,t,V,prop_func_pp, tau)
            new_X = new_X + V[:,j]

        else:
            method='jump diff. tau'
            new_X, new_t = tau_leap_step(X,t,V,prop_func_pp, Delta_t)

    print(f'{beta} :: {method}')

    return new_X, new_t

