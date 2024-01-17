import numpy as np
from scipy.stats import poisson, multivariate_normal
from scipy.special import factorial
from numba import jit

rng = np.random.default_rng()

def gen_weights(Yj,Xj,pi) -> np.array:
    '''
    Constructs weights for particle filter w_k = \pi(yj | xj[k]), returns result as a numpy array.
    Xj represents the particles at state j, where xj[k] represents the k-th particle.
    pi must be given as a lambda function.
    '''
    N = Xj.shape[0]
    w = np.zeros((N,))
    for k in range(N):

        w[k] = pi(Yj, Xj[k])

        #print(w[k])

    return w

def simulate_particles(Xj1, delta_tj, c_star, method) -> np.array:
    '''
    Evolves the particles from X(j-1)[k] to X(j)[k] with time difference delta_tj[k] using method.
    '''
    N = Xj1.shape[0]
    Xj = np.zeros_like(Xj1)

    for k in range(N):
        Xj[k] = method(Xj1[k], delta_tj[k], c_star)

    return Xj

def resample_particles(Xj : np.array, weights: np.array):
    N = Xj.shape[0]
    indx = rng.choice(N, p=weights, size=(N,))
    return Xj[indx.astype(int)]


def pihat_Yc_star(weights: np.array):
    '''
    Compute the probability \hat{pi}(y | c*) using the weights.
    '''
    N = weights.shape[0]
    M = weights.shape[1]
    return np.prod(np.asum(weights, axis=0))/(N**M)


def draw_c_star(ci1, default=True, q=None):
    '''
    Draws c* from conditional distribution q(.|c(i-1)). If default is True, then q(.|c(i-1)) is a normal random variable
    with scale 1 and mean c(i-1).
    Note that ci1 must be an array with shape (M,)

    '''
    if default:
        return ci1 + rng.normal(scale=1., size=ci1.shape)
    if q is None:
        raise Exception('Transition kernel q cannot be None if default is False')

    return q(ci1)


def priorC(c, n=8):
    '''
    Computes probability of prior of log of c1,c2,c4,c5, are indendent U(-n,n) (by default n=8)
    '''
    # lo = np.log(c)
    # for i in [0,1,3,4]:
    #     loi = lo[i]
    #     if (loi < -n) or (loi > n):
    #         return 0.

    c1 = c[[0,1,3,4]]
    res = (c1 < np.exp(n)) * (c1 > np.exp(-n)) * np.reciprocal(c1) / (2*n)

    return np.prod(res)

def logpriorC(c, n=8):
    '''
    Computes probability of prior of log of c1,c2,c4,c5, are indendent U(-n,n) (by default n=8)
    '''
    # lo = np.log(c)
    # for i in [0,1,3,4]:
    #     loi = lo[i]
    #     if (loi < -n) or (loi > n):
    #         return 0.

    c1 = c[[0,1,3,4]]
    res = (c1 < n) * (c1 > -n)  / (2*n)

    return np.prod(res)

def genPriorC(n=8):
    '''
    Return a sample from the considered distribution.
    '''
    c1 = rng.uniform(low=-n, high=n, size=(4,))
    c = np.zeros(5)
    c[[0,1,3,4]] = np.exp(c1)
    c[2] = .02
    return c

def com_varlogc(chist):
    indx = [0,1,3,4]

    if chist is None or chist.shape[1] <= 1:
        varlogc = np.eye(4)

    else:
        chist2 = chist.copy()
        chist2 = np.log(chist2[indx,:])
        varlogc = np.cov(chist2)

    return varlogc




# def genQC(c, varlogc=np.eye(4), gamma=1.):
#     '''
#     Generates Gaussian proposal for log(c). c_3 is fixed to .02
#     If convariance matrix of log(c) is given, uses this in Gaussian increment, else identity matrix is used.
#     Gamma is a tuning parameter (proposal is given by log c* = log c + gamma * N(0, varlogc)).
#     '''
#     indx = [0,1,3,4]
#     logc1 = np.log(c[indx])

#     cs1 = logc1 + gamma*rng.multivariate_normal(np.zeros(4), varlogc)

#     cs = np.zeros((5,))
#     cs[indx] = np.exp(cs1)
#     cs[2] = .02

#     return cs

# def qC(cn, cc, varlogc=np.eye(4), gamma=1.):
#     #varlogc = com_varlogc(chist)
#     indx = [0,1,3,4]
#     return multivariate_normal.pdf(np.log(cn[indx]), mean=np.log(cc[indx]), cov=(gamma**2)*varlogc)

def pi_data(Y : np.array, X : np.array) -> float:
    '''
    Probability of measuring Y given parameter X
    '''

    if (np.rint(X) < 0.).any(): # nuclear option
        return 0.

    weights = poisson.pmf(Y,X * (X >= 0)) # this to take care of the case when rint(X) = 0, X < 0

    for i in range(X.shape[0]):
        if np.rint(X[i]) == 0.:
            if Y[i] == 1:
                weights[i] = .1
            elif Y[i] == 0:
                weights[i] = .9
            else:
                #qweights[i] = 0.
                return 0

    return np.product(weights)

@jit(nopython=True)
def pi_data2(Y : np.array, X : np.array) -> float:

    if (np.rint(X) < 0.).any(): # nuclear option
        return 0.

    # working around the fact that I cannot use poisson.pmf
    slogY = np.zeros_like(Y)
    for jj in range(slogY.shape[0]):
        slogY[jj] = np.sum(np.log(np.arange(2,Y[jj]+1)))

    logweights = np.log(X) * Y - X - slogY
    weights = np.exp(logweights)

    for i in range(X.shape[0]):
        if np.rint(X[i]) == 0.:
            if Y[i] == 1:
                weights[i] = .1
            elif Y[i] == 0:
                weights[i] = .9
            else:
                return 0.


    return np.prod(weights)


def log_pi_data(Y : np.array, X : np.array) -> float:
    '''
    Probability of measuring Y given parameter X
    '''

    if (np.rint(X) < 0.).any(): # nuclear option
        return 0.

    logweights = poisson.logpmf(Y,X * (X >= 0)) # this to take care of the case when rint(X) = 0, X < 0

    for i in range(X.shape[0]):
        if np.rint(X[i]) == 0.:
            if Y[i] == 1:
                logweights[i] = np.log(.1)
            elif Y[i] == 0:
                logweights[i] = np.log(.9)
            else:
                return -np.inf

    return np.sum(logweights)

def priorX0():
    return np.zeros(2)

def gaussian_proposal_logc(logc : np.array, cov : np.array) -> np.array:
    mean = np.delete(logc, 2)
    rv = multivariate_normal(mean=mean, cov=cov)
    logcstar = np.insert(rv.rvs(), 2, np.log(1/50))
    while np.amax(np.abs(logcstar)) >= 12:
        logcstar = np.insert(rv.rvs(), 2, np.log(1/50))
    return logcstar


def gaussian_proposal_logc2(logc : np.array, cov : np.array) -> np.array:
    indxs = [0,1,4]
    mean = logc[indxs]
    rv = multivariate_normal(mean=mean, cov=cov)
    logcstar = np.insert(rv.rvs(), 2, [np.log(1/50), np.log(1)])
    while np.amax(np.abs(logcstar)) >= 12:
        logcstar = np.insert(rv.rvs(), 2, [np.log(1/50), np.log(1)])
    return logcstar

def prior_logc():
    return np.insert(rng.uniform(low=-8., high=8., size=4), 2 , np.log(1/50))

def accept_cstar(log_pihat_y_cstar : float, log_pihat_y_c : float) -> np.array:
    p = np.exp(np.minimum(0., log_pihat_y_cstar - log_pihat_y_c))
    return rng.binomial(n=1, p=p)
