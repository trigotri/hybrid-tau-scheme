import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import matplotlib.pyplot as plt
import part_filt_utils as pfu
from scipy.stats import poisson

rng = np.random.default_rng()
save=True

#### SYSTEM PARAMS ####
for sc in [10**n for n in range(4)]:
    S1, S2 = ['S1'], ['S2']
    emptySet = []
    c1,c2, c3, c4, c5 = 2., sc, 1/50, 1., 1/(50*sc)

    r1 = cu.Reaction(emptySet, S1, c1)
    r2 = cu.Reaction(emptySet, S2, c2)
    r3 = cu.Reaction(S1, emptySet, c3)
    r4 = cu.Reaction(S2, emptySet, c4)
    r5 = cu.Reaction(S1+S2, 20*S2, c5)
    rs = cu.ReactionSet(r1, r2, r3, r4, r5)

    V = rs.stochioimetric_matrix()
    prop_func = lambda X: rs.propensity_f(X)

    X0 = np.zeros((2,))
    X = X0.copy()

    t = 0
    T = 50.

    Xh,th = [X],[t]

    while t < T:
        X,t = cs.ssa_step(X, t, V, prop_func)
        Xh.append(X)
        th.append(t)

    dataX = np.array(Xh)
    datat = np.array(th)


    tmesh = np.arange(0, 51)
    a = tmesh.reshape((len(tmesh), 1)) < datat.reshape((1, len(datat)))
    indx = np.argmax(a, axis=1)-1
    mdataX = dataX[indx]

    ndataX = rng.poisson(mdataX) + (mdataX == 0)*rng.binomial(1, .1, size=mdataX.shape)

    for i in range(2):
        plt.plot(datat, dataX[:,i])
        #plt.plot(tmesh, mdataX[:,i], 'o')

        plt.scatter(tmesh, ndataX[:,i])

    plt.show()

    if save:
        np.save(f'dat/cdataX-sc-{int(sc)}', dataX)
        np.save(f'dat/cdatat-sc-{int(sc)}', datat)

        np.save(f'dat/ndataX-sc-{int(sc)}', ndataX)
        np.save(f'dat/ndatat-sc-{int(sc)}', tmesh)
