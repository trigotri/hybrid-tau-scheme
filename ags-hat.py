import numpy as np
import chem_algs as ca
import chem_utils as cu
import chem_step as cs
import time
import os
import sys
import matplotlib.pyplot as plt

###### EDDIE PARAMS ######

M = int(sys.argv[1])
I1, I2 = float(sys.argv[2]), float(sys.argv[3])
proc = int(sys.argv[4])

###### SYSTEM PARAMS #####

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
prop_func = lambda X: rs.propensity_f(X)

delta_t, Delta_t = 1e-1, 1e-1


def fbeta(x, I1, I2):
    if x <= I1:
        return 1.0
    elif x >= I2:
        return 0.0
    else:
        return (I2 - x) / (I2 - I1)


def beta_f(X):

    # ['DNA', 'DNA0', 'P', 'mRNA']
    a1 = 1 - fbeta(X[0], I1, I2)
    b1 = 1 - fbeta(X[1], I1, I2)
    c1 = 1 - fbeta(X[2], I1, I2)
    d1 = 1 - fbeta(X[3], I1, I2)

    f1 = 1 - a1 * c1 * b1

    return np.array([1 - a1 * d1, 1 - d1 * c1, f1, f1, 1 - d1, 1 - c1])


sa = np.zeros((4, M))
T = 1e3

t0 = time.process_time()
for i in range(M):

    X = np.array([1, 0, 0, 0])
    t = 0

    while t < T:
        # X,t = cs.hybrid1_step(X, t, V, prop_func, beta_f, delta_t, Delta_t)
        X, t = cs.hybrid1_tau_step(X, t, V, prop_func, beta_f, delta_t, Delta_t)
        # X,t = cs.ssa_step(X,t, V, prop_func)
    sa[:, i] = X


t1 = time.process_time()


print(f"Elapsed time: {t1-t0:.6g}")

# np.save(f'AGS-ha1-samples-{int(I1)}-{int(I2)}-{M}.{proc}', sa)

# for i in range(4):
#     plt.plot(th, Xh[:,i], label=f'S{i+1}')

# plt.plot(th, Xh[:, 3], label='mRNA')
# plt.plot(th, Xh[:, 2], label='P')

# plt.grid()
# plt.legend()
# plt.show()
