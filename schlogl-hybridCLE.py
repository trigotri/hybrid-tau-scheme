import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import os
import sys
import matplotlib.pyplot as plt

'''
Generates [M] final-time realisations of Schlogl model with Hybrid CLE up to time T=50, and saves as "schlogl-samples-HA1-[TSS]-[IS]-[M].[proc].npy" .
Time Step Set [TSS] (1 or 2) and Interval Set [IS] (1 or 2) are also passed.
'''

# executed on a cluster, and the data re-assembled afterwards

##### EDDIE PARAMS #####
M = int(sys.argv[1])
proc = int(sys.argv[2])
TSS = int(sys.argv[3])
IS = int(sys.argv[4])

if TSS==1:
    delta_t,Delta_t = 1e-2,1e-3
elif TSS==2:
    delta_t, Delta_t = .25,.25
else:
    raise Exception('Invalid Time Step Set')

if IS==1:
    I1,I2 = 35.,45.
elif IS==2:
    I1,I2 = 45.,55.
else:
    raise Exception('Invalid Interval Set')

##### SYSTEM PARAMS #####

S = ['S']
emptySet = []

c1 = 3e-7
c2 = 1e-4
c3 = 1e-3
c4 = 3.5

N1 = 1e5
N2 = 2e5

r1 = cu.Reaction(2*S, 3*S, c1)
r2 = cu.Reaction(3*S, 2*S, c2)
r3 = cu.Reaction(emptySet, S, c3)
r4 = cu.Reaction(S, emptySet, c4)
rs = cu.ReactionSet(r1, r2, r3, r4)

c1N12 = c1*N1/2
c26 = c2/6
c3N2 = c3*N2

V = rs.stochioimetric_matrix()

prop_func = lambda X : np.array([c1N12*X[0]*(X[0]-1),
                                c26*X[0]*(X[0]-1)*(X[0]-2),
                                c3N2,
                                c4*X[0]])

def fbeta(x, I1, I2):
    if x <= I1:
        return 1.
    elif x >= I2:
        return 0.
    else:
        return (I2-x)/(I2-I1)

def beta_f(X):
    a = fbeta(X[0],I1,I2)
    return np.array([a,a,a,a])

save_arr = np.zeros((M,)).astype('uint16')


t0 = time.process_time()
for i in range(M):
    t=0
    X = np.array([250])

    T = 50
    while t < T:
        X,t = cs.hybrid1_step(X, t, V, prop_func, beta_f, delta_t, Delta_t)

    save_arr[i] = X[0]
t1 = time.process_time()

np.save(f'dat/schlogl-samples-HA1-{TSS}-{IS}-{M}.{proc}', save_arr)
print(f'Elapsed time: {t1-t0:.5f}')
