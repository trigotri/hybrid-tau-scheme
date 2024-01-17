import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import os
import sys

'''
Generates [M] final-time realisations of Schlogl model with SSA up to time T=50, and saves as "schlogl-samples-SSA-[M].[proc].npy" .
'''

# executed on a cluster, and the data re-assembled afterwards

##### EDDIE PARAMS #####
M = int(sys.argv[1])
proc = int(sys.argv[2])


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

# using these to speed up lambda-evaluation
c1N12 = c1*N1/2
c26 = c2/6
c3N2 = c3*N2

V = rs.stochioimetric_matrix()

prop_func = lambda X : np.array([c1N12*X[0]*(X[0]-1),
                                c26*X[0]*(X[0]-1)*(X[0]-2),
                                c3N2,
                                c4*X[0]])
# SSA
save_arr = np.zeros((M,)).astype('uint16')

t0 = time.process_time()
for i in range(M):
    t=0
    X = np.array([250])

    T = 50
    while t < T:
        X,t = cs.ssa_step(X, t, V, prop_func)

    save_arr[i] = X[0]
t1 = time.process_time()

np.save(f'dat/schlogl-samples-SSA-{M}.{proc}', save_arr)
print(f'Elapsed time: {t1-t0:.5f}')
