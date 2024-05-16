import numpy as np
import chem_utils as cu
import chem_step as cs
import time
import os
import sys
import matplotlib.pyplot as plt


###### EDDIE PARAMS ######

M = int(sys.argv[1])
proc = int(sys.argv[2])

# os.chdir("/exports/eddie/scratch/s2096295/ags")

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

sa = np.zeros((4, M))
T = 1e3

t0 = time.process_time()

th = []
Xh = []

for i in range(M):

    X = np.array([1, 0, 0, 0])
    t = 0

    while t < T:
        X, t = cs.ssa_step(X, t, V, prop_func)
        th.append(t)
        Xh.append(X)

    sa[:, i] = X


t1 = time.process_time()


Xh = np.array(Xh)

print(f"Elapsed time: {t1-t0:.6g}")

#np.save(f"AGS-ssa-samples-{M}.{proc}", sa)

plt.figure(figsize=(9,6))
plt.fill_between(th, 5, 10, alpha=.4, label='5-10')
plt.fill_between(th, 10, 15, alpha=.4, label='10-15')
plt.fill_between(th, 15, 20,  alpha=.4, label='15-20')

for i,name in enumerate(["DNA", "mRNA", "P", "DNA0"]):
    plt.plot(th, Xh[:,i], label=name)

#plt.hlines(5, th[0], th[-1], label="zone 1")
#plt.hlines(10, th[0], th[-1], label="zone 2")
#plt.hlines(15, th[0], th[-1])
#plt.hlines(20, th[0], th[-1], label="zone 3")

plt.grid()
plt.legend()
plt.savefig("fig1.pdf", format='pdf')
plt.show()
