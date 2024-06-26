import numpy as np
import matplotlib.pyplot as plt

N = 10000
nsteps = 7
tspan = np.arange(nsteps)


names = ['ssa', 'hcle', 'htau', 'cle', 'tau']
linestyles = ['-', '--', '--', '-.', '-.']
labels = ['SSA', 'H CLE', 'H $\\tau$', 'CLE', '$\\tau$-leap']


fig = plt.figure(0, figsize=(9,6))
plt.rcParams.update({"font.size" : 16})

for name,ls,lab in zip(names, linestyles, labels):
    fname = f'dat/{name}-{N}-{nsteps}.npy'
    dat = np.load(fname)
    mdat = dat.mean(axis=0)
    plt.plot(tspan, mdat[:,0], ls, label=lab)

plt.xlabel('$t$')
plt.ylabel('$\mathbb{E}[A(t)]$')

plt.grid()
plt.yscale('log')
plt.legend()
fig.tight_layout()
plt.savefig('dat/means-At.pdf', format='pdf')
plt.show()
