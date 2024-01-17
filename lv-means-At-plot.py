import numpy as np
import matplotlib.pyplot as plt

N = 10000
nsteps = 7
tspan = np.arange(nsteps)


names = ['ssa', 'hcle', 'htau', 'cle']
linestyles = ['-', '-.', '--', '--']


fig = plt.figure(0, figsize=(9,6))

for name,ls in zip(names, linestyles):
    fname = f'lv/{name}-{N}-{nsteps}.npy'
    dat = np.load(fname)
    mdat = dat.mean(axis=0)
    plt.plot(tspan, mdat[:,0], ls, label=f'{name}')

plt.xlabel('$t$')
plt.ylabel('$\mathbb{E}[A(t)]$')
plt.title('Evolution of $A(t)$')

plt.grid()
plt.yscale('log')
plt.legend()
fig.tight_layout()
plt.savefig('lv/means-At.pdf', format='pdf')
