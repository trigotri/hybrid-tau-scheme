import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})


names = ["ssa", "tau", "cle", "htau", "hcle"]
labels = ["SSA", "$\\tau$-leap", "CLE", "H $\\tau$", "H CLE"]
lss = ["--", "-", "--", "-", "--"]

for name,label,ls in zip(names, labels, lss):
    dat = np.load(f"./dat/{name}-times.npy")
    save_times= np.load(f"./dat/{name}-save-times.npy")

    plt.plot(save_times, dat.mean(axis=0), ls, label=label)

plt.plot(save_times, 1e-4 * save_times, '--', alpha=.4, label='$\\mathcal{O}(t)$')
plt.legend()
plt.grid()
plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel('computational time [s]')
plt.tight_layout()

plt.savefig('dat/lv-timings.pdf', format='pdf')

plt.show() if True else None
