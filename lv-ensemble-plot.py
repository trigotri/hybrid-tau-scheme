import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.colors as colors

# loads data stored in hist-sparse.npz
# reduces size for speed rendering and displays result

reduce_size = True

spm = sparse.load_npz('hist-sparse.npz')
arr = spm.todense()

N = arr.shape[0]
k = 10
nN = N//k

arr2 = np.zeros((nN,nN))

if reduce_size:
    for i in range(nN):
        for j in range(nN):
            arr2[i,j] = np.sum(arr[k*i:k*i+k, k*j:k*(j+1)])

    arr=arr2


arr = arr/np.sum(arr)

fig, ax = plt.subplots(figsize=(9,6))

plt.pcolormesh(arr, rasterized=True, norm=colors.LogNorm(vmin=1e-9, vmax=1e-2), cmap='jet')

if reduce_size:
    plt.xticks(np.arange(0, nN+1, 2000//k), labels=np.arange(0, N+1, 2000))
    plt.yticks(np.arange(0, nN+1, 2000/k), labels=np.arange(0, N+1, 2000))


plt.colorbar()
#plt.title('Lotka-Volterra Ensemble')
plt.xlabel('A')
plt.ylabel('B')

plt.grid()
ax.set_axisbelow(True)
fig.tight_layout()

plt.savefig('plots/lv-plot')
plt.show()
