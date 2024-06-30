
import numpy as np

ssa_mean = np.load("ags-ssa-times.npy").mean()

for j in range(2):
    for i in range(3):
        hcle_mean = np.load(f"ags-hcle-times-IS{i}-TS{j}.npy").mean()
        print(f"{hcle_mean:.5e}\t{ssa_mean / hcle_mean :.4g}")

print(f"{ssa_mean:.5e}\t1")
