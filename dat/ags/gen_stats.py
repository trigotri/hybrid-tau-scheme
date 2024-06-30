import numpy as np


def stats(arr: np.array):
    return arr.mean(axis=0), arr.std(axis=0)

def fstr(s : float):
    return f"{s:<8.5g}"

ssa_states = np.load("ags-ssa-states.npy")

for alg in ["htau", "hcle"]:
    means_string = ""
    std_string = ""

    for i in range(2):
        for j in range(3):
            alg_stats = stats(np.load(f"ags-{alg}-states-IS{j}-TS{i}.npy"))
            means_string += "\t".join(list(map(fstr, alg_stats[0]))) + "\n"
            std_string += "\t".join(list(map(fstr, alg_stats[1]))) + "\n"

alg_stats = stats(ssa_states)
means_string += "\t".join(list(map(fstr, alg_stats[0]))) + "\n"
std_string += "\t".join(list(map(fstr, alg_stats[1]))) + "\n"

print(f'Means {alg}')
print(means_string)


print(f'Std {alg}')
print(std_string)


