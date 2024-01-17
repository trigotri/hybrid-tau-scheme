import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# SSA stuff
Ns_ssa = [800, 640, 80, 320]
scs_ssa = [1, 10, 100, 1000]
numit_ssa = 200000
gammas_ssa = [0.05, 4.0, 1.0, 2.0]


Ns_tau = [320, 1280, 160, 160]
scs_tau = [1, 10, 100, 1000]
numit_tau = 200000
gammas_tau = [2.0, 1.0, 1.0, 1.0]

scs_str = list(map(str, scs_ssa))
numit = 200000

def clean_rows(arr):
    mod = 0
    mod = np.argmax(1 - (arr[0] == 0))
    stop = np.where(arr[0][mod:] == 0)[0][0] + mod if (arr[0][mod:] == 0).any() else numit
    return arr[:,:stop]

for ssa_stuff, tau_stuff in zip(zip(scs_ssa,Ns_ssa,gammas_ssa), zip(scs_tau,Ns_tau,gammas_tau)):

    sc_ssa,N_ssa,gamma_ssa = ssa_stuff
    sc_tau,N_tau,gamma_tau = tau_stuff

    dat_hybridtau = np.load(f'hybrid-tau-logc-dat-{sc_tau}-{numit_tau}-{N_tau}-{gamma_tau}.npy')


    dat_hybridtau = clean_rows(dat_hybridtau)
    expdat_hybridtau = np.exp(dat_hybridtau)

    dat_ssa = np.load(f'ssa-logc-dat-{sc_ssa}-{numit_ssa}-{N_ssa}-{gamma_ssa}.npy')
    dat_ssa = clean_rows(dat_ssa)
    expdat_ssa = np.exp(dat_ssa)

    truec = np.array([2,sc_ssa,1/50,1,1/(50*sc_ssa)])
    indxs = [0,3,1,4]

    swapindxs = [0,2,1,3]

    expdat_ssa = expdat_ssa[swapindxs]
    expdat_hybridtau = expdat_hybridtau[swapindxs]


    fig, axes = plt.subplots(1,len(indxs), figsize=(12,6))

    for i in range(2):

        axes[i].set_title(f'$c_{indxs[i]+1}$')
        sns.boxplot(data=[expdat_ssa[i], expdat_hybridtau[i]], ax=axes[i])
        #sns.boxplot(data=[expdat_ssa[i]], ax=axes[i])
        axes[i].axhline(y = truec[indxs[i]],    # Line on y = 0.2
                xmin = 0.1, # From the left
                xmax = 0.9,
                color = "red", linestyle = "dashed") # To the right
        axes[i].set_xticklabels(['ssa', 'tau-leap'])

    for i in range(2, 4):

        #fig, ax = plt.subplots()
        axes[i].set_title(f'$c_{indxs[i]+1}$')
        axes[i].set_yscale('log')
        sns.boxplot(data=[expdat_ssa[i], expdat_hybridtau[i]], ax=axes[i])
        #sns.boxplot(data=[expdat_ssa[i]], ax=axes[i])
        axes[i].axhline(y = truec[indxs[i]],    # Line on y = 0.2
                xmin = 0.1, # From the left
                xmax = 0.9,
                color = "red", linestyle = "dashed") # To the right
        axes[i].set_xticklabels(['ssa', 'tau-leap'])

    fig.suptitle(f'sc = {sc_ssa}')
    fig.tight_layout()
    fig.savefig(f'../plots/violinplot-{sc_ssa}-comparison.pdf', format='pdf')

    #plt.show()
