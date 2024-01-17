import numpy as np
import chem_utils as cu
import part_filt_utils as pfu
import part_filt_algs as pfa
import time
import sys
import os

def remove_mean_covar_files(sc : int):

    fnames = [f'dat/ssa-logc-mean-{sc}.npy', f'dat/ssa-logc-cov-{sc}.npy']
    for fname in fnames:
        if os.path.isfile(fname):
            os.remove(fname)

def update_mean_and_covar(sc : int, numit : int, N : int, gamma : float):

    mean = np.load(f'dat/ssa-logc-mean-{sc}-{numit}-{N}-{gamma}.npy')
    np.save(f'dat/ssa-logc-mean-{sc}', mean)

    covar = np.load(f'dat/ssa-logc-covar-{sc}-{numit}-{N}-{gamma}.npy')
    np.save(f'dat/ssa-logc-cov-{sc}', covar)


def tune_gamma_and_N(sc : int, N : int, numit = 100):

    vals = np.load(f'dat/ssa-logc-mean-{sc}.npy')
    logc = np.array(np.log((2,sc,1/50,1,1/(50*sc))))
    indx = [0,1,3,4]
    logc[indx] = vals


    noisy_data = np.load(f'dat/ndataX-sc-{int(sc)}.npy')
    V = np.array([[1., 0., -1., 0,  -1.],
                  [0., 1.,  0., -1,  19.]])

    # set N
    print(f'Setting N... (sc = {sc})')

    logpihist = np.zeros((numit,))
    for i in range(numit):
        logpihist[i] = pfa.run_pf_ssa_prop(N=N, V=V, logc=logc, noisy_data=noisy_data)
    var = np.var(logpihist)
    print(f'Tried N = {N}, var is {var}')

    while var > 2.5:
        N*=2
        logpihist = np.zeros((numit,))
        for i in range(numit):
            logpihist[i] = pfa.run_pf_ssa_prop(N=N, V=V, logc=logc, noisy_data=noisy_data)
        var = np.var(logpihist)
        print(f'Tried N = {N}, var is {var}')

    print(f'N set to {N}')

    print('Setting gamma...')
    gamma = 1.

    run_xp_pf_ssa(sc, numit, N, gamma)
    accrej = np.load(f'dat/ssa-accrej-{sc}-{numit}-{N}-{gamma}.npy')[0]

    while accrej < .02 or accrej > .15:

        if accrej <= .02:
            gamma/=2
        elif accrej > .15:
            gamma*=2
        else:
            raise Exception("this shouldn't be possible...")

        print(f'Accrej = {accrej} for gamma = {gamma}')

        run_xp_pf_ssa(sc, numit, N, gamma)
        accrej = np.load(f'dat/ssa-accrej-{sc}-{numit}-{N}-{gamma}.npy')[0]

    with open(f'dat/ssa-xp-dat.txt', 'a+') as file:
        file.write(f'{sc}\t200000\t{N}\t{gamma}')

    print("Completed gamma and N")

    return gamma, N

def run_xp_pf_ssa(sc : int, numit : int, N : int, gamma : float, load_data=True, hack_mean = False):

    indx = [0,1,3,4]
    logc = np.array(np.log((2,sc,1/50,1,1/(50*sc))))
    #c1,c2,c3,c4,c5 = np.exp(logc)
    logc_hatcov = np.eye(len(indx))

    if load_data:
        fname = f'dat/ssa-logc-mean-{sc}.npy'
        if os.path.isfile(fname):
            print('Loading mean(logc_hat)...')
            logc[indx] = np.load(fname)
        else:
            print('Using perturbed value for starting guess')
            logc+= .5 * np.random.normal(size=(logc.shape[0]))

        ### load hat covariance if exists, else take id
        fname = f'dat/ssa-logc-cov-{sc}.npy'
        if os.path.isfile(fname):
            print('Loading cov(logc_hat)...')
            logc_hatcov = np.load(fname)
        else:
            print('Using identity...')

    ### EXPERIMENT SETUP
    logc_hist = np.zeros((len(indx),int(numit+1)))
    log_pi_hist = np.zeros((numit+1,))

    noisy_data = np.load(f'dat/ndataX-sc-{int(sc)}.npy')

    # stoichiometric matrix
    V = np.array([[1., 0., -1., 0,  -1.],
                [0., 1.,  0., -1,  19.]])

    ### run particle filter once, pi^(y | c)
    log_pihat_y_c = pfa.run_pf_ssa_prop(N=N, V=V, logc=logc, noisy_data=noisy_data)
    #print('log_pihat(y|c) =', log_pihat_y_c)

    if np.isinf(log_pihat_y_c) or np.isnan(log_pihat_y_c):
        raise Exception('First guess is not valid')

    indx = [0,1,3,4]
    log_pi_hist[0] = log_pihat_y_c
    logc_hist[:,0] = logc[indx]

    t0 = time.time()

    tally = 0

    ### MCMC
    for i_ in range(numit):

        if (i_ + 1) % 1000 == 0:
            print(i_+1)
        #print(i_+1)

        # sample c*
        logc_star = pfu.gaussian_proposal_logc(logc, cov=gamma*logc_hatcov)
        #pfu.gaussian_proposal_logc2(logc=logc, cov=gamma*logc_hatcov)

        # run particle filter with that candidate, compute pi^(y | c*)
        log_pihat_y_cstar = pfa.run_pf_ssa_prop(N=N, V=V, logc=logc_star, noisy_data=noisy_data)

        # accept/reject
        logc, log_pihat_y_c, change = pfa.select_new_c(logc, logc_star, log_pihat_y_c, log_pihat_y_cstar)

        logc_hist[:,i_+1] = logc[indx]
        log_pi_hist[i_+1] = log_pihat_y_c
        tally+=change

        if (i_+1) % 1000 == 0:
            np.save(f'dat/ssa-logc-dat-{sc}-{numit}-{N}-{gamma}', logc_hist)
            np.save(f'dat/ssa-log-pi-hist-{sc}-{numit}-{N}-{gamma}', log_pi_hist)

        if (i_+1) % 500 == 0:
            print('Iteration ', i_+1)
            print('Acc/rej', tally/i_)
            print('logc', logc)

    t1 = time.time()

    with open(f'dat/ssa.txt', 'a+') as f:
        f.write(f'{sc}\t{numit}\t{N}\t{gamma}\t{t1 - t0}\t{tally/numit}\n')

    np.save(f'dat/ssa-accrej-{sc}-{numit}-{N}-{gamma}', np.array([tally/numit]))
    np.save(f'dat/ssa-logc-dat-{sc}-{numit}-{N}-{gamma}', logc_hist)
    np.save(f'dat/ssa-log-pi-hist-{sc}-{numit}-{N}-{gamma}', log_pi_hist)

    np.save(f'dat/ssa-logc-mean-{sc}-{numit}-{N}-{gamma}', np.mean(logc_hist, axis=1))
    np.save(f'dat/ssa-logc-covar-{sc}-{numit}-{N}-{gamma}', np.cov(logc_hist))

    # this is a trick for burn-in..
    if hack_mean:
        np.save(f'dat/ssa-logc-mean-{sc}-{numit}-{N}-{gamma}', logc_hist[:,-1])


if __name__ == '__main__':

    ## EXPERIMENT PARAMETERS
    sc = int(sys.argv[1])
    N = int(sys.argv[2])
    gamma = float(sys.argv[3])

    setup = True
    verbose = True

    if setup:

        dryN = 20
        drynumit = 100
        drygamma = 0.1

        burnin = 200

        remove_mean_covar_files(sc)

        if verbose:
            print('Setting up covariance...')

        run_xp_pf_ssa(sc, drynumit, dryN, drygamma) # for covariance
        update_mean_and_covar(sc, drynumit, dryN, drygamma)

        if verbose:
            print('Setting up N and gamma...')

        gamma, N = tune_gamma_and_N(sc, N = dryN, numit=drynumit) # for gamma, N
        if verbose:
            print('Burn-in phase...')

        run_xp_pf_ssa(sc, burnin, N, gamma, hack_mean=True)
        update_mean_and_covar(sc, burnin, N, gamma)

        if verbose:
            print('Setup complete.')
            print(f'Assigned value for N ::', N)
            print(f'Assigned value for gamma ::', gamma)

    numit = 200000
    run_xp_pf_ssa(sc, numit, N, gamma, load_data=setup)
