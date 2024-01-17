import numpy as np
import chem_utils as cu
import part_filt_utils as pfu
import part_filt_algs as pfa
import time
import sys
import os


def remove_mean_covar_files(sc : int):

    fnames = [f'dat/hybrid-tau-logc-mean-{sc}.npy', f'dat/hybrid-tau-logc-cov-{sc}.npy']
    for fname in fnames:
        if os.path.isfile(fname):
            os.remove(fname)

def update_mean_and_covar(sc : int, numit : int, N : int, gamma : float):

    mean = np.load(f'dat/hybrid-tau-logc-mean-{sc}-{numit}-{N}-{gamma}.npy')
    np.save(f'dat/hybrid-tau-logc-mean-{sc}', mean)

    covar = np.load(f'dat/hybrid-tau-logc-covar-{sc}-{numit}-{N}-{gamma}.npy')
    np.save(f'dat/hybrid-tau-logc-cov-{sc}', covar)

    print('mean', mean)
    print('covar', covar)


def tune_gamma_and_N(sc : int, N : int, numit : int, I1 : int, I2 : int):

    vals = np.load(f'dat/hybrid-tau-logc-mean-{sc}.npy')
    logc = np.array(np.log((2,sc,1/50,1,1/(50*sc))))
    indx = [0,1,3,4]
    logc[indx] = vals

    Delta_t, delta_t = .01, .1

    noisy_data = np.load(f'dat/ndataX-sc-{int(sc)}.npy')
    V = np.array([[1., 0., -1., 0,  -1.],
                  [0., 1.,  0., -1,  19.]])

    # set N
    print('Setting N...')

    logpihist = np.zeros((numit,))
    val1 = pfa.run_pf_hybridtau_prop(N, V, logc, noisy_data, I1, I2, delta_t, Delta_t)
    i = 0
    while i < numit:
        if not np.isinf(val1):
            logpihist[i] = val1
            i+=1
        val1 = pfa.run_pf_hybridtau_prop(N, V, logc, noisy_data, I1, I2, delta_t, Delta_t)
    var = np.var(logpihist)

    if np.isnan(var):
        print(logpihist)
        raise Exception('Var is nan...')

    print(f'Tried N = {N}, var = {var}')


    while var < .2 or var > 2.5:
        N*=2
        logpihist = np.zeros((numit,))
        val1 = pfa.run_pf_hybridtau_prop(N, V, logc, noisy_data, I1, I2, delta_t, Delta_t)
        i = 0
        while i < numit:
            if not np.isinf(val1):
                logpihist[i] = val1
                i+=1
            val1 = pfa.run_pf_hybridtau_prop(N, V, logc, noisy_data, I1, I2, delta_t, Delta_t)
        var = np.var(logpihist)
        print(f'Tried N = {N}, var is {var}')

    print(f'N set to {N}')

    print('Setting gamma')
    gamma = 1.


    print(f'N : {N}, numit : {numit}, I1 : {I1}, I2 : {I2}')
    run_xp_pf_hybridtau(sc, numit, N, gamma, I1, I2)
    accrej = np.load(f'dat/hybrid-tau-accrej-{sc}-{numit}-{N}-{gamma}.npy')[0]
    print(f'Got out of first iteration, accrej = {accrej}')

    while accrej < .02 or accrej > .15:

        if accrej < .02:
            gamma/=2
        elif accrej > .15:
            gamma*=2
        else:
            raise Exception("this shouldn't be possible...")

        print(f'Trying gamma = {gamma}')

        print(f'sc = {sc}, accrej = {accrej}, gamma = {gamma}')

        run_xp_pf_hybridtau(sc, numit, N, gamma, I1, I2)
        accrej = np.load(f'dat/hybrid-tau-accrej-{sc}-{numit}-{N}-{gamma}.npy')[0]

    with open(f'dat/hybrid-tau-xp-dat.txt', 'a+') as file:
        file.write(f'{sc}\t200000\t{N}\t{gamma}')

    print("Completed gamma and N")

    return gamma, N


def run_xp_pf_hybridtau(sc : int, numit : int, N : int, gamma : float, I1 : int, I2 : int, load_data=True, hack_mean = False, save_data=True):

    indx = [0,1,3,4]
    logc = np.array(np.log((2,sc,1/50,1,1/(50*sc))))
    logc_hatcov = np.eye(len(indx))

    if load_data:
        fname = f'dat/hybrid-tau-logc-mean-{sc}.npy'
        if os.path.isfile(fname):
            print('Loading mean(logc_hat)...')
            logc[indx] = np.load(fname)
        else:
            print('Using perturbed value for starting guess')
            logc+= .5 * np.random.normal(size=(logc.shape[0]))

        ### load hat covariance if exists, else take id
        fname = f'dat/hybrid-tau-logc-cov-{sc}.npy'
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

    prop_func = lambda X : np.array([c1, c2, c3*X[0], c4*X[1], c5*X[0]*X[1]])

    ## hybrid algorithm stuff
    Delta_t, delta_t = .01, .1

    ### run particle filter once, pi^(y | c)
    log_pihat_y_c = pfa.run_pf_hybridtau_prop(N, V, logc, noisy_data, I1, I2, delta_t, Delta_t)
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
            print('Iteration', i_+1)

        logc_star = pfu.gaussian_proposal_logc(logc=logc, cov=gamma*logc_hatcov)

        # run particle filter with that candidate, compute pi^(y | c*)
        log_pihat_y_cstar = pfa.run_pf_hybridtau_prop(N, V, logc_star, noisy_data, I1, I2, delta_t, Delta_t)

        # accept/reject
        logc, log_pihat_y_c, change = pfa.select_new_c(logc, logc_star, log_pihat_y_c, log_pihat_y_cstar)

        logc_hist[:,i_+1] = logc[indx]
        log_pi_hist[i_+1] = log_pihat_y_c
        tally+=change

        if (i_+1) % 1000 == 0:
            np.save(f'dat/hybrid-tau-logc-dat-{sc}-{numit}-{N}', logc_hist)
            np.save(f'dat/hybrid-tau-log-pi-hist-{sc}-{numit}-{N}', log_pi_hist)

        if (i_+1) % 500 == 0:
            print('Iteration ', i_+1)
            print('Acc/rej', tally/i_)
            print('logc', logc)

    t1 = time.time()

    if save_data:
        with open(f'dat/hybrid-tau.txt', 'a+') as f:
            f.write(f'{sc}\t{numit}\t{N}\t{gamma}\t{t1 - t0}c\t{tally/numit}\n')

        np.save(f'dat/hybrid-tau-accrej-{sc}-{numit}-{N}-{gamma}', np.array([tally/numit]))
        np.save(f'dat/hybrid-tau-logc-dat-{sc}-{numit}-{N}-{gamma}', logc_hist)
        np.save(f'dat/hybrid-tau-log-pi-hist-{sc}-{numit}-{N}-{gamma}', log_pi_hist)

        np.save(f'dat/hybrid-tau-logc-mean-{sc}-{numit}-{N}-{gamma}', np.mean(logc_hist, axis=1))
        np.save(f'dat/hybrid-tau-logc-covar-{sc}-{numit}-{N}-{gamma}', np.cov(logc_hist))

        # this is a trick for burn-in..
        if hack_mean:
            np.save(f'dat/ssa-logc-mean-{sc}-{numit}-{N}-{gamma}', logc_hist[:,-1])

if __name__ == '__main__':
    ### EXPERIMENT PARAMETERS
    sc = int(sys.argv[1])
    N = int(sys.argv[2])
    gamma = float(sys.argv[3])

    I1,I2 = 20,60

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

        run_xp_pf_hybridtau(sc, drynumit, dryN, drygamma, I1, I2)
        update_mean_and_covar(sc, drynumit, dryN, drygamma)

        if verbose:
            print('Setting up N and gamma...')

        gamma, N = tune_gamma_and_N(sc, N = dryN, numit = drynumit, I1 = I1, I2 = I2)

        if verbose:
            print('Burn-in phase...')

        run_xp_pf_hybridtau(sc, drynumit, N, gamma, I1, I2, hack_mean=True)
        update_mean_and_covar(sc, drynumit, N, gamma)

        if verbose:
            print('Setup complete.')
            print(f'Assigned value for N ::', N)
            print(f'Assigned value for gamma ::', gamma)

    numit = 200000
    run_xp_pf_hybridtau(sc, numit, N, gamma, I1, I2)
