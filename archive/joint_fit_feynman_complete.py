import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["MKL_THREADING_LAYER"] = "sequential"

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
from scipy.optimize import Bounds
from multiprocessing import Pool
import emcee
from datetime import datetime
import pandas as pd
import corner
import exoplanet as xo
import time

date = datetime.now().strftime("%Y-%m-%d-%H-%M")
input_folder = '/Users/ah258874/Documents/Thèse/Punto'  # sys.argv[1]
output_location = '/Users/ah258874/Documents/Thèse/Punto/test'  # sys.argv[2]


def rv_model(parameters, t, P, T0, Mstar):
    # K in m/s
    # M_star in units of solar mass
    # P in day -> converted to years here
    # t in day
    # T0 in day
    # Mp_sini in Jupiter mass,  Mp_sini = Mp * np.sin(i)/M_jup

    Mp_sini = parameters[0]

    f = 2 * np.pi * (t - T0) / P # true anomaly for circular orbits # (2 * np.pi / P)
    K = 28.4 * ((P/365) / 1) ** (-1 / 3) * Mp_sini * Mstar ** (-2 / 3)  # Torres et al., 2008; 28.4 in m/s,
    # Perryman 2018 (eq. 2.28 and 2.29 p. 21)
    # [DO NOT USE] 203.29
    vr = -K * np.sin(f)  # K * np.cos(f)

    return vr


def phase_curve_model(parameters, time, period, T0, M_star, R_star, sini, Teff):
    # M_star in units of solar mass
    # Mp_sini in Jupiter mass,  Mp_sini = Mp * np.sin(i)/M_jup where Mp is the planet mass
    # period in day
    # T0 in day
    # time in day
    # R_star in units of solar radius
    # ceoff_ellip = alpha_ellip * np.sin(i)
    # ceoff_refl = alpha_refl * np.sin(i) * (Rp/R_jup)**2 where Rp is the planet radius and i the orbit inclination
    # Teff is the stellar effective temperature in K
    # sini is the orbital inclination

    Mp_sini = parameters[0]
    coeff_refl = parameters[1]
    log_coeff_ellip = parameters[2]
    A0 = parameters[3]
    delta = parameters[4]

    phi = (time - T0) / period

    alpha_beam = -6 * (10 ** -4) * Teff + 7.2  # Kepler bandpass, Lillo-Box, 2021 and Millholland & Laughlin 2017
    coeff_ellip = 10 ** log_coeff_ellip
    # alpha_ellip = -2.2 * (10 ** -4) * Teff + 2.6  # Kepler bandpass, Lillo-Box, 2021 and Millholland & Laughlin 2017

    Arefl = 57 * coeff_refl * M_star ** (-2 / 3) * (period / 1) ** (-4 / 3)  # Lillo-Box, 2021
    Aellip = 13 * coeff_ellip * R_star ** 3 * M_star ** (-2) * (period / 1) ** (
        -2) * Mp_sini  # Lillo-Box, 2021
    Abeam = 2.7 * alpha_beam * (period / 1) ** (-1 / 3) * M_star ** (-2 / 3) * Mp_sini  # Lillo-Box, 2021

    F = A0 - Arefl * np.cos(2 * np.pi * (phi + delta)) - Aellip * np.cos(4 * np.pi * phi) + Abeam * np.sin(
        2 * np.pi * phi)

    return F


def binning(lightcurve, bin_size):
    # Gets the remainder of the floor division between lightcurve size and bin size
    division_remainder = np.mod(len(lightcurve), bin_size)

    # We  remove the points that could  not be  part of a full bin
    tmp_data = lightcurve[division_remainder:]

    binned_lightcurve = []
    length = int(len(tmp_data) / bin_size)

    # We bin the data
    for i in range(length):
        tmp_bin = np.mean(tmp_data[(i * bin_size):((i + 1) * bin_size)])
        binned_lightcurve.append(tmp_bin)

    return np.asarray(binned_lightcurve)


def log_likelihood_phase(parameters, x, y, yerr, period, T0, M_star, R_star, sini, Teff):
    mod = phase_curve_model(parameters, x, period, T0, M_star, R_star, sini, Teff)

    sigma2 = yerr ** 2

    return -0.5 * np.sum((y - mod) ** 2 / sigma2)  # + np.log(sigma2) ) #+ np.log(2*np.pi))


def log_likelihood_rv(parameters, x, y, yerr, period, T0, M_star):
    mod = rv_model(parameters, x, period, T0, M_star)

    sigma2 = yerr ** 2

    return -0.5 * np.sum((y - mod) ** 2 / sigma2)  # + np.log(sigma2) ) #+ np.log(2*np.pi))


def log_probability(parameters, X, Y, Z, period, T0_phase, M_star, R_star, sini, Teff):
    x_phase = X[0]
    y_phase = Y[0]
    yerr_phase = Z[0]
    x_rv = X[1]
    y_rv = Y[1]
    yerr_rv = Z[1]
    params_rv = [parameters[0]]
    params_phase = parameters

    lp = log_prior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_phase(params_phase, x_phase, y_phase, yerr_phase, period, T0_phase, M_star, R_star, sini,
                                     Teff) + log_likelihood_rv(params_rv, x_rv, y_rv, yerr_rv, period, T0_phase, M_star)


def log_prior(parameters):
    Mp_sini = parameters[0]
    coeff_refl = parameters[1]
    log_coeff_ellip = parameters[2]
    A0 = parameters[3]
    delta = parameters[4]

    if 0 < Mp_sini <= 10 and \
        0 < coeff_refl <= 10 and \
        -12 <= log_coeff_ellip <= -2 and \
        - 100 <= A0 <= 100 and \
        -2*np.pi <= delta <= 2*np.pi:
            return 0.0
    return -np.inf


def phase_fold(x, y, period, t0):
    df = pd.DataFrame(np.column_stack((x, y)), columns=['t', 'f'])

    # t0 = df['t'][0]
    df['p'] = (df['t'] - t0) % period  # + 0.5 * period - 0.5 * period

    df = df.sort_values(by='p').reset_index(drop=True)

    df = df.groupby(df['p'].index).mean()

    return df['p'], df['f']


if __name__ == '__main__':

    print("=================================================================================================")
    print("                                   OUTPUT FOLDER CREATION                                        ")

    output_folder = os.path.join(output_location, 'exo' + date)
    os.makedirs(output_folder)
    print(f'Results will be saved in : {output_folder}')

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                       PLANET PROPERTIES                                         ")

    period = 0.6047  # in days
    M_star = 1.36  # in units of solar mass
    R_star = 1.54  # in units of solar radius
    i = 30  # degrees
    sini = np.sin(i * 2 * np.pi / 360)
    Teff = 6358  # stellar effective temperature in K

    print(f'Orbital period : {period} days \n'
          f'Stellar mass : {M_star} solar mass\n'
          f'Stellar radius : {R_star} solar radius \n'
          f'inclination: {i} degrees \n'
          f'Stellar effective temperature : {Teff} K \n')

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                       PHOTOMETRIC DATA                                          ")

    file_long_cadence_punto = os.path.join(input_folder, '9139163_lc_filtered.txt')

    print(f"Photometric data file is : {file_long_cadence_punto}")
    data_highpass_punto = np.loadtxt(file_long_cadence_punto)
    time_phase = binning(data_highpass_punto[:, 0], 10)
    x_phase = time_phase
    y_phase = binning(data_highpass_punto[:, 1], 10)
    T0_phase = x_phase[0] + 0.5 * period
    yerr_phase = 1

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                       RV DATA                                                   ")

    filename_rv = os.path.join(input_folder, 'KIC9139163.dat')
    print(f"RV data file is : {filename_rv}")

    data_rv = np.loadtxt(filename_rv, usecols=(0, 1, 2))
    time_rv = np.array(data_rv[:, 0])
    # x_rv = np.linspace(time_rv.min() - 5, time_rv.max() + 5, 1000)
    x_rv = time_rv
    y_rv = np.array(data_rv[:, 1])
    yerr_rv = np.array(data_rv[:, 2])

    T0_rv = x_rv[0]  # NOT USING IT

    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    ax.errorbar(x_rv - T0_rv, y_rv, yerr=yerr_rv, fmt=".", color='royalblue', markersize=1.5, linewidth=0.5,
                markerfacecolor='green', markeredgecolor='red', capsize=1, capthick=0.5)
    ax.set_xlabel("Time [BJD - {:.0f}]".format(x_rv[0]), fontsize=6)
    ax.set_ylabel("Radial velocity [m s$^{-1}$]", fontsize=6)
    ax.set_ylim(-30, 12)
    fig.tight_layout()
    ax.tick_params(labelsize=6, pad=2)
    # plt.savefig(os.path.join(output_folder, date + 'rv_data.png'))
    # plt.show()

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                       MLE ON RV DATA                                            ")

    periods = [period]
    Ks = xo.estimate_semi_amplitude(periods, x_rv, y_rv, yerr_rv, t0s=None)
    print(Ks, "m/s")
    Ms_sini = (Ks * period ** (1 / 3) * M_star ** (2 / 3)) * 4.919 * 10 ** (-3)  # in Jupiter mass
    print(Ms_sini, "Mjup sini")

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                       INITIALISATION                                            ")

    t_rv = np.linspace(x_rv.min(), x_rv.max(), 10000)
    rv_model_value = rv_model(Ms_sini, t_rv, period, T0_phase, M_star)

    fold_t_rv, fold_rv_model = phase_fold(t_rv, rv_model_value, period, T0_phase)
    fold_x_rv, fold_y_rv = phase_fold(x_rv, y_rv, period, T0_phase)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    ax.scatter(fold_x_rv, fold_y_rv, color='k', s=0.7)
    ax.plot(fold_t_rv, fold_rv_model)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, date + 'rv_data_folded.png'))
    # plt.show()

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                       MLE ON PHOTOMETRIC DATA                                   ")

    np.random.seed()

    nll = lambda *args: -log_likelihood_phase(*args)

    parameters_phase = [Ms_sini[0], 0.3, -3, -5, -1]
    print(parameters_phase)
    initial = parameters_phase

    bounds = Bounds([1 * 10**(-3), 1 * 10**(-4), -7,  -2*np.pi, -np.inf], [+np.inf, +np.inf, -1, 2*np.pi, 0])

    print('Initial values', initial)

    soln = minimize(nll, initial, args=(x_phase, y_phase, yerr_phase, period, T0_phase, M_star, R_star, sini, Teff),
                    method='Nelder-mead',
                    options={'xatol': 1e-8, 'maxiter': 5000, 'disp': True, 'adaptive': True}, bounds=bounds)

    params_mle = soln.x
    conv = soln.success
    print(f'Convergence: {conv}')
    print("Maximum likelihood estimates:")
    print(params_mle)

    fold_x_phase, fold_y_phase = phase_fold(x_phase, y_phase, period, T0_phase)
    phase_curve_model_value = phase_curve_model(params_mle, x_phase, period, T0_phase, M_star, R_star, sini, Teff)
    fold_x_phase, fold_phase_curve_model = phase_fold(x_phase, phase_curve_model_value, period, T0_phase)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    ax.scatter(fold_x_phase, fold_y_phase, color='k', s=0.7)
    ax.scatter(fold_x_phase, fold_phase_curve_model, color='r', s=0.7)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, date + 'MLE_phasecurve.png'))
    plt.show()

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                        RESIDUALS ANALYSIS                                       ")

    residuals_phase = phase_curve_model(params_mle, x_phase, period, T0_phase, M_star, R_star, sini,
                                        Teff) - phase_curve_model(parameters_phase, x_phase, period, T0_phase, M_star,
                                                                  R_star, sini, Teff)
    rms = np.sqrt(np.std(residuals_phase) ** 2 + np.mean(residuals_phase) ** 2)
    print(f"Residuals calculation, RMS mean is: {rms}")

    print("=================================================================================================")
    print("                                                                                                 ")
    print("=================================================================================================")
    print("                                               MCMC                                              ")

    yerr_phase = np.zeros_like(y_phase) + rms

    joint_parameters = np.array((params_mle[0], params_mle[1], params_mle[2], params_mle[3], params_mle[4]))

    # print(f"Used parameters in the right order: "
    #       f"Planetary mass time the sinus of the inclination"
    #       f"Reflexion coefficient"
    #       f"A0 amplitude"
    #       f"Phase shift")
    print(f"Parameters first guess : {joint_parameters}")

    a = log_prior(joint_parameters)
    print(f"Log prior value is : {a}")
    # Using the soln.x result from MLE to initialize MCMC

    pos = joint_parameters + 1e-4 * np.random.randn(100, 5)
    nwalkers, ndim = pos.shape
    print(pos)
    print(f"Nwalkers, Ndim: {nwalkers}, {ndim}")

    X = [x_phase, x_rv]
    Y = [y_phase, y_rv]
    Z = [yerr_phase, yerr_rv]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(X, Y, Z, period, T0_phase, M_star, R_star, sini, Teff), pool=pool)
        start = time.time()
        max_n = 5000
        print(f"Number of iterations is {max_n}")
        sampler.run_mcmc(pos, max_n, progress=True)

        # We'll track how the average autocorrelation time estimate changes
        # index = 0
        # autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        # old_tau = np.inf

        # Now we'll sample for up to max_n steps
        # for sample in sampler.sample(pos, iterations=max_n, progress=False):
            # Only check convergence every 100 steps
            # if sampler.iteration % 100:  # 1000
                # continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            # tau = sampler.get_autocorr_time(tol=0)
            # autocorr[index] = np.mean(tau)
            # index += 1

            # Check convergence
            # converged = np.all(tau * 100 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            # if converged:
                # break
            # old_tau = tau

        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    print('Gets the chain')
    # samples = sampler.get_chain()

    # tau = sampler.get_autocorr_time()
    # print(f'Autocorrelation time is tau[Ndim]: {tau}')
    # burnin = int(2 * np.max(tau))
    # thin = int(0.5 * np.min(tau))
    # print(f'Burnin is {burnin}, Thin is {thin}')

    # flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    np.savetxt(os.path.join(output_folder, f"flat_samples_{date}.txt"), flat_samples)
    print('Flat samples saved')

    # print("=================================================================================================")
    # print("                                                                                                 ")
    # print("=================================================================================================")
    # print("                                    AUTOCORRELATION TIME ESTATE                                  ")
    #
    # n = 1000 * np.arange(1, index + 1)
    # corr = autocorr[:index]
    #
    # plt.plot(n, n / 100.0, "--k")
    # plt.plot(n, corr)
    # plt.xlim(0, n.max())
    # plt.ylim(0, corr.max() + 0.1 * (corr.max() - corr.min()))
    # plt.xlabel("number of steps")
    # plt.ylabel(r"mean $\hat{\tau}$")
    # plt.title('Autocorrelation time estate')
    # plt.savefig(os.path.join(output_folder, date + 'autocorrelation_time.png'))

    print("                                                MCMC DONE                                            ")

    fig, axes = plt.subplots(5, figsize=(5, 4), sharex=True)
    labels = ["Mp_sini", "coeff_refl", "coeff_ellip", "A0", "Delta"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(flat_samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, len(flat_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    fig.suptitle('Burnt-in discarded, chain thinned and flattened')
    # plt.savefig(os.path.join(output_folder, date + 'flat_chain.png'))
    plt.show()

    fig = corner.corner(
        flat_samples, labels=labels, truths=joint_parameters
    )
    # plt.savefig(os.path.join(output_folder, date + 'corner.png'))
    plt.show()

    inds = np.random.randint(len(flat_samples), size=200)

    # PLOT PHASE CURVE RESULTS

    fig, ax = plt.subplots(figsize=(5, 4))
    for ind in inds:
        sample = flat_samples[ind]
        best_fit_phase = phase_curve_model(sample, x_phase, period, T0_phase, M_star, R_star, sini, Teff)
        fold_x_phase, fold_best_fit_phase = phase_fold(x_phase, best_fit_phase, period, T0_phase)
        ax.plot(fold_x_phase, fold_best_fit_phase, alpha=0.1)
    ax.scatter(fold_x_phase, fold_y_phase, color='k', s=0.7)
    ax.set_xlabel("Time in days")
    ax.set_ylabel("Flux in MJy");
    plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, date + 'best_fit_phasecurve.png'))
    plt.show()

    # PLOT RV RESULTS

    fig, ax = plt.subplots(figsize=(5, 4))
    for ind in inds:
        sample = flat_samples[ind]
        best_fit_rv = rv_model([sample[0]], x_rv, period, T0_rv, M_star)
        fold_x_rv, fold_best_fit_rv = phase_fold(x_rv, best_fit_rv, period, T0_phase)
        ax.plot(fold_x_rv, fold_best_fit_rv, alpha=0.1)
    ax.scatter(fold_x_rv, fold_y_rv, color='k', s=0.7)
    ax.set_xlabel("Time in days")
    ax.set_ylabel("RV is m/s");
    plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, date + 'best_fit_rv.png'))
    plt.show()

    print("FINISHED")



