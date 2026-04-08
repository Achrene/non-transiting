import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["MKL_THREADING_LAYER"] = "sequential"

import numpy as np
import matplotlib.pyplot as plt
# import sys
from scipy.optimize import minimize
from scipy.optimize import Bounds
# from multiprocessing import Pool
# import emcee
import nestle
from datetime import datetime
import pandas as pd
import corner
import exoplanet as xo
import time

date = datetime.now().strftime("%Y-%m-%d-%H-%M")
input_folder = '/Users/ah258874/Documents/Thèse/Punto'  # sys.argv[1]
output_location = '/Users/ah258874/Documents/Thèse/Punto/test'  # sys.argv[2]

period = 0.6047  # in days
M_star = 1.36  # in units of solar mass
R_star = 1.54  # in units of solar radius
i = 30  # degrees
sini = np.sin(i * 2 * np.pi / 360)
Teff = 6358  # stellar effective temperature in K


class Model(object):
    def __init__(self, time_phase, time_rv, period, T0, M_star, R_star, sini, Teff):
        self.time_phase = time_phase
        self.time_rv = time_rv
        self.period = period
        self.T0 = T0
        self.M_star = M_star
        self.R_star = R_star
        self.sini = sini
        self.Teff = Teff

    def model_rv(self, parameters):
        # K in m/s
        # M_star in units of solar mass
        # P in day -> converted to years here
        # t in day
        # T0 in day
        # Mp_sini in Jupiter mass,  Mp_sini = Mp * np.sin(i)/M_jup

        Mp_sini = parameters[0]

        f = 2 * np.pi * (self.time_rv - self.T0) / self.period  # true anomaly for circular orbits # (2 * np.pi / P)
        K = 28.4 * ((self.period / 365) / 1) ** (-1 / 3) * Mp_sini * self.M_star ** (-2 / 3)
        # Torres et al., 2008; 28.4 in m/s,
        # Perryman 2018 (eq. 2.28 and 2.29 p. 21)
        # [DO NOT USE] 203.29
        vr = -K * np.sin(f)  # K * np.cos(f)

        return vr

    def model_photometry(self, parameters):
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

        # Mp_sini = parameters[0]
        coeff_refl = parameters[0]
        # coeff_ellip = parameters[2]
        A0 = parameters[1]
        delta = parameters[2]

        phi = (self.time_phase - self.T0) / self.period

        # alpha_beam = -6 * (10 ** -4) * self.Teff + 7.2
        # Kepler bandpass, Lillo-Box, 2021 and Millholland & Laughlin 2017

        # alpha_ellip = -2.2 * (10 ** -4) * Teff + 2.6
        # Kepler bandpass, Lillo-Box, 2021 and Millholland & Laughlin 2017

        Arefl = 57 * coeff_refl * self.M_star ** (-2 / 3) * (self.period / 1) ** (-4 / 3)  # Lillo-Box, 2021
        # Aellip = 13 * coeff_ellip * self.R_star ** 3 * self.M_star ** (-2) * (self.period / 1) ** (-2) * Mp_sini
        # Lillo-Box, 2021

        # Abeam = 2.7 * alpha_beam * (self.period / 1) ** (-1 / 3) * self.M_star ** (-2 / 3) * Mp_sini  # Lillo-Box, 2021

        F = A0 - Arefl * np.cos(2 * np.pi * (phi + delta)) # - Aellip * np.cos(4 * np.pi * phi) + Abeam * np.sin(
            #2 * np.pi * phi)

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


def phase_fold(x, y, period, t0):
    df = pd.DataFrame(np.column_stack((x, y)), columns=['t', 'f'])

    # t0 = df['t'][0]
    df['p'] = (df['t'] - t0) % period  # + 0.5 * period - 0.5 * period

    df = df.sort_values(by='p').reset_index(drop=True)

    df = df.groupby(df['p'].index).mean()

    return df['p'], df['f']


file_long_cadence_punto = os.path.join(input_folder, '9139163_lc_filtered.txt')

print(f"Photometric data file is : {file_long_cadence_punto}")
data_highpass_punto = np.loadtxt(file_long_cadence_punto)
time_phase = binning(data_highpass_punto[:, 0], 50)
x_phase = time_phase
y_phase = binning(data_highpass_punto[:, 1], 50)
T0_phase = x_phase[0] + 0.5 * period
yerr_phase = 25

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

model = Model(time_phase, time_rv, period, T0_phase, M_star, R_star, sini, Teff)

rv_model_value = model.model_rv(Ms_sini)

# fold_t_rv, fold_rv_model = phase_fold(t_rv, rv_model_value, period, T0_phase)
# fold_x_rv, fold_y_rv = phase_fold(x_rv, y_rv, period, T0_phase)
#
# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# ax.scatter(fold_x_rv, fold_y_rv, color='k', s=0.7)
# ax.plot(fold_t_rv, fold_rv_model)
# ax.tick_params(labelsize=5)
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, date + 'rv_data_folded.png'))
# plt.show()


def log_likelihood_phase(parameters):
    mod = model.model_photometry(parameters)

    sigma2 = yerr_phase ** 2

    return -0.5 * np.sum((y_phase - mod) ** 2 / sigma2)  # + np.log(sigma2) ) #+ np.log(2*np.pi))


def log_likelihood_rv(parameters):
    mod = model.model_rv(parameters)

    sigma2 = yerr_rv ** 2

    return -0.5 * np.sum((y_rv - mod) ** 2 / sigma2) # + np.log(sigma2) ) + np.log(2*np.pi))


def log_probability(parameters):

    return log_likelihood_phase(parameters) #+ log_likelihood_rv(parameters)


def prior_transform(parameters):
    #np.array([10. * parameters[0], 10. * parameters[1], 10. * parameters[2], 200. * parameters[3] - 100.,
              #4 * np.pi * parameters[4] - 2 * np.pi])

    return np.array([10. * parameters[0], 200. * parameters[1] - 100.,
                     4 * np.pi * parameters[2] - 2 * np.pi])


print("=================================================================================================")
print("                                                                                                 ")
print("=================================================================================================")
print("                                       MLE ON PHOTOMETRIC DATA                                   ")

np.random.seed()

nll = lambda *args: -log_likelihood_phase(*args)

parameters_phase = [0.3, -5, -1]
print(parameters_phase)
initial = parameters_phase

bounds = Bounds([1 * 10 ** (-4), -2 * np.pi, -np.inf],
                [+np.inf, 2 * np.pi, 0])

print('Initial values', initial)

soln = minimize(nll, initial,
                method='Nelder-mead',
                options={'xatol': 1e-8, 'maxiter': 5000, 'disp': True, 'adaptive': True}, bounds=bounds)

params_mle = soln.x
conv = soln.success
print(f'Convergence: {conv}')
print("Maximum likelihood estimates:")
print(params_mle)

# fold_x_phase, fold_y_phase = phase_fold(x_phase, y_phase, period, T0_phase)
# phase_curve_model_value = model.model_photometry(params_mle)
# fold_x_phase, fold_phase_curve_model = phase_fold(x_phase, phase_curve_model_value, period, T0_phase)
#
# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# ax.scatter(fold_x_phase, fold_y_phase, color='k', s=0.7)
# ax.scatter(fold_x_phase, fold_phase_curve_model, color='r', s=0.7)
# ax.tick_params(labelsize=5)
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, date + 'MLE_phasecurve.png'))
# plt.show()

print("=================================================================================================")
print("                                                                                                 ")
print("=================================================================================================")
print("                                        RESIDUALS ANALYSIS                                       ")

residuals_phase = model.model_photometry(params_mle) - model.model_photometry(parameters_phase)

rms = np.sqrt(np.std(residuals_phase) ** 2 + np.mean(residuals_phase) ** 2)
print(f"Residuals calculation, RMS mean is: {rms}")

yerr_phase = np.zeros_like(y_phase) + rms

joint_parameters = np.array((params_mle[0], params_mle[1], params_mle[2]))

# print(f"Used parameters in the right order: "
#       f"Planetary mass time the sinus of the inclination"
#       f"Reflexion coefficient"
#       f"A0 amplitude"
#       f"Phase shift")
print(f"Parameters first guess : {joint_parameters}")


print("=================================================================================================")
print("                                                                                                 ")
print("=================================================================================================")
print("                                    NESTED SAMPLING                                              ")

start = time.time()

res = nestle.sample(log_probability, prior_transform, 3, method='multi', maxiter = None,
                    npoints=600, callback=nestle.print_progress)
print(res.summary())

end = time.time()
multi_time = end - start
print("Nested sampling took {0:.1f} seconds".format(multi_time))

print('Gets the chain')
# samples = sampler.get_chain()

# flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
flat_samples = res.samples
weights = res.weights
# weighted average and covariance:
p, cov = nestle.mean_and_cov(flat_samples, weights)

# np.savetxt(os.path.join(output_folder, f"flat_samples_{date}.txt"), flat_samples)
# print('Flat samples saved')

print("                                          NESTED SAMPLING DONE                                            ")

labels = ["coeff_refl", "A0", "Delta"]
fig = corner.corner(flat_samples, weights=weights, labels=labels, truths=joint_parameters, bins=30)
plt.show()

ndim = 3

fig, axes = plt.subplots(3, figsize=(5, 4), sharex=True)
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

ppp, aaa = np.shape(flat_samples)
inds = 100 # np.random.randint(len(flat_samples), size=200)
fold_x_phase, fold_y_phase = phase_fold(x_phase, y_phase, period, T0_phase)

# PLOT PHASE CURVE RESULTS

fig, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(fold_x_phase, fold_y_phase, yerr = yerr_phase, marker='.', color='k',linestyle='None', markersize=4, alpha=0.7)
for ind in range(inds):
    sample = flat_samples[ppp-(ind+1), :]
    best_fit_phase = model.model_photometry(sample)
    fold_x_phase, fold_best_fit_phase = phase_fold(x_phase, best_fit_phase, period, T0_phase)
    ax.plot(fold_x_phase, fold_best_fit_phase, alpha=0.1)
 # color='k', s=0.7
ax.set_xlabel("Time in days")
ax.set_ylabel("Flux in MJy");
plt.tight_layout()
# plt.savefig(os.path.join(output_folder, date + 'best_fit_phasecurve.png'))
plt.show()

# PLOT RV RESULTS

fold_x_rv, fold_y_rv = phase_fold(x_rv, y_rv, period, T0_phase)

fig, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(fold_x_rv, fold_y_rv, yerr=yerr_rv,  marker='.', color='k',linestyle='None', markersize=4, alpha=0.5)
for ind in range(inds):
    sample = flat_samples[ppp-(ind+1), :]
    best_fit_rv = model.model_rv(sample)
    fold_x_rv, fold_best_fit_rv = phase_fold(x_rv, best_fit_rv, period, T0_phase)
    ax.plot(fold_x_rv, fold_best_fit_rv, alpha=0.1)
# ax.errorbar(fold_x_rv, fold_y_rv, yerr=yerr_rv,  marker='.', color='k',linestyle='None', markersize=4, zorder=1) # color='k', s=0.7
ax.set_xlabel("Time in days")
ax.set_ylabel("RV is m/s");
plt.tight_layout()
# plt.savefig(os.path.join(output_folder, date + 'best_fit_rv.png'))
plt.show()

print("FINISHED")



