import numpy as  np
import os
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.lines as lines
import json
import corner
import ultranest
from ultranest.stepsampler import SliceSampler, generate_region_oriented_direction
import sys
import utils

import cmocean as cm

from matplotlib import rc
rc('image', origin='lower')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 16})
rc('text', usetex=False)
rc('lines', linewidth=0.5)
rc('ytick', right=True, direction = 'in')
rc('xtick', top = True, direction = 'in')
rc('axes', axisbelow = False)
rc('mathtext', fontset = 'cm')

colors_matter = cm.cm.matter_r(np.linspace(0,3,10))


#  inputs
targetname = '9139163'
period = 0.604734  # in days
nphase = 100  # discretization of phases
phase_model = np.linspace(0, 360., nphase)
wavearray = np.array([0.430, 0.890])  # Kepler bandpass in micron
planetarymasssini = 11.4  # Earth mass, from RV fit
planetarymasssini = planetarymasssini * 0.00314558  # conversion into Jupiter mass
inclination = 17  # degrees
model_name = f'model_grid_phase_offset'
effectivetemperature = 6358
stellarmass = 1.390
stellarradius = 1.558

plot_grid = False
perform_fit = True
fit_gravitational_effects = False
add_gravitational_effects_model = True

if fit_gravitational_effects is True and add_gravitational_effects_model is True:
    print("We can't fit and fix the gravitational effects at the same time. Turn one of them to False.")
    sys.exit()

references_path = '/Users/adyrek/PycharmProjects/punto/references'
model_grid_folder = '/Users/adyrek/PycharmProjects/punto/results_grid/model_grid/phase_offset'
main_result_folder= '/Users/adyrek/PycharmProjects/punto/results_grid/ultranest/freefit'
if fit_gravitational_effects is True:
    result_folder = os.path.join(main_result_folder, 'phase_offset/fitted_grav_params')
else:
    if add_gravitational_effects_model is True:
        result_folder = os.path.join(main_result_folder, 'phase_offset/fixed_grav_params')
    else:
        result_folder = os.path.join(main_result_folder, 'phase_offset')


print("Reading the model grid.")
# Load the saved grid
model_grid = np.load(os.path.join(model_grid_folder, f"{model_name}.npz"))

planetaryradius = model_grid["planetaryradius"]
albedo = model_grid["albedo"]
redistribution = model_grid["redistribution"]
inclination_grid = model_grid["inclination"]
inclination_grid = np.asarray(inclination_grid)
albedo_min = model_grid["albedo_min"]
albedo_min = np.asarray(albedo_min)
cloud_offset = model_grid["cloud_offset"]
cloud_offset = np.asarray(cloud_offset)
normalized_flux = model_grid["flux"]

# load metadata
with open(os.path.join(model_grid_folder, f"{model_name}.json"), "r") as f:
    metadata = json.load(f)

# Example: Access metadata
print("Author:", metadata["author"])
print("Date created:", metadata["date_created"])
print("Fixed parameters:", metadata["fixed_parameters"])
print("Param1 name:", metadata["planetaryradius"]["name"])
print("Param1 unit:", metadata["planetaryradius"]["unit"])
print("Param2 name:", metadata["albedo"]["name"])
print("Param2 unit:", metadata["albedo"]["unit"])
print("Param3 name:", metadata["redistribution"]["name"])
print("Param3 unit:", metadata["redistribution"]["unit"])
print("Param4 name:", metadata["inclination"]["name"])
print("Param4 unit:", metadata["inclination"]["unit"])
print("Param5 name:", metadata["albedo_min"]["name"])
print("Param5 unit:", metadata["albedo_min"]["unit"])
print("Param6 name:", metadata["cloud_offset"]["name"])
print("Param6 unit:", metadata["cloud_offset"]["unit"])

if plot_grid is True:
    print("'plot_grid' is True: plotting the grid.")

    def plot_photometry_model(normalized_flux, planetaryradius, albedo, redistribution,
                              inclination, albedo_min, cloud_offset):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)
        for angle in [0, 90, 180, 270, 360]:
            ax.axvline(angle, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)

        n_r, n_a, n_rcl, n_inc, n_amin, n_c, n_phase = normalized_flux.shape
        total_curves = n_r * n_a * n_rcl * n_inc * n_amin * n_c

        flux_reshaped = normalized_flux.reshape((total_curves, n_phase))
        i_vals, j_vals, k_vals, l_vals, m_vals, n_vals = np.meshgrid(
            np.arange(n_r),
            np.arange(n_a),
            np.arange(n_rcl),
            np.arange(n_inc),
            np.arange(n_amin),
            np.arange(n_c),
            indexing='ij'
        )

        i_flat = i_vals.flatten()
        j_flat = j_vals.flatten()
        k_flat = k_vals.flatten()
        l_flat = l_vals.flatten()
        m_flat = m_vals.flatten()
        n_flat = n_vals.flatten()

        # Loop over flattened index arrays
        for idx in range(total_curves):
            i, j, k, l, m, n = i_flat[idx], j_flat[idx], k_flat[idx], l_flat[idx], m_flat[idx], n_flat[idx]
            flux_curve = flux_reshaped[idx]
            if not np.all(np.isnan(flux_curve)):
                ax.plot(
                    phase_model,
                    flux_curve,
                    linewidth=2,
                    linestyle='-',
                    color=colors_matter[i]
                )

        # Axis formatting
        xticks = [0, 90, 180, 270, 360]
        xlabels = [f'{x}°' for x in xticks]
        ax.set_xticks(xticks, labels=xlabels)
        ax.set_xlabel(r'Phase [$^{\circ}$]')
        ax.set_ylabel(r'Planet-star contrast $F_p / F_{\star} \times 10^6$ [ppm]')
        legend_text = (
            f"$R_\\mathrm{{p}}$ = [{np.min(planetaryradius):.2f}, {np.max(planetaryradius):.2f}] $R_\\mathrm{{J}}$\n"
            f"A = [{np.min(albedo):.2f}, {np.max(albedo):.2f}]\n"
            f"Redist. = [{np.min(redistribution):.2f}, {np.max(redistribution):.2f}]\n"
            f"i. = [{np.min(inclination_grid):.2f}, {np.max(inclination_grid):.2f}]\n"
            f"A min. = [{np.min(albedo_min):.2f}, {np.max(albedo_min):.2f}]\n"
            f"dC. = [{np.min(cloud_offset):.2f}, {np.max(cloud_offset):.2f}]\n"
            
            f"Discretisation = [{n_r}, {n_a}, {n_rcl}, {n_inc}, {n_amin}, {n_c}]\n]"
        )
        ax.text(0.65, 0.98, legend_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))
        plt.tight_layout()
        plt.savefig(os.path.join(model_grid_folder, f'{model_name}.png'),
                    format='png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_photometry_model(normalized_flux, planetaryradius, albedo, redistribution)

else:
    pass


# Open data
input_file = '/Users/adyrek/Documents/STScI_fellowship/Punto/Kepler_long_cadence/9139163_lc_filtered.txt'

data = np.loadtxt(input_file)
time_array = data[:, 0]
y = data[:, 1]
t0_RV = 2459664.752 #from the RV fit
t0_Kepler = 2454953.5394706232
ref_time = t0_RV - t0_Kepler

# get the errorbars
referencetimefiename = 't0_estimation.csv'
referencetimedata = pd.read_csv(os.path.join(references_path, referencetimefiename))
target =referencetimedata.loc[referencetimedata['KIC'] == int(targetname)]
repetitions = len(time_array)
yerr = np.repeat(np.asarray(target['noise_jenkins'])[0], repetitions)
#  yerr = 1

# Phase-fold
foldx, foldy = utils.phase_fold(time_array, y, period, ref_time)

# modify observation phase array: foldx to get it in phase between 0-360
phase_obs = ((foldx + 0.5 * period) / period) * 360

# interpolate over radius, albedo, redistribution, inclination, albedo_min and phase_offset
model_interpolator = RegularGridInterpolator((planetaryradius, albedo, redistribution, inclination, albedo_min,
                                              cloud_offset),
                                             normalized_flux,
                                             bounds_error=False,
                                             fill_value=None)


alpha_ellip = -2.2 * (10 ** -4) * effectivetemperature + 2.6
alpha_beam = -6 * (10 ** -4) * effectivetemperature + 7.2
Aellip_fixed = 13 * alpha_ellip * np.sin(inclination * np.pi / 180) * (stellarradius ** 3) * \
         (stellarmass ** (-2)) * (period ** (-2)) * planetarymasssini
Abeam_fixed = 2.7 * alpha_beam * (period ** (-1 / 3)) * (stellarmass ** (-2 / 3)) * planetarymasssini
print(f'Aellip is {Aellip_fixed} ppm.')
print(f'Abeam is {Abeam_fixed} ppm.')

def make_model_flux_function(model_interpolator, phase_model, phase_obs, fit_gravitational_effects=False,
                             add_gravitational_effects_model=False, precomputed_Abeam=None, precomputed_Aellip=None):

    # Precompute sinusoidal terms (since phase_obs is fixed)
    cos_term = np.cos(4 * np.pi * phase_obs)
    sin_term = np.sin(2 * np.pi * phase_obs)

    def get_model_flux(p):

        if fit_gravitational_effects:
            offset = p[-1]
            Aellip, Abeam = p[-2], p[-3]
            model_params = p[:-3]
        else:
            offset = p[-1]
            model_params = p[:-1]

        #  Interpolate base model flux
        pts = np.atleast_2d(model_params)  # shape (1, ndim)
        flux_grid_res = model_interpolator(pts)  # likely shape (1, n_phase)
        flux_grid = np.squeeze(flux_grid_res)  # shape (n_phase,) guaranteed

        flux_interp = interp1d(phase_model, flux_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
        flux_interp_squeeze = flux_interp(phase_obs)

        flux = flux_interp_squeeze + offset

        # --- Add gravitational effects if needed ---
        if fit_gravitational_effects:
            flux -= Aellip * cos_term
            flux += Abeam * sin_term

        elif add_gravitational_effects_model:
            # use precomputed fixed amplitudes
            flux -= precomputed_Aellip * cos_term
            flux += precomputed_Abeam * sin_term

        return flux

    return get_model_flux


ultranest_output = os.path.join(main_result_folder, f'ultranestoutput/i{inclination}')
os.makedirs(ultranest_output, exist_ok=True)

# Print Aellip and Abeam
alpha_ellip = -2.2 * (10 ** -4) * effectivetemperature + 2.6
alpha_beam = -6 * (10 ** -4) * effectivetemperature + 7.2
Aellip_fixed = 13 * alpha_ellip * np.sin(inclination * np.pi / 180) * (stellarradius ** 3) * \
         (stellarmass ** (-2)) * (period ** (-2)) * planetarymasssini
Abeam_fixed = 2.7 * alpha_beam * (period ** (-1 / 3)) * (stellarmass ** (-2 / 3)) * planetarymasssini
print(f'Aellip is {Aellip_fixed} ppm.')
print(f'Abeam is {Abeam_fixed} ppm.')

get_model_flux = make_model_flux_function(
        model_interpolator,
        phase_model,
        phase_obs,
        fit_gravitational_effects=fit_gravitational_effects,
        add_gravitational_effects_model=add_gravitational_effects_model,
        precomputed_Abeam=Abeam_fixed,
        precomputed_Aellip=Aellip_fixed,
    )

if perform_fit is True:
    print("'perform_fit' is True: performing fit using UltraNest Nested Sampling.")

    # bounds
    if fit_gravitational_effects is False:
        param_names = ["Planetary Radius", "Albedo", "Redistribution", "Albedo min", "Cloud offset", "Offset"]
        param_bounds = [
            (planetaryradius.min(), planetaryradius.max()),
            (albedo.min(), albedo.max()),
            (redistribution.min(), redistribution.max()),
            (albedo_min.min(), albedo_min.max()),
            (cloud_offset.min(), cloud_offset.max()),
            (-5 * np.mean(yerr), 5 * np.mean(yerr))]
    else:
        param_names = ["Planetary Radius", "Albedo", "Redistribution", "Albedo min",
                       "Cloud offset", "Abeam", "Aellip", "Offset"]
        param_bounds = [
            (planetaryradius.min(), planetaryradius.max()),
            (albedo.min(), albedo.max()),
            (redistribution.min(), redistribution.max()),
            (albedo_min.min(), albedo_min.max()),
            (cloud_offset.min(), cloud_offset.max()),
            (1e-2, +500),
            (-500, + 500),
            (-5 * np.mean(yerr), 5 * np.mean(yerr))]


    def make_prior_transform(param_names, param_bounds):
        def prior_transform(unit_cube):
            params = []
            for i, (u, (low, high)) in enumerate(zip(unit_cube, param_bounds)):
                if param_names[i] in ["Abeam", "Aellip"]:
                    # Log-scale these to improve sampling stability
                    val = 10 ** (np.log10(max(low, 1e-6)) + u[i] * (np.log10(max(high, 1e-6))
                                                                    - np.log10(max(low, 1e-6))))

                elif param_names[i] == "Albedo":
                    val = low + u * (high - low)
                else:
                    val = low + u * (high - low)
                val = np.clip(val, low, high)
                params.append(val)
            return np.array(params)
        return prior_transform


    def log_likelihood(p):
        model_flux = get_model_flux(p)
        residuals = (foldy - model_flux) / np.clip(yerr, 1e-12, np.inf)

        loglike = -0.5 * np.sum(residuals ** 2 + np.log(2 * np.pi * yerr ** 2))
        if np.isnan(loglike) or np.isinf(loglike):
            return -1e300
        return loglike

    prior_transform = make_prior_transform(param_names, param_bounds)
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=ultranest_output)

    # Assign a step-based sampler to improve exploration efficiency
    sampler.stepsampler = SliceSampler(
        nsteps=10,
        generate_direction=generate_region_oriented_direction
    )

    # run nested sampling
    result = sampler.run(min_num_live_points=400, dlogz=0.5, frac_remain=1e-8, max_iters=40000)
    sampler.print_results()

    samples = result['samples']

    best_fit_params = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    # Get best-fit model flux for plotting
    best_model = get_model_flux(best_fit_params)

    print("\nBest-fit parameters:")
    print("\nPosterior mean ± std:")
    for i, name in enumerate(param_names):
        print(f"{name} ± σ: {best_fit_params[i]:.3f} ± {std[i]:.3f}")
    # Save results
    np.savez_compressed(os.path.join(result_folder, f"fit_results_ultranest_i{inclination}.npz"),
                        best_fit_params=best_fit_params,
                        param_names=param_names,
                        sigmas=std,
                        best_model=best_model,
                        samples=samples,
                        ultranest_result=result)

else:
    print("'perform_fit' is False: reading file.")

    fit_results = np.load(os.path.join(result_folder, f"fit_results_ultranest_i{inclination}.npz"))
    best_fit_params = fit_results["best_fit_params"]
    std = fit_results["std"]
    best_model = fit_results["best_model"]
    param_names = fit_results["param_names"]

    print("\nBest-fit parameters:")
    print("\nPosterior mean ± std:")
    for i, name in param_names:
        print(f"{name} ± σ: {best_fit_params[i]:.3f} ± {std[i]:.3f}")

# Plot the fit

fig = corner.corner(samples, labels=param_names,
                    truths=best_fit_params, show_titles=True, title_fmt='.5f', title_kwargs={"fontsize": 12})
plt.savefig(os.path.join(result_folder, f'corner_plot_ultranest_i{inclination}.png'),
            format='png', dpi=300, bbox_inches='tight')
plt.show()

resi = best_model - foldy
binsize = 50
binned_residuals, res_err = utils.bin_data(resi, binsize, err=None)
binned_time, time_err = utils.bin_data(phase_obs, binsize, err=None)
binned_flux, binned_err = utils.bin_data(foldy, binsize, err=yerr)
bin_time = utils.compute_binning_time(time_array, binsize)

fig, ax = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, dpi=100)
ax[0].plot(phase_obs / 360, best_model, color=colors_matter[3], linewidth=1.3, zorder=3, alpha=1)
ax[0].errorbar(phase_obs / 360, foldy, yerr=yerr, linestyle=' ', fmt='.',
               color='lightgrey', zorder=1,
               alpha=0.5)
ax[0].errorbar(binned_time / 360, binned_flux, yerr=binned_err, linestyle=' ', fmt='.',
               color=colors_matter[1], zorder=2,
               alpha=0.7)
ax[0].set_ylabel('Normalised lightcurve [ppm]')
full_data_label = lines.Line2D([], [], color='lightgrey',
                         markersize=3, marker='.', label='full data')
fit_label = lines.Line2D([], [], color=colors_matter[3],
                          markersize=3, label=f'best fit model (i = {inclination}$^\circ$)')
binned_data_label = lines.Line2D([], [], color=colors_matter[1],
                          markersize=3, label=f'{int(np.round(bin_time, 0))} day binned data')
ax[0].legend(handles=[full_data_label, fit_label, binned_data_label], fontsize=13, loc="upper right")
ax[0].set_xlim(0, 1)

ax[1].scatter(phase_obs / 360, resi,
              alpha=0.5, color='lightgrey', s=8, marker='o')
ax[1].scatter(binned_time / 360, binned_residuals,
              alpha=0.7, color=colors_matter[1], s=8, marker='o')
ax[1].axhline(y=0, xmin=0, xmax=360, linewidth=1.3, color=colors_matter[3])
ax[1].set_xlabel('Orbital phase')  # Time [day]
ax[1].set_ylabel('Residuals [ppm]')
ax[1].set_xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(result_folder, f'best_fit_ultranest_i{inclination}.png'),  format='png', dpi=300,
            bbox_inches='tight')
plt.show()


print('done')






