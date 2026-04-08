import numpy as  np
from numpy.linalg import inv
import os
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
import matplotlib.lines as lines
from datetime import datetime
import json
import seaborn as sns
import sys

from model import utils
import ExoplanetarySystem

import cmocean as cm

from matplotlib import rc
rc('image', origin='lower')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 16})
rc('text', usetex=True)
rc('lines', linewidth=0.5)
rc('ytick', right=True, direction = 'in')
rc('xtick', top = True, direction = 'in')
rc('axes', axisbelow = False)
rc('mathtext', fontset = 'cm')

colors_matter = cm.cm.matter_r(np.linspace(0,3,10))


@contextmanager
def tqdm_joblib(tqdm_object):
    from joblib import parallel
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

#  inputs
targetname = '9139163'
period = 0.604734  # in days
wavearray = np.array([0.430, 0.890])  # Kepler bandpass in micron
nphase = 100  # discretization of phases
phase_model = np.linspace(0, 360., nphase)

planetarymasssini = 0.0078  # Jupiter mass sin i
inclination = 62  # degrees
sini = np.sin(inclination * np.pi / 180)
planetarymass = planetarymasssini / sini  # Jupiter mass

references_path = '/Users/ah258874/PycharmProjects/punto/references'
model_grid_folder = '/Users/ah258874/PycharmProjects/punto/results_grid/model_grid'
result_folder = '/Users/ah258874/PycharmProjects/punto/results_grid/least_squares'



create_grid = False
plot_grid = False
perform_fit = True

if create_grid is True:
    if os.path.exists(os.path.join(model_grid_folder, 'grid_model.npz')):
        print('"create_grid" is True: a grid file already exists. Please delete or remove it from the '
              'folder to proceed.')
        sys.exit()
    else:
        print("'create_grid' is True: creating the model grid.")

    if plot_grid is False:
        print('"create_grid" is True and "plot_grid" is False: the grid will not be plotted.')

    else:
        pass

    #  free params
    planetaryradius = np.linspace(0.1, 1, 10)
    albedo = np.linspace(0.01, 0.5, 10)
    redistribution = np.linspace(0.01, 0.5, 10)

    # creating the grid of model
    def run_model(planetaryradius, albedo, redistribution):
        punto_system = ExoplanetarySystem.ExoplanetarySystem(orbitalperiod=period, effectivetemperature=6358,
                                                             stellarmass=1.390,
                                                             stellarradius=1.558,
                                                             semimajoraxis=None, planetaryradius=planetaryradius,
                                                             planetarymass=planetarymass, inclination=inclination,
                                                             redistribution=np.round(redistribution, 1),
                                                             albedo=np.round(albedo, 1),
                                                             wavearray=wavearray,
                                                             longitudearray=None, latitudearray=None, checking=False,
                                                             internaltemperature=100, area=None, phase=phase_model,
                                                             atmospherictemperature=None,
                                                             totalplanetartintensity=None,
                                                             emittedplanetaryintensity=None,
                                                             reflectedplanetartintensity=None,
                                                             contrast_ppm=None, contrast_ppm_refl=None,
                                                             mission='Kepler', response_nu=None, response_vals=None)
        return punto_system.compute_flux()


    Rp, A, Re = np.meshgrid(planetaryradius, albedo, redistribution, indexing='ij')
    points = list(zip(Rp.flatten(), A.flatten(), Re.flatten()))
    num_points = len(planetaryradius) * len(albedo) * len(redistribution)
    # parallel computation

    # Run in parallel with progress bar
    WHITE = '\033[97m'
    RESET = '\033[0m'
    with tqdm_joblib(tqdm(desc=f"{WHITE}Computing grid{RESET}", total=num_points)) as progress_bar:
        results = Parallel(n_jobs=-1)(delayed(run_model)(r, a, re) for r, a, re in points)

    normalized_flux = np.array(results).reshape(len(planetaryradius), len(albedo), len(redistribution), len(phase_model))

    # Saving the model
    np.savez_compressed(os.path.join(model_grid_folder, "model_grid.npz"),
                        planetaryradius=planetaryradius,
                        albedo=albedo,
                        recirculation=redistribution,
                        flux=normalized_flux)

    # Save metadata
    metadata = {
        "planetaryradius": {"name": "Planetary radius", "unit": "Jupiter radii"},
        "albedo": {"name": "albedo", "unit": "dimensionless"},
        "redistribution": {"name": "redistribution", "unit": "dimensionless"},
        "flux": "Normalized flux as a function of the phase angle",
        "author": "A. Dyrek",
        "date_created": datetime.now().isoformat(),
        "description": "Precomputed 3D model grid phase-curve normalised flux.",

        "fixed_parameters": {"target name": f"KIC {targetname}",
                             "period": f"{period} day",
                             "wavelength range": f"{wavearray} micron (Kepler Bandpass)",
                             "phase discretisation": f"{nphase} values (in degrees)",
                             "Planetary mass * sini": f"{planetarymasssini} Jupiter mass * sini",
                             "Inclination": f"{inclination} degrees"}

    }

    with open(os.path.join(model_grid_folder, "model_grid_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
else:
    print("'create_grid' is False: reading the model grid.")
    # Load the saved grid
    model_grid = np.load(os.path.join(model_grid_folder, "model_grid.npz"))

    planetaryradius = model_grid["planetaryradius"]
    albedo = model_grid["albedo"]
    redistribution = model_grid["redistribution"]
    normalized_flux = model_grid["flux"]

    # load metadata
    with open(os.path.join(model_grid_folder, "model_grid_metadata.json"), "r") as f:
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

if plot_grid is True:
    print("'plot_grid' is True: plotting the grid.")

    def plot_photometry_model(normalized_flux, planetaryradius, albedo, redistribution):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)
        for angle in [0, 90, 180, 270, 360]:
            ax.axvline(angle, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)

        # Flatten the parameter combinations
        n_r, n_a, n_rcl, n_phase = normalized_flux.shape
        total_curves = n_r * n_a * n_rcl

        flux_reshaped = normalized_flux.reshape((total_curves, n_phase))
        i_vals, j_vals, k_vals = np.meshgrid(
            np.arange(n_r),
            np.arange(n_a),
            np.arange(n_rcl),
            indexing='ij'
        )

        i_flat = i_vals.flatten()
        j_flat = j_vals.flatten()
        k_flat = k_vals.flatten()

        # Loop over flattened index arrays
        for idx in range(total_curves):
            i, j, k = i_flat[idx], j_flat[idx], k_flat[idx]
            flux_curve = flux_reshaped[idx]
            if not np.all(np.isnan(flux_curve)):
                ax.plot(
                    phase_model,
                    flux_curve,
                    linewidth=0.8,
                    linestyle='-',
                    color=colors_matter[i]
                )

        # Axis formatting
        xticks = [0, 90, 180, 270, 360]
        xlabels = [f'{x}°' for x in xticks]
        ax.set_xticks(xticks, labels=xlabels)
        ax.set_xlabel('Phase angle')
        ax.set_ylabel('Amplitude [ppm]')

        legend_text = (
            f"$R_\\mathrm{{p}}$ = [{np.min(planetaryradius):.2f}, {np.max(planetaryradius):.2f}] $R_\\mathrm{{J}}$\n"
            f"Albedo = [{np.min(albedo):.2f}, {np.max(albedo):.2f}]\n"
            f"Redist. = [{np.min(redistribution):.2f}, {np.max(redistribution):.2f}]\n"
            f"Discretisation = [{n_r}, {n_a}, {n_rcl}]"
        )
        ax.text(0.65, 0.98, legend_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))
        plt.tight_layout()
        plt.savefig(os.path.join(model_grid_folder, 'grid_model.png'),  format='png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_photometry_model(normalized_flux, planetaryradius, albedo, redistribution)

else:
    pass

# Open data
input_file = '/Users/ah258874/Documents/STScI_fellowship/Punto/Kepler_long_cadence/9139163_lc_filtered.txt'

data = np.loadtxt(input_file)
time_array = data[:, 0]
y = data[:, 1]
ref_time = data[0, 0] - 0.27 * period

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

# interpolate over radius, albedo and redistribution
model_interpolator = RegularGridInterpolator((planetaryradius, albedo, redistribution), normalized_flux,
                                        bounds_error=False,
                                        fill_value=None)


# interpolate over phase
def get_model_flux(p, model_interpolator, phase_model, phase_obs):
    offset = p[-1]  # last parameter is the offset
    model_params = p[:-1]  # remaining are radius, albedo, redistribution

    # Interpolate the model flux for the given parameters
    flux_grid = model_interpolator(model_params)  # shape: (n_phase,)
    flux_interp = interp1d(phase_model, flux_grid, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Add the offset after interpolating
    return flux_interp(phase_obs) + offset


if perform_fit is True:
    print("'perform_fit' is True: performing fit.")

    def residuals(p, foldy, yerr, model_interpolator, phase_model, phase_obs):
        model_flux = get_model_flux(p, model_interpolator, phase_model, phase_obs)
        model_flux = np.squeeze(model_flux)
        return (foldy - model_flux) / yerr


    # do the fit
    x0 = [0.3, 0.2, 0.2, 0]
    bounds = ([planetaryradius.min(), albedo.min(), redistribution.min(), -np.inf],
          [planetaryradius.max(), albedo.max(), redistribution.max(), +np.inf])

    res = least_squares(residuals, x0, bounds=bounds,
                    args=(foldy, yerr, model_interpolator, phase_model, phase_obs))

    #  get the best parameters
    best_fit_params = res.x

    # get the confidence intervals
    def confidence_intervals(result):
        J = result.jac
        residual_var = np.sum(result.fun**2) / (len(result.fun) - len(result.x))
        cov = residual_var * inv(J.T @ J)
        sigmas = np.sqrt(np.diag(cov))

        return sigmas, cov


    def compute_correlation(cov):
        std_devs = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_devs, std_devs)

        return corr

    sigmas, cov = confidence_intervals(res)
    corr = compute_correlation(cov)

    # Parameter names for labels
    param_names = ["Planetary Radius", "Albedo", "Redistribution", "Offset"]

    # Plot covariance matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.3)
    sns.heatmap(cov, xticklabels=param_names, yticklabels=param_names, annot=True, fmt=".2e", cmap="coolwarm")
    plt.title("Covariance Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'covariance.png'),  format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.3)
    sns.heatmap(corr, xticklabels=param_names, yticklabels=param_names, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'correlation.png'), format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # plot model VS data
    best_model = get_model_flux(best_fit_params, model_interpolator, phase_model, phase_obs)
    best_model = np.squeeze(best_model)

    print("\nBest-fit parameters:")
    print(f"Planetary radius ± 1σ: {best_fit_params[0]:.3f} ± {sigmas[0]:.3f} Jupiter radii")
    print(f"Albedo ± 1σ: {best_fit_params[1]:.3f} ± {sigmas[1]:.3f}")
    print(f"Redistribution ± 1σ: {best_fit_params[2]:.3f} ± {sigmas[2]:.3f}")
    print(f"Offset ± 1σ: {best_fit_params[3]:.3f} ± {sigmas[3]:.3f} ppm")

    # save the fit
    np.savez_compressed(os.path.join(result_folder, "fit_results.npz"),
                        best_fit_params=best_fit_params,
                        sigmas=sigmas,
                        best_model=best_model
                        )

else:
    print("'perform_fit' is False: reading file.")
    fit_results = np.load(os.path.join(result_folder, "fit_results.npz"))
    best_fit_params = fit_results["best_fit_params"]
    sigmas = fit_results["sigmas"]
    best_model = fit_results["best_model"]

    print("\nBest-fit parameters:")
    print(f"Planetary radius ± 1σ: {best_fit_params[0]:.3f} ± {sigmas[0]:.3f} Jupiter radii")
    print(f"Albedo ± 1σ: {best_fit_params[1]:.3f} ± {sigmas[1]:.3f}")
    print(f"Redistribution ± 1σ: {best_fit_params[2]:.3f} ± {sigmas[2]:.3f}")
    print(f"Offset ± 1σ: {best_fit_params[3]:.3f} ± {sigmas[3]:.3f} ppm")


# Plot the fit
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
                          markersize=3, label='best fit model')
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
plt.savefig(os.path.join(result_folder, 'best_fit.png'),  format='png', dpi=300, bbox_inches='tight')
plt.show()


# X2 map over a 2D slice: fixing one param
def chi2_grid(fixed_param_index, fixed_param_value, param_grid, obs_flux):
    """
    Plot χ² over a 2D slice by fixing one parameter.
    fixed_param_index: 0=radius, 1=albedo, 2=redistribution
    """
    i, j = sorted(set([0, 1, 2]) - {fixed_param_index})
    x_vals = [planetaryradius, albedo, redistribution][i]
    y_vals = [planetaryradius, albedo, redistribution][j]

    chi2_map = np.zeros((len(x_vals), len(y_vals)))

    for xi, xv in enumerate(x_vals):
        for yj, yv in enumerate(y_vals):
            p = [0, 0, 0, 0]
            p[fixed_param_index] = fixed_param_value
            p[i] = xv
            p[j] = yv
            model = get_model_flux(p, model_interpolator, phase_model, phase_obs)
            model = np.squeeze(model)
            chi2_map[xi, yj] = np.sum((model - obs_flux)**2)

    return x_vals, y_vals, chi2_map.T  # transpose for plotting


# fix redistribution and plot radius vs albedo X2 map
xvals_1, yvals_1, chi2_map_1 = chi2_grid(fixed_param_index=2, fixed_param_value=best_fit_params[2],
                                   param_grid=(planetaryradius, albedo), obs_flux=foldy)


fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
contour = ax.contourf(xvals_1, yvals_1, chi2_map_1, levels=20, cmap=cm.cm.matter_r)
plt.colorbar(contour, label='Chi-squared')
ax.set_xlabel('Planetary Radius [Jupiter Radii]')
ax.set_ylabel('Albedo')
ax.set_title('Chi-squared map (Redistribution fixed)')
ax.scatter(best_fit_params[0], best_fit_params[1], color='red',s=18, edgecolors='black',
           linewidths=0.8, label='Best-fit')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'chisquared_radius_albedo.png'),  format='png', dpi=300, bbox_inches='tight')
plt.show()

# fix planetary radius and plot albedo vs redistribution X2 map
xvals_2, yvals_2, chi2_map_2 = chi2_grid(fixed_param_index=0, fixed_param_value=best_fit_params[0],
                                   param_grid=(albedo, redistribution), obs_flux=foldy)

fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
contour = ax.contourf(xvals_2, yvals_2, chi2_map_2, levels=20, cmap=cm.cm.matter_r)
plt.colorbar(contour, label='Chi-squared')
ax.set_xlabel('Albedo')
ax.set_ylabel('Redistribution')
ax.set_title('Chi-squared map (Planetary radius fixed)')
ax.scatter(best_fit_params[1], best_fit_params[2], color='red',s=18, edgecolors='black',
           linewidths=0.8, label='Best-fit')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'chisquared_albedo_redistribution.png'),  format='png', dpi=300, bbox_inches='tight')
plt.show()

# fix albedo and plot planetary radius vs redistribution X2 map
xvals_3, yvals_3, chi2_map_3 = chi2_grid(fixed_param_index=1, fixed_param_value=best_fit_params[1],
                                   param_grid=(planetaryradius, redistribution), obs_flux=foldy)

fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
contour = ax.contourf(xvals_3, yvals_3, chi2_map_3, levels=20, cmap=cm.cm.matter_r)
plt.colorbar(contour, label='Chi-squared')
ax.set_xlabel('Planetary Radius [Jupiter Radii]')
ax.set_ylabel('Redistribution')
ax.set_title('Chi-squared map (Albedo fixed)')
ax.scatter(best_fit_params[0], best_fit_params[2], color='red', s=18, edgecolors='black',
           linewidths=0.8, label='Best-fit')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'chisquared_radius_redistribution.png'),  format='png', dpi=300,
            bbox_inches='tight')
plt.show()

print('done')






