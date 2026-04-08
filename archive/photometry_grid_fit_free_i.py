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
import sys
from archive import ExoplanetarySystem
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from datetime import datetime
from model import utils

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


def plot_photometry(time, lightcurve, lightcurverror):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
    ax.errorbar(time, lightcurve, lightcurverror,
                color=colors_matter[0], zorder=1, alpha=0.2, marker='.', linestyle=' ', linewidth=None)
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Amplitude [ppm]')
    plt.tight_layout()
    plt.show()


def plot_phase_folded(foldedphase, foldedlightcurve, lightcurverror, binning=False, binned_folded_phase=None,
                      binned_folded_lightcurve=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
    if binning==True:
        ax.scatter(binned_folded_phase, binned_folded_lightcurve,
                color=colors_matter[8], zorder=2, alpha=0.5, marker='.')
    else:
        pass
    ax.errorbar(foldedphase/360, foldedlightcurve, yerr=lightcurverror,
                color=colors_matter[1], zorder=1, alpha=0.2, marker='.', linestyle=' ', linewidth=None)
    ax.set_xlabel(r'Orbital phase')
    ax.set_ylabel('Amplitude [ppm]')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


#  inputs
targetname = '9139163'
period = 0.604734  # in days
nphase = 100  # discretization of phases
phase_model = np.linspace(0, 360., nphase)
wavearray = np.array([0.430, 0.890])  # Kepler bandpass in micron
planetarymasssini = 12.5  # Earth mass, from RV fit
planetarymasssini = planetarymasssini * 0.00314558  # conversion into Jupiter mass -> not used in the mode right now


references_path = '/references'
model_grid_folder = '/Users/adyrek/PycharmProjects/punto/results_grid/model_grid/full_1_62'
main_result_folder = '/Users/adyrek/PycharmProjects/punto/results_grid/ultranest/'
result_folder = os.path.join(main_result_folder, 'fit_free_i/change_T0')

fit_section = True
perform_fit = True
Kepler = True
TESS = False
plot_data = True
create_grid = False
plot_grid = False

if create_grid is True:

    if os.path.exists(os.path.join(model_grid_folder, f'model_grid.npz')):
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
    inclination = np.linspace(1, 62, 62)

    # creating the grid of model
    def run_model(planetaryradius, albedo, redistribution, inclination):
        sini = np.sin(inclination * np.pi / 180)
        planetarymass = planetarymasssini / sini  # Jupiter mass

        # previously used: stellar mass -- 1.36 Sun mass and stellar radius -- 1.54 Sun radius

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


    Rp, A, Re, inc = np.meshgrid(planetaryradius, albedo, redistribution, inclination, indexing='ij')
    points = list(zip(Rp.flatten(), A.flatten(), Re.flatten(), inc.flatten()))
    num_points = len(planetaryradius) * len(albedo) * len(redistribution) * len(inclination)
    # parallel computation

    # Run in parallel with progress bar
    WHITE = '\033[97m'
    RESET = '\033[0m'
    with tqdm_joblib(tqdm(desc=f"{WHITE}Computing grid{RESET}", total=num_points)) as progress_bar:
        results = Parallel(n_jobs=-1)(delayed(run_model)(r, a, re, i) for r, a, re, i in points)

    normalized_flux = np.array(results).reshape(len(planetaryradius), len(albedo), len(redistribution),
                                                len(inclination), len(phase_model))

    # Saving the model
    np.savez_compressed(os.path.join(model_grid_folder, f"model_grid.npz"),
                        planetaryradius=planetaryradius,
                        albedo=albedo,
                        redistribution=redistribution,
                        inclination=inclination,
                        flux=normalized_flux)

    # Save metadata
    metadata = {
        "planetaryradius": {"name": "Planetary radius", "unit": "Jupiter radii"},
        "albedo": {"name": "albedo", "unit": "dimensionless"},
        "redistribution": {"name": "redistribution", "unit": "dimensionless"},
        "inclination": {"name": "inclination", "unit": "degrees"},
        "flux": "Normalized flux as a function of the phase angle",
        "author": "A. Dyrek",
        "date_created": datetime.now().isoformat(),
        "description": "Precomputed 3D model grid phase-curve normalised flux.",

        "fixed_parameters": {"target name": f"KIC {targetname}",
                             "period": f"{period} day",
                             "wavelength range": f"{wavearray} micron (Kepler Bandpass)",
                             "phase discretisation": f"{nphase} values (in degrees)",
                             "Planetary mass * sini": f"{planetarymasssini} Jupiter mass * sini"
                             }

    }

    with open(os.path.join(model_grid_folder, "model_grid.json"), "w") as f:
        json.dump(metadata, f, indent=2)
else:
    print("'create_grid' is False: reading the model grid.")
    # Load the saved grid
    model_grid = np.load(os.path.join(model_grid_folder, "model_grid.npz"))

    planetaryradius = model_grid["planetaryradius"]
    albedo = model_grid["albedo"]
    redistribution = model_grid["redistribution"]
    inclination = model_grid["inclination"]
    inclination = np.asarray(inclination)
    normalized_flux = model_grid["flux"]

    # load metadata
    with open(os.path.join(model_grid_folder, "model_grid.json"), "r") as f:
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


if plot_grid is True:
    print("'plot_grid' is True: plotting the grid.")
    inclination = np.asarray(inclination)

    def plot_photometry_model(normalized_flux, planetaryradius, albedo, redistribution, inclination):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)
        for angle in [0, 90, 180, 270, 360]:
            ax.axvline(angle, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)

        # Flatten the parameter combinations
        n_r, n_a, n_rcl, n_inc, n_phase = normalized_flux.shape
        total_curves = n_r * n_a * n_rcl * n_inc

        flux_reshaped = normalized_flux.reshape((total_curves, n_phase))
        i_vals, j_vals, k_vals, l_vals = np.meshgrid(
            np.arange(n_r),
            np.arange(n_a),
            np.arange(n_rcl),
            np.arange(n_inc),
            indexing='ij'
        )

        i_flat = i_vals.flatten()
        j_flat = j_vals.flatten()
        k_flat = k_vals.flatten()
        l_flat = l_vals.flatten()

        # Loop over flattened index arrays
        for idx in range(total_curves):
            i, j, k, l = i_flat[idx], j_flat[idx], k_flat[idx], l_flat[idx]
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
            f"$i = [{np.min(inclination)}, {np.max(inclination)}]^\circ$\n"
            f"$R_\\mathrm{{p}}$ = [{np.min(planetaryradius):.2f}, {np.max(planetaryradius):.2f}] $R_\\mathrm{{J}}$\n"
            f"Albedo = [{np.min(albedo):.2f}, {np.max(albedo):.2f}]\n"
            f"Redist. = [{np.min(redistribution):.2f}, {np.max(redistribution):.2f}]\n"
            f"Discretisation = [{n_r}, {n_a}, {n_rcl}, {n_inc}]"
        )
        ax.text(0.65, 0.98, legend_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))
        plt.tight_layout()
        plt.savefig(os.path.join(model_grid_folder, 'model_grid.png'),  format='png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_photometry_model(normalized_flux, planetaryradius, albedo, redistribution, inclination)

else:
    pass

if fit_section:
    print("'fit_section' is on, fitting the lightcurve or printing the fit results.")

    if Kepler:
        # Open data
        input_file = '/Users/adyrek/Documents/STScI_fellowship/Punto/Kepler_long_cadence/9139163_lc_filtered.txt'

        data = np.loadtxt(input_file)
        time_array = data[:, 0]
        y = data[:, 1]
        t0_RV = 2459664.752
        t0_Kepler = 2454953.5394706232
        ref_time = t0_RV - t0_Kepler  #data[0, 0] - 0.27 * period

        # get the errorbars
        referencetimefiename = 't0_estimation.csv'
        referencetimedata = pd.read_csv(os.path.join(references_path, referencetimefiename))
        target =referencetimedata.loc[referencetimedata['KIC'] == int(targetname)]
        repetitions = len(time_array)
        yerr = np.repeat(np.asarray(target['noise_jenkins'])[0], repetitions)

    elif TESS:
        input_file = '/Users/adyrek/Documents/STScI_fellowship/Punto/TESS/' \
                     'Cleaned_TIC_164670309_1800s_regularly_sampled_1day_filtered_all_sector.data'
        df = pd.read_csv(input_file, delimiter=' ')
        time_array = df['Time']
        time_array = np.asarray(time_array)
        y = df['Flux']
        y = np.asarray(y)
        ref_time = time_array[0] - 0.5 * period

        noise_level = 93.4  # ppm
        repetitions = len(time_array)
        yerr = np.repeat(np.asarray(noise_level), repetitions)

    else:
        print("No dataset selected. No fit will be performed.")
        sys.exit()

    # Phase-fold
    foldx, foldy = utils.phase_fold(time_array, y, period, ref_time)

    # modify observation phase array: foldx to get it in phase between 0-360
    phase_obs = ((foldx + 0.5 * period) / period) * 360

    if plot_data:
        plot_photometry(time_array, y, yerr)
        plot_phase_folded(phase_obs, foldy, yerr, binning=False, binned_folded_phase=None,
                          binned_folded_lightcurve=None)

    # interpolate over radius, albedo and redistribution
    model_interpolator = RegularGridInterpolator((planetaryradius, albedo, redistribution, inclination), normalized_flux,
                                                 bounds_error=False,
                                                 fill_value=None)

    # interpolate over phase
    def get_model_flux(p, model_interpolator, phase_model, phase_obs):
        offset = p[-1]  # last parameter is the offset
        model_params = p[:-1]  # remaining are radius, albedo, redistribution, inclination

        # Interpolate the model flux for the given parameters
        flux_grid = model_interpolator(model_params)  # shape: (n_phase,)
        flux_interp = interp1d(phase_model, flux_grid, kind='linear', bounds_error=False, fill_value="extrapolate")

        # Add the offset after interpolating
        return flux_interp(phase_obs) + offset


    ultranest_output = os.path.join(main_result_folder, f'ultranestoutput/inc')
    os.makedirs(ultranest_output, exist_ok=True)

    if perform_fit is True:
        print("'perform_fit' is True: performing fit using UltraNest Nested Sampling.")

        # bounds
        param_names = ["Planetary Radius", "Albedo", "Redistribution", "Inclination", "Offset"]
        param_bounds = [
            (planetaryradius.min(), planetaryradius.max()),
            (albedo.min(), albedo.max()),
            (redistribution.min(), redistribution.max()),
            (inclination.min(), inclination.max()),
            (-5 * np.mean(yerr), 5 * np.mean(yerr))]


        def prior_transform(unit_cube):
            params = []
            for u, (low, high) in zip(unit_cube, param_bounds):
                params.append(low + u * (high - low))

            return np.array(params)


        def log_likelihood(p):
            model_flux = get_model_flux(p, model_interpolator, phase_model, phase_obs)
            model_flux = np.squeeze(model_flux)
            residuals = (foldy - model_flux) / yerr

            loglike = -0.5 * np.sum(residuals ** 2 + np.log(2 * np.pi * yerr ** 2))

            return loglike


        sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform,
                                                  log_dir=ultranest_output)

        # run nested sampling
        result = sampler.run(min_num_live_points=400, dlogz=0.5, frac_remain=1e-8, max_iters=40000)
        sampler.print_results()

        samples = result['samples']

        best_fit_params = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        # Get best-fit model flux for plotting
        best_model = get_model_flux(best_fit_params, model_interpolator, phase_model, phase_obs)
        best_model = np.squeeze(best_model)

        print("\nBest-fit parameters:")
        print("\nPosterior mean ± std:")
        print(f"Planetary radius ± σ: {best_fit_params[0]:.3f} ± {std[0]:.3f} Jupiter radii")
        print(f"Albedo ± σ:           {best_fit_params[1]:.3f} ± {std[1]:.3f}")
        print(f"Redistribution ± σ:   {best_fit_params[2]:.3f} ± {std[2]:.3f}")
        print(f"Inclination ± σ:      {best_fit_params[3]}     ± {std[3]}$^\circ$")
        print(f"Offset ± σ:           {best_fit_params[4]:.3f} ± {std[4]:.3f} ppm")

        # Save results
        np.savez_compressed(os.path.join(result_folder, f"fit_results_ultranest_inc.npz"),
                            best_fit_params=best_fit_params,
                            sigmas=std,
                            best_model=best_model,
                            samples=samples,
                            ultranest_result=result,
                            param_names=param_names)

    else:
        if os.path.exists(os.path.join(result_folder, f"fit_results_ultranest_inc.npz")):
            print("'perform_fit' is False: reading file.")
        else:
            print("'perform_fit' is False but no file found. Exiting process.")
            sys.exit()

        fit_results = np.load(os.path.join(result_folder, f"fit_results_ultranest_inc.npz"))
        best_fit_params = fit_results["best_fit_params"]
        std = fit_results["sigmas"]
        best_model = fit_results["best_model"]
        samples = fit_results["samples"]
        param_names = fit_results["param_names"]

        print("\nBest-fit parameters:")
        print("\nPosterior mean ± std:")
        print(f"Planetary radius ± σ: {best_fit_params[0]:.3f} ± {std[0]:.3f} Jupiter radii")
        print(f"Albedo ± σ:           {best_fit_params[1]:.3f} ± {std[1]:.3f}")
        print(f"Redistribution ± σ:   {best_fit_params[2]:.3f} ± {std[2]:.3f}")
        print(f"Inclination ± σ:      {best_fit_params[3]}     ± {std[3]}$^\circ$")
        print(f"Offset ± σ:           {best_fit_params[4]:.3f} ± {std[4]:.3f} ppm")

    # Plot the fit

    fig = corner.corner(samples, labels=param_names,
                        truths=best_fit_params, show_titles=True, title_fmt='.5f', title_kwargs={"fontsize": 12})
    plt.savefig(os.path.join(result_folder, f'corner_plot_ultranest_inc.png'),
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
                              markersize=3, label=f'best fit model')
    binned_data_label = lines.Line2D([], [], color=colors_matter[1],
                              markersize=3, label=f'{int(np.round(bin_time, 0))} day binned data')
    ax[0].legend(handles=[full_data_label, fit_label, binned_data_label], fontsize=13, loc="upper right",
                 facecolor='darkgrey')
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
    plt.savefig(os.path.join(result_folder, f'best_fit_ultranest_inc.png'),  format='png', dpi=300,
                bbox_inches='tight')
    plt.show()

else:
    print("'fit_section' is off, no fitting will be performed nor loaded.")

print('done')






