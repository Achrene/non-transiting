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
from statsmodels.stats.stattools import durbin_watson
from astropy.timeseries import LombScargle

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


def run_fit_for_fixed_inclination(fixed_inclination_value, model_grid_folder, main_result_folder, result_folder,
                                    perform_fit=False):

    #  inputs
    targetname = '9139163'
    period = 0.604734  # in days
    nphase = 100  # discretization of phases
    phase_model = np.linspace(0, 360., nphase)

    references_path = '/Users/ah258874/PycharmProjects/punto/references'

    # Load the saved grid
    model_grid = np.load(os.path.join(model_grid_folder, "model_grid.npz"))

    planetaryradius = model_grid["planetaryradius"]
    albedo = model_grid["albedo"]
    redistribution = model_grid["redistribution"]
    inclination = model_grid["inclination"]
    normalized_flux = model_grid["flux"]

    # load metadata
    with open(os.path.join(model_grid_folder, "model_grid.json"), "r") as f:
        metadata = json.load(f)

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

    # Fix grid to one redistribution value
    inc_idx = (np.abs(inclination - fixed_inclination_value)).argmin()
    nearest_inclination = inclination[inc_idx]

    print(f"fit_results_ultranest_i{int(nearest_inclination)}.npz")

    print(f"Fixed redistribution to nearest available value: {int(nearest_inclination)}")
    model_grid_2d = normalized_flux[:, :, :, inc_idx]

    # Open data
    input_file = '/Users/ah258874/Documents/STScI_fellowship/Punto/Kepler_long_cadence/9139163_lc_filtered.txt'

    data = np.loadtxt(input_file)
    time_array = data[:, 0]
    y = data[:, 1]
    ref_time = data[0, 0] - 0.27 * period

    # get the errorbars
    referencetimefilename = 't0_estimation.csv'
    referencetimedata = pd.read_csv(os.path.join(references_path, referencetimefilename))
    target = referencetimedata.loc[referencetimedata['KIC'] == int(targetname)]
    repetitions = len(time_array)
    yerr = np.repeat(np.asarray(target['noise_jenkins'])[0], repetitions)

    # Phase-fold
    foldx, foldy = utils.phase_fold(time_array, y, period, ref_time)

    # modify observation phase array: foldx to get it in phase between 0-360
    phase_obs = ((foldx + 0.5 * period) / period) * 360

    # interpolate over radius and albedo
    model_interpolator = RegularGridInterpolator(
        (planetaryradius, albedo, redistribution),
        model_grid_2d,
        bounds_error=False,
        fill_value=None
    )

    # interpolate over phase
    def get_model_flux(p, model_interpolator, phase_model, phase_obs):
        offset = p[-1]
        model_params = p[:-1]  # radius, albedo, recirculation

        flux_grid = model_interpolator(model_params)
        flux_interp = interp1d(phase_model, flux_grid, kind='linear', bounds_error=False, fill_value="extrapolate")

        return flux_interp(phase_obs) + offset

    ultranest_output = os.path.join(main_result_folder, 'ultranestoutput/')
    os.makedirs(ultranest_output, exist_ok=True)

    if perform_fit is True:
        print("'perform_fit' is True: performing fit using UltraNest Nested Sampling.")

        # bounds
        param_names = ["Planetary Radius", "Albedo", "Recirculation", "Offset"]
        param_bounds = [
            (planetaryradius.min(), planetaryradius.max()),
            (albedo.min(), albedo.max()),
            (redistribution.min(), redistribution.max()),
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
        resi = best_model - foldy

        print(f"Inclination fixed : {int(nearest_inclination)}")
        print("\nBest-fit parameters:")
        print("\nPosterior mean ± std:")
        print(f"Planetary radius ± σ: {best_fit_params[0]:.4f} ± {std[0]:.4f} Jupiter radii")
        print(f"Albedo ± σ:           {best_fit_params[1]:.4f} ± {std[1]:.4f}")
        print(f"Redistribution ± σ:   {best_fit_params[2]:.4f} ± {std[2]:.4f}")
        print(f"Offset ± σ:           {best_fit_params[3]:.4f} ± {std[3]:.4f} ppm")

        # Save results
        np.savez_compressed(os.path.join(result_folder, f"fit_results_ultranest_i{int(nearest_inclination)}.npz"),
                            best_fit_params=best_fit_params,
                            sigmas=std,
                            best_model=best_model,
                            residuals=resi,
                            phase=phase_obs,
                            yerr=yerr,
                            samples=samples,
                            ultranest_result=result,
                            param_names=param_names,
                            logz=result['logz'],
                            logzerr=result['logzerr'])

    else:
        print("'perform_fit' is False: reading file.")

        fit_results = np.load(os.path.join(result_folder,
                                           f"fit_results_ultranest_i{int(nearest_inclination)}.npz"))
        best_fit_params = fit_results["best_fit_params"]
        samples = fit_results["samples"]
        std = fit_results["sigmas"]
        best_model = fit_results["best_model"]
        resi = fit_results["residuals"]
        phase_obs = fit_results["phase"],
        yerr = fit_results["yerr"]
        param_names = fit_results["param_names"]

        phase_obs = np.array(phase_obs)
        phase_obs = np.squeeze(phase_obs)
        resi = np.array(resi)

        print(f"Inclination fixed:  {int(nearest_inclination)}")
        print("\nBest-fit parameters:")
        print("\nPosterior mean ± std:")
        print(f"Planetary radius ± σ: {best_fit_params[0]:.4f} ± {std[0]:.4f} Jupiter radii")
        print(f"Albedo ± σ:           {best_fit_params[1]:.4f} ± {std[1]:.4f}")
        print(f"Redistribution ± σ:   {best_fit_params[2]:.4f} ± {std[2]:.4f}")
        print(f"Offset ± σ:           {best_fit_params[3]:.4f} ± {std[3]:.4f} ppm")

    # Plot the fit

    fig = corner.corner(samples, labels=param_names,
                        truths=best_fit_params,  show_titles=True,
                        title_fmt='.5f', title_kwargs={"fontsize": 12})
    plt.savefig(os.path.join(result_folder, f'corner_plot_ultranest_i{int(nearest_inclination)}.png'),
                format='png',
                dpi=300, bbox_inches='tight')
    plt.close()

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
                             markersize=3, label=f'best fit model (i = {int(nearest_inclination)}$^\circ$)')
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
    plt.savefig(os.path.join(result_folder, f'best_fit_ultranest_i{int(nearest_inclination)}.png'), format='png',
                dpi=300,
                bbox_inches='tight')
    plt.close()


def compute_bayes_factor(logZ1, logZ2):
    delta_logZ = logZ1 - logZ2
    bayes_factor = np.exp(delta_logZ)
    if abs(delta_logZ) < 1:
        interpretation = "Inconclusive"
    elif abs(delta_logZ) < 3:
        interpretation = "Substantial"
    elif abs(delta_logZ) < 5:
        interpretation = "Strong"
    else:
        interpretation = "Very strong"
    return delta_logZ, bayes_factor, interpretation


def compute_rms_diagnostic(residuals, max_bins=50):
    """Compute RMS vs bin size and diagnostic ratio η."""
    n = len(residuals)
    rms_all = []
    for binsize in range(1, max_bins + 1):
        nbins = n // binsize
        reshaped = residuals[:nbins * binsize].reshape(nbins, binsize)
        binned = reshaped.mean(axis=1)
        rms_all.append(np.std(binned))
    rms_all = np.array(rms_all)
    rms_expected = rms_all[0] / np.sqrt(np.arange(1, max_bins + 1))

    # diagnostic ratio: measured vs expected at largest bin size
    eta = rms_all[-1] / rms_expected[-1]
    return rms_all, rms_expected, eta


def plot_residual_diagnostics(phase, residuals, rms_all, rms_exp, inc_val, freq_to_test,
                              power_val, fap_val, dw_val, result_folder):
    """
    freq_to_test, power_val, fap_val are from astropy LombScargle
    dw_val is from the Durbin–Watson statistical test
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=100, gridspec_kw={'height_ratios': [3, 3, 1]})

    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    residuals_sorted = residuals[sort_idx]

    # Top panel: phase-folded residuals
    ax[0].scatter(phase_sorted / 360, residuals_sorted, alpha=0.5, color='lightgrey', s=8, marker='o')
    ax[0].axhline(y=0, xmin=0, xmax=360, linewidth=1.6, color=colors_matter[0])
    ax[0].set_ylabel("Residuals [ppm]")
    ax[0].set_xlabel("Orbital phase")
    ax[0].set_xlim(0, 1)
    ax[0].set_title(f"Inclination = {inc_val}°")

    # Bottom panel: RMS vs bin size
    bin_sizes = np.arange(1, len(rms_all) + 1)
    ax[1].loglog(bin_sizes, rms_all, 'o-', color=colors_matter[1], label='Residual RMS')
    ax[1].loglog(bin_sizes, rms_exp, '--', linewidth=1.6, color=colors_matter[0], label='White noise exp.')
    ax[1].set_xlabel("Bin size [nb points]")
    ax[1].set_ylabel("RMS [ppm]")
    ax[1].legend(fontsize=13)
    ax[1].grid(True, which='both', ls=':')

    ax[2].axis('off')  # we use text rather than axes
    txt = []
    txt.append(f"Tested frequency = {freq_to_test}")
    txt.append(f"Power = {power_val:.4f}")
    txt.append(f"FAP = {fap_val:.2e}")
    txt.append(f"Durbin-Watson = {dw_val:.3f}")
    # if you computed eta earlier:
    # txt.append(f"RMS diagnostic η = {eta:.3f}")
    ax[2].text(0.01, 0.6, "\n".join(txt), fontsize=11, family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'residual_diag_phase_i{inc_val}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def main():

    model_grid_folder = '/Users/ah258874/PycharmProjects/punto/results_grid/model_grid/full_1_62'
    main_result_folder = '/Users/ah258874/PycharmProjects/punto/results_grid/ultranest'
    result_folder = os.path.join(main_result_folder, 'fit_fixed_i/full_1_62')

    grid_file = os.path.join(model_grid_folder, "model_grid.npz")
    inclination_values = np.arange(1, 62, 1)  # np.load(grid_file)["inclination"]

    summary = {}

    for inc_val in inclination_values:
        print(f"\nFitted model with inclination = {int(inc_val)}")

        if os.path.exists(os.path.join(result_folder, f'fit_results_ultranest_i{int(inc_val)}.npz')):
            print(f'Inclination {int(inc_val)} deg: fit already performed, reading result file.')
            perform_fit = False
        else:
            print(f"Inclination {int(inc_val)} deg: performing fit.")
            perform_fit = True

        run_fit_for_fixed_inclination(inc_val, model_grid_folder, main_result_folder, result_folder,
                                         perform_fit=perform_fit)

        result_file = os.path.join(result_folder, f"fit_results_ultranest_i{int(inc_val)}.npz")
        result = np.load(result_file)
        logz = result["logz"]
        print(f"LogZ = {logz:.3f}")
        best_fit_params = result["best_fit_params"]
        std = result["sigmas"]
        resi = result["residuals"]
        phase_obs = result["phase"],
        yerr = result["yerr"]

        phase_obs = np.array(phase_obs)
        phase_obs = np.squeeze(phase_obs)
        resi = np.array(resi)

        rms_all, rms_exp, eta = compute_rms_diagnostic(resi, max_bins=50)
        dw = durbin_watson(resi)

        f_orb_phase = 1.0  # one cycle per orbit in phase units

        phase_obs_1d = np.ravel(phase_obs)  # or phase.flatten()
        resi_1d = np.ravel(resi)
        yerr_1d = np.ravel(yerr)

        ls = LombScargle(phase_obs_1d, resi_1d, yerr_1d)
        power = ls.power(f_orb_phase)
        fap = ls.false_alarm_probability(power)

        summary[inc_val] = dict(
            logz=logz,
            params=best_fit_params,
            std=std,
            rms_ratio=eta,
            dw=dw,
            ls_power=power,
            ls_fap=fap
        )

        plot_residual_diagnostics(phase_obs, resi, rms_all, rms_exp, inc_val, f_orb_phase,
                              power, fap, dw, result_folder)

        print(f"Inclination fixed:  {int(inc_val)} degrees")
        print("\nBest-fit parameters:")
        print("\nPosterior mean ± std:")
        print(f"Planetary radius ± σ: {best_fit_params[0]:.4f} ± {std[0]:.4f} Jupiter radii")
        print(f"Albedo ± σ:           {best_fit_params[1]:.4f} ± {std[1]:.4f}")
        print(f"Redistribution ± σ:   {best_fit_params[2]:.4f} ± {std[2]:.4f}")
        print(f"Offset ± σ:           {best_fit_params[3]:.4f} ± {std[3]:.4f} ppm")

    print("\n=== Model Comparison Summary ===")
    best_inc = max(summary, key=lambda k: summary[k]["logz"])
    best_logz = summary[best_inc]["logz"]
    print(f"Best model: inclination = {int(best_inc)} deg, logZ = {best_logz:.3f}\n")

    print(f"{'Inc':<5}{'logZ':<12}{'Delta logZ':<10}{'BF':<12}{'eta (RMS ratio)':<16}"
          f"{'DW':<8}{'LS pow':<12}{'FAP':<12}{'Interp'}")
    print("-" * 95)
    for inc, vals in sorted(summary.items()):
        delta, bf, _ = compute_bayes_factor(vals["logz"], best_logz)

        # Interpret residuals
        if vals["rms_ratio"] <= 1.2:
            res_interp = "White-like"
        elif vals["rms_ratio"] <= 1.5:
            res_interp = "Marginal"
        else:
            res_interp = "Correlated"

        print(f"{inc:<5}{vals['logz']:<12.3f}{delta:<10.3f}{bf:<12.2f}"
              f"{vals['rms_ratio']:<16.3f}{vals['dw']:<8.2f}"
              f"{vals['ls_power']:<12.3f}{vals['ls_fap']:<12.1e}{res_interp}")

    # --- Combined summary plots ---
    inclinations = np.array(sorted(summary.keys()))
    logz_vals = np.array([summary[i]["logz"] for i in inclinations])
    rms_ratios = np.array([summary[i]["rms_ratio"] for i in inclinations])

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=100)

    # Left: logZ vs inclination
    ax[0].plot(inclinations, logz_vals - best_logz, 'o-', color=colors_matter[1])
    ax[0].axhline(0, color='k', lw=1, ls='--')
    ax[0].set_xlabel("Inclination [$^{\circ}$]")
    ax[0].set_ylabel(r" $\Delta$logZ (relative to best)")

    # Right: eta vs inclination
    ax[1].plot(inclinations, rms_ratios, 'o-', color=colors_matter[2])
    ax[1].axhline(1.2, color='orange', ls='--', lw=1, label='$\eta$ = 1.2')
    ax[1].axhline(1.5, color='red', ls='--', lw=1, label='$\eta$ = 1.5')
    ax[1].set_xlabel("Inclination [$^{\circ}$]")
    ax[1].set_ylabel("Residual RMS ratio $\eta$")
    ax[1].legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "summary_diagnostics.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # --- Automatic cutoff finder ---
    safe_inclinations = []
    unlikely_inclinations = []

    for inc, vals in sorted(summary.items()):
        delta, _, _ = compute_bayes_factor(vals["logz"], best_logz)

        if (delta >= -3) and (vals["rms_ratio"] <= 1.2):
            safe_inclinations.append(inc)
        else:
            unlikely_inclinations.append(inc)

    print("\n=== Inclination Cutoff Assessment ===")
    if safe_inclinations:
        print(f"Safe inclinations (good evidence + white-like residuals): {safe_inclinations}")
        print(f"Threshold starts at ~{min(safe_inclinations)}°")
    else:
        print("No inclinations meet both evidence and residual thresholds.")

    if unlikely_inclinations:
        print(f"Unlikely inclinations: {unlikely_inclinations}")


if __name__ == "__main__":
    main()


print('done')






