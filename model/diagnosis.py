import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
from Cython.Utils import path_exists
import sys
import utils
from statsmodels.stats.stattools import durbin_watson
from astropy.timeseries import LombScargle
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib import colormaps
import matplotlib.lines as lines

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

colors_matter = cm.cm.matter_r(np.linspace(0,2,10))
colors_phase = cm.cm.phase(np.linspace(0,2,10))
cmap = colormaps['jet']
colors_rainbow = cmap(np.linspace(2, 4, 5))
colors_brewery = ['#e7298a', '#7570b3', '#66a61e', '#d95f02']

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
                              power_val, fap_val, dw_val, t0, result_folder):
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
    ax[0].set_title(f"Inclination = {inc_val}°, t0 = {t0}")

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
    plt.savefig(os.path.join(result_folder, f'residual_diag_phase_i{inc_val}_{t0}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    target = '9139163'
    main_result_folder = '/Users/adyrek/PycharmProjects/punto/fit'
    references_path = '/Users/adyrek/PycharmProjects/punto/references'
    references_path = Path(references_path)
    data_filename = '9139163_lc_filtered.txt'
    period = 0.604734

    # Load photometry data
    if path_exists(references_path / data_filename):
        data = np.loadtxt(references_path / data_filename)
    else:
        print('Cannot locate data file. Change configuration file.')
        sys.exit()

    time_array, y = data[:, 0], data[:, 1]

    t0_RV = 2459664.752
    t0_Kepler = 2454953.5394706232

    ref_table = pd.read_csv(os.path.join(references_path, "t0_estimation.csv"))
    target = ref_table.loc[ref_table['KIC'] == int(target)]
    yerr = np.repeat(np.asarray(target['noise_jenkins'])[0], len(y))

    result_folders = glob.glob(os.path.join(main_result_folder, '*LS'))

    inclination_values = np.arange(17, 62, 5)  # np.load(grid_file)["inclination"]

    summary = {}
    inclinations = []
    t0_values = []

    pattern = re.compile(
        r"i_fixed(?P<inc>\d+(?:\.\d+)?).*?t0(?P<t0>[A-Za-z0-9]*)"
    )

    for path in result_folders:
        folder = Path(path)  # absolute path
        folder_name = folder.name  # just the basename

        match = pattern.search(folder_name)
        if match:
            inc = float(match.group("inc"))
            t0 = match.group("t0")  # string (may be "")

            inclinations.append(inc)
            t0_values.append(t0)

            print(f"{folder} -> inc={inc}, t0='{t0}'")

            if 't3sigmas' in t0:
                t0_RV = t0_RV + 3 * 0.018
            elif 't5sigmas' in t0:
                t0_RV = t0_RV + 5 * 0.018
            elif 'sigmas' in t0:
                t0_RV = t0_RV + 0.018
            else:
                t0_RV = t0_RV

            ref_time = t0_RV - t0_Kepler

            foldx, foldy = utils.phase_fold(time_array, y, period, ref_time)
            phase_obs = ((foldx + 0.5 * period) / period) * 360

            result_file = folder / f"{folder_name}_fit_results.npz"

            print(result_file)  # check the path
            result = np.load(result_file)

            param_names = result["param_names"]
            param_bounds = result["param_bounds"]
            best_params = result["best_params"]
            best_model = result["best_model"]
            std = result["sigmas"]

            best_model = np.squeeze(best_model)
            resi = best_model - foldy
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

            summary.setdefault(inc, {})[t0] = dict(
                params=best_params,
                std=std,
                rms_ratio=eta,
                dw=dw,
                ls_power=power,
                ls_fap=fap
            )

            plot_residual_diagnostics(phase_obs, resi, rms_all, rms_exp, inc, f_orb_phase,
                                  power, fap, dw, t0, folder)

            print(f"Inclination fixed:  {int(inc)} degrees")
            print(f"t0 is:  {t0}")
            print("\nBest-fit parameters:")
            print("\nPosterior mean ± std:")
            print(f"Planetary radius ± σ: {best_params[0]:.4f} ± {std[0]:.4f} Jupiter radii")
            print(f"Albedo ± σ:           {best_params[1]:.4f} ± {std[1]:.4f}")
            print(f"Redistribution ± σ:   {best_params[2]:.4f} ± {std[2]:.4f}")
            print(f"Cloud Offset ± σ:     {best_params[3]:.4f} ± {std[3]:.4f} degrees")
            print(f"Amplitude Offset ± σ: {best_params[4]:.4f} ± {std[4]:.4f} ppm")

    print("\n=== Model Comparison Summary ===")

    for inc, t0_dict in sorted(summary.items()):
        for t0, vals in sorted(t0_dict.items()):

            # Interpret residuals
            if vals["rms_ratio"] <= 1.2:
                res_interp = "White-like"
            elif vals["rms_ratio"] <= 1.5:
                res_interp = "Marginal"
            else:
                res_interp = "Correlated"

            print(f"inc={inc}, t0={t0} -> {res_interp}")

            print(
                  f"{vals['rms_ratio']:<16.3f}{vals['dw']:<8.2f}"
                  f"{vals['ls_power']:<12.3f}{vals['ls_fap']:<12.1e}{res_interp}")

    t0_values = sorted({t0 for inc in summary for t0 in summary[inc]})
    t0_color = {t0: colors_brewery[i % len(colors_brewery)] for i, t0 in enumerate(t0_values)}

    t0_mapping = [
        ('5sigma', 'T$_0$ + 5$\sigma$'),
        ('3sigma', 'T$_0$ + 3$\sigma$'),
        ('sigma', 'T$_0$ + $\sigma$'),
        ('', 'T$_0$')
    ]

    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=100)
    for t0 in t0_values:
        inc_list = []
        rms_list = []
        for inc in sorted(summary.keys()):
            if t0 in summary[inc]:
                inc_list.append(inc)
                rms_list.append(summary[inc][t0]["rms_ratio"])

        t0_legend = r'T0'  # default
        for key, label in t0_mapping:
            if key and key in t0:
                t0_legend = label
                break
            elif key == '' and t0 == '':
                t0_legend = label
                break

        ax.plot(
            inc_list,
            rms_list,
            'o-',
            color=t0_color[t0],
            markerfacecolor=t0_color[t0],  # fill color
            markeredgecolor='black',  # black circle around each marker
            markersize=8,
            label=t0_legend
        )


    ax.axhline(1.2, color='green', ls='--', lw=1)
    ax.axhline(1.5, color='peru', ls='--', lw=1)
    ax.text(48, 1.27, s='RMS ratio $\chi$ = 1.2', fontfamily='sans-serif', color='green', fontsize=18)
    ax.text(48, 1.57, s='RMS ratio $\chi$ = 1.5', fontfamily='sans-serif', color='peru', fontsize=18)

    ax.set_xlabel("Inclination [$^{\circ}$]", fontsize=22)
    ax.set_ylabel("Residual RMS ratio $\chi$", fontsize=22)

    ax.tick_params(axis="both", which ="both", direction ="in", top = True, right = True)
    ax.tick_params(which="major", length = 8, width = 1.4, labelsize = 20)
    ax.tick_params(which="minor", length = 4, width = 1.0)
    ax.minorticks_on()

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 3, 1, 2]  # We want 'First Item' (index 1) to be first, 'Second Item' (index 0) to be second

    # Reorder the handles and labels
    reordered_handles = [handles[idx] for idx in order]
    reordered_labels = [labels[idx] for idx in order]

    ax.legend(reordered_handles, reordered_labels, ncols=2,  loc='upper left', bbox_to_anchor=(0.48, 0.92), fontsize=18)

    rect = Rectangle(
        (15,0.7),
        50,
        0.5,
        facecolor='green',  # Fill color
        alpha=0.1,  # Transparency
        edgecolor='green',  # Border color
        linewidth=1  # Border width
    )
    ax.text(16.2, 0.98, s=r'No correlation', color='green', fontfamily='sans-serif', fontsize=18)

    # 4. Add the patch to the Axes
    ax.add_patch(rect)
    ax.set_xlim(15, 65)


    plt.tight_layout()
    plt.savefig(os.path.join('/Users/adyrek/PycharmProjects/punto/fit', "summary_diagnostics_LS.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # print results
    results = {}

    for iparam, pname in enumerate(param_names):
        values = []
        errors = []
        incs = []
        t0s = []

        for inc in summary:
            for t0 in summary[inc]:
                values.append(summary[inc][t0]["params"][iparam])
                errors.append(summary[inc][t0]["std"][iparam])
                incs.append(inc)
                t0s.append(t0)

        values = np.array(values)
        errors = np.array(errors)
        incs = np.array(incs)
        t0s = np.array(t0s)

        # min / max with corresponding errors
        imin = np.argmin(values)
        imax = np.argmax(values)

        results[pname] = {
            "min": {
                "value": values[imin],
                "error": errors[imin],
                "inc": incs[imin],
                "t0": t0s[imin],
            },
            "max": {
                "value": values[imax],
                "error": errors[imax],
                "inc": incs[imax],
                "t0": t0s[imax],
            },
            "mean": {
                "value": np.mean(values),
                "error": np.sqrt(np.sum(errors ** 2)) / len(errors),
            },
        }

    for pname, res in results.items():
        print(f"\n{pname.upper()}")

        print(
            f"Min  : {res['min']['value']:.4f} ± {res['min']['error']:.4f} "
            f"(inc={res['min']['inc']}, t0={res['min']['t0']})"
        )

        print(
            f"Max  : {res['max']['value']:.4f} ± {res['max']['error']:.4f} "
            f"(inc={res['max']['inc']}, t0={res['max']['t0']})"
        )

        print(
            f"Mean : {res['mean']['value']:.4f} ± {res['mean']['error']:.4f}"
        )

    print('done')

    return

main()

print('done')