from typing import Optional
import numpy as np
from Cython.Utils import path_exists
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import shutil
import json
import corner
import ultranest
from ultranest.stepsampler import SliceSampler, generate_region_oriented_direction
import utils
from pathlib import Path
import yaml
from types import SimpleNamespace
from dataclasses import dataclass
import sys
import os
import pandas as pd
from scipy.optimize import least_squares
from numpy.linalg import inv
import matplotlib.patheffects as pe
import matplotlib.lines as lines
import matplotlib.patches as mpatches

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

@dataclass
class FitConfiguration:
    use_uniform_albedo: bool
    fit_inclination: bool
    fit_gravitational_effects: bool
    add_gravitational_model: bool
    fixed_inclination: Optional[float]  # value or None
    period: float
    effectivetemperature: float
    stellarmass: float
    stellarradius: float
    target: str
    planetarymasssini: float
    fitting_method: str
    yerr: Optional[np.ndarray] = None



    def validate(self):
        if self.fit_gravitational_effects and self.add_gravitational_model:
            raise ValueError("Cannot both fit gravitational effects and fix them. "
                             "Set one of fit_gravity or add_gravity_model to False.")
        if not self.fit_inclination and self.fixed_inclination is None:
            raise ValueError("If inclination is not fitted, fixed_inclination must be provided.")


class Priors:
    def __init__(self, config: FitConfiguration, grid_axes):
        self.names = []
        self.bounds = []

        # grid parameters
        for name, axis_values in grid_axes.items():
            # Skip albedo_min and cloud_offset if uniform albedo
            if config.use_uniform_albedo and name in ["Albedo min"]:
                continue
            # Skip inclination if fixed
            if name == "Inclination" and not config.fit_inclination:
                continue

            self.names.append(name)
            self.bounds.append((axis_values.min(), axis_values.max()))
        

        # Extra nuisance parameters
        self.names.append("Amplitude Offset")
        self.bounds.append((-5 * np.mean(config.yerr), 5 * np.mean(config.yerr)))

        # Delta T0 only in uniform-albedo mode
        if config.use_uniform_albedo:
             self.names.append("Delta phase (deg)")
             self.bounds.append((-54, 54)) # degrees, comes from 0.15 phase uncertainty on the RV time of sup. conj.

        # Gravitational amplitudes (free or fixed)
        if config.fit_gravitational_effects:
            self.names += ["Abeam", "Aellip"]
            self.bounds += [(1e-2, 500), (-500, 500)]

        #self.names += ["log jitter"]
        #self.bounds += [(-4, 6)]

    def prior_transform(self, u):
        values = []

        for ui, (low, high), name in zip(u, self.bounds, self.names):

            # Log-scale for jitter
            #if name in ["jitter"]:
                #val = 10 ** (np.log10(low) + ui * (np.log10(high) - np.log10(low)))

            # Gaussian prior for T0 offset (centered at T0)
            #elif name == "Delta T0 (days)":
            #    val = norm.ppf(ui, loc=0.0, scale=5 * 0.018)
            #    val = np.clip(val, low, high)

            # Linear priors

            val = low + ui * (high - low)

            values.append(val)

        return np.array(values)


class PhotometryModel:
    def __init__(self, config: FitConfiguration, interpolator, grid_axes, phase_model):
        self.config = config
        self.interpolator = interpolator
        self.grid_axes = grid_axes
        self.phase_model = phase_model
        self.param_space = None  # filled later

    def compute_gravitational_effects(self, inclination):
        alpha_ellip = -2.2e-4 * self.config.effectivetemperature + 2.6
        alpha_beam = -6e-4 * self.config.effectivetemperature + 7.2
        inc_rad = np.deg2rad(inclination)
        mass_jup = self.config.planetarymasssini * 0.00314558  # conversion to Jupiter mass

        Aellip = (13 * alpha_ellip * np.sin(inc_rad) *
                    self.config.stellarradius ** 3 *
                    self.config.stellarmass ** (-2) *
                    self.config.period ** (-2) *
                    mass_jup)

        Abeam = (2.7 * alpha_beam *
                    self.config.period ** (-1 / 3) *
                    self.config.stellarmass ** (-2 / 3) *
                    mass_jup)
        return Aellip, Abeam

    def flux(self, p, phase_obs):
        # Convert vector → dict
        parameters = dict(zip(self.param_space.names, p))

        # Determine inclination
        if self.config.fit_inclination:
            incl = parameters["Inclination"]  # Fitting for inclination
        else:
            incl = self.config.fixed_inclination  # Using fixed inclination value

        # Build interpolation
        interp_point = []
        for axis_name, axis_vals in self.grid_axes.items():
            if axis_name == "Inclination" and not self.config.fit_inclination:
                interp_point.append(incl)
            elif axis_name in parameters:
                interp_point.append(parameters[axis_name])
            elif self.config.use_uniform_albedo and axis_name in ["Albedo min"]:
                continue  # skip
            else:
                raise RuntimeError(f"Axis {axis_name} missing from parameters")

        interp_point = np.array([interp_point])



        # Interpolate planetary flux
        base_flux = self.interpolator(interp_point).squeeze()
        # Avoid having Nans returned when asked for a point outside the grid.
        if np.isnan(base_flux).any():
            # return something that gives extremely low loglike (bad point)
            return np.full_like(phase_obs, 1e6)  # huge mismatch = very low likelihood



        # Apply T0 shift phase
        if self.config.use_uniform_albedo:
             dT0 = parameters["Delta phase (deg)"]
             phase_obs = (phase_obs + dT0) % 360

        # Interpolate to observation phases
        f = np.interp(phase_obs, self.phase_model, base_flux)

        # Add amplitude offset
        f += parameters["Amplitude Offset"]


        # Add gravitational effects
        if self.config.fit_gravitational_effects:
            Abeam, Aellip = parameters["Abeam"], parameters["Aellip"]
        elif self.config.add_gravitational_model:
            Aellip, Abeam = self.compute_gravitational_effects(incl)
        else:
            Aellip = Abeam = 0.0

        f -= Aellip * np.cos(4 * np.pi * phase_obs / 360)
        f += Abeam * np.sin(2 * np.pi * phase_obs / 360)

        return f

def run_fit_least_square(config: FitConfiguration,
            grid_axes,
            interpolator,
            phase_model,
            foldx, foldy, phase_obs, yerr,
            output_folder):

    config.validate()

    # Build model + param space
    model = PhotometryModel(config, interpolator, grid_axes, phase_model)
    param_space = Priors(config, grid_axes)
    model.param_space = param_space

    print(f"Number of parameters: {len(param_space.names)}")


    def residuals(p, foldy, yerr, phase_obs):
        m = model.flux(p, phase_obs)
        m = np.squeeze(m)
        return (foldy - m) / yerr

    random_x0 = np.zeros(len(param_space.names))
    bounds_min = []
    bounds_max = []
    # do the fit
    rng = np.random.default_rng()
    for i, b in enumerate(param_space.bounds):
        bounds_min.append(b[0])
        bounds_max.append(b[1])

        random_x0[i] = rng.uniform(low=b[0], high=b[1])

    bounds = [np.asarray(bounds_min), np.asarray(bounds_max)]

    res = least_squares(residuals, random_x0, bounds=bounds,
                    args=(foldy, yerr, phase_obs))

    #  get the best parameters
    best = res.x
    best_model = model.flux(best, phase_obs)

    # get the confidence intervals
    def confidence_intervals(result):
        J = result.jac
        residual_var = np.sum(result.fun ** 2) / (len(result.fun) - len(result.x))
        cov = residual_var * inv(J.T @ J)
        sigmas = np.sqrt(np.diag(cov))

        return sigmas, cov

    sigmas, cov = confidence_intervals(res)

    print("\nBest-fit parameters:")
    for i, name in enumerate(param_space.names):
        print(f'{name} ± 1σ = {best[i]:.3f} ± {sigmas[i]:.3f}')


    np.savez_compressed(
        build_output_filename(output_folder, "fit_results.npz"),
        param_names=np.array(param_space.names),
        param_bounds= np.array(param_space.bounds),
        best_params=best,
        best_model=best_model,
        sigmas=sigmas,
    )

    return best, best_model, sigmas, param_space.names, param_space.bounds


def run_fit_ultranest(config: FitConfiguration,
            grid_axes,
            interpolator,
            phase_model,
            foldx, foldy, phase_obs, yerr,
            output_folder):

    config.validate()

    # Build model + param space
    model = PhotometryModel(config, interpolator, grid_axes, phase_model)
    param_space = Priors(config, grid_axes)
    model.param_space = param_space

    print(f"Number of parameters: {len(param_space.names)}")
    # Quick test for Nans in the code. If Nans, needs fixing.
    us = np.random.rand(len(param_space.names))
    print("prior transform test:", param_space.prior_transform(us))
    print("any NaN in prior transform?", np.isnan(param_space.prior_transform(us)).any())

    # Log-likelihood
    def loglike(p):
        m = model.flux(p, phase_obs)
        sigma = yerr
        r = (foldy - m) / sigma
        ll = -0.5 * np.sum(r * r + np.log(2 * np.pi * sigma**2))
        return ll

    #sampler = ultranest.ReactiveNestedSampler(
    #    param_space.names,
    #    loglike,
    #    param_space.prior_transform,
    #    log_dir=str(output_folder / "ultranest"),
    #    resume='overwrite',
    #)

    sampler = ultranest.NestedSampler(
        param_space.names,
        loglike,
        param_space.prior_transform,
        log_dir=str(output_folder / "ultranest"),
        num_live_points=400
    )

    sampler.stepsampler = SliceSampler(
        nsteps=12,
        generate_direction=generate_region_oriented_direction
    )

    #result = sampler.run(
    #    min_num_live_points=400,
    #    dlogz=0.5,
    #    frac_remain=1e-8,
    #    max_iters=10000
    #)

    result = sampler.run(
        max_iters=100000,
        dlogz=0.5,
    )

    sampler.print_results()

    # Extract results
    samples = result["samples"]
    best = samples.mean(axis=0)
    sigma = samples.std(axis=0)
    best_model = model.flux(best, phase_obs)

    # Save output
    np.savez_compressed(
        build_output_filename(output_folder, "fit_results.npz"),
        param_names=np.array(param_space.names),
        param_bounds= np.array(param_space.bounds),
        best_params=best,
        sigmas=sigma,
        samples=samples,
        best_model=best_model,
        ultranest_result=result
    )

    # Corner plot
    fig = corner.corner(samples, labels=param_space.names,
                        truths=best, show_titles=True, range=param_space.bounds)
    fig.savefig(build_output_filename(output_folder,"corner.png"), dpi=250)
    plt.close(fig)

    return (best, sigma, best_model, samples, param_space.names, param_space.bounds)

def load_model_grid(grid_folder, grid_filename):
    """
    Load a model grid (NPZ + JSON metadata) with any arbitrary filename.
    """
    grid_path = Path(grid_folder) / f"{grid_filename}.npz"
    meta_path = Path(grid_folder) / f"{grid_filename}.json"

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    grid = np.load(grid_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return grid, metadata


def build_output_folder(config: FitConfiguration, model_name: str, root_out: Path):
    """
    Build a fully descriptive folder name based on all fitting choices.
    Example:
        fit_target9139163__model_phaseoffset__i_fixed17__albedo_uniform__gravity_fixed
    """

    # incl
    if config.fit_inclination:
        inc_label = "i_fitted"
    else:
        inc_label = f"i_fixed{config.fixed_inclination}"

    # albedo
    alb_label = "albedo_uniform" if config.use_uniform_albedo else "albedo_nonuniform"

    # gravity
    if config.fit_gravitational_effects:
        grav_label = "gravity_fitted"
    elif config.add_gravitational_model:
        grav_label = "gravity_fixed"
    else:
        grav_label = "gravity_none"

    if config.fitting_method == 'LS':
        method = 'LS'
    elif config.fitting_method == 'NS':
        method = 'NS'

    # Target
    target_label = f"target{config.target}"
    # t0_name = 't05sigma'
    # t0_name = 't0'

    folder_name = "__".join([target_label, inc_label, alb_label, grav_label, method])

    out_folder = root_out / folder_name

    # If exists → stop
    existed = out_folder.exists()
    # Create directories
    if config.fitting_method == 'NS':
        (out_folder / "ultranest").mkdir(parents=True, exist_ok=True)
    else:
        out_folder.mkdir(parents=True, exist_ok=True)

    return out_folder, existed

def build_output_filename(base_folder: Path, base_name: str):
    """
    Use the folder name as a prefix for all output files.
    """
    prefix = base_folder.name
    return base_folder / f"{prefix}_{base_name}"

def plot_fit(foldx, time_array, foldy, phase_obs, yerr, phase_model, best_model, out):
    resi = best_model - foldy
    binsize = 50

    bres, _ = utils.bin_data(resi, binsize)
    btime, _ = utils.bin_data(phase_obs, binsize)
    bflux, berr = utils.bin_data(foldy, binsize, err=yerr)
    bin_time = utils.compute_binning_time(time_array, binsize)

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, dpi=100)
    ax[0].plot(phase_obs / 360, best_model, color='deepskyblue', linewidth=3, zorder=3, alpha=1)  #colors_matter[3]
    ax[0].errorbar(phase_obs / 360, foldy, yerr=yerr, linestyle=' ', fmt='.',
                   color='lightgrey', zorder=1,
                   alpha=0.1)
    ax[0].errorbar(btime / 360, bflux, yerr=berr, linestyle=' ', fmt='o', markersize=5,
                   markerfacecolor=colors_matter[1], markeredgecolor='black', markeredgewidth=0.3, zorder=2,
                   alpha=1)
    ax[0].set_ylabel('Normalised lightcurve [ppm]')
    ax[0].axvline(0.5, linestyle='--', linewidth=1.5, color='grey')
    ax[0].text(0.52, -90, s='Time of inferior \n conjunction', fontfamily='sans-serif', color='grey', fontsize=12)
    #full_data_label = lines.Line2D([], [], color='lightgrey',
                                   #markersize=5, marker='.', label='full data')
    fit_label = lines.Line2D([], [], color='deepskyblue',
                             linewidth=2, label=f'best-fit model'
    )
    #fit_label = mpatches.Rectangle((40, 90), 10, 10,edgecolor='black', facecolor=colors_matter[3], linewidth=1, label=f'best fit model')
    binned_data_label = lines.Line2D([], [], marker='o', markersize=6,
                                     markerfacecolor=colors_matter[1],
                                     markeredgecolor='black', markeredgewidth=0.5, linestyle=' ',
                                     label=f'{binsize}-point binning')
    ax[0].legend(handles=[fit_label, binned_data_label], ncols=2,
                prop={'family': 'sans-serif', 'size':12}, labelcolor='black', loc="upper right",
                frameon=False)  #facecolor='darkblue', framealpha=0.5
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-100, 100)
    ax[0].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax[0].tick_params(which="major", length=8, width=1.4, labelsize=16)
    ax[0].tick_params(which="minor", length=4, width=1.0)
    ax[0].minorticks_on()

    ax[1].scatter(phase_obs / 360, resi,
                  alpha=0.1, color='lightgrey', s=8, marker='o')
    ax[1].scatter(btime / 360, bres,
                  alpha=1, c=colors_matter[1], edgecolors = 'black', linewidths = 0.3, s=18, marker='o')
    ax[1].axhline(y=0, xmin=0, xmax=360, linewidth=2.5, color='deepskyblue')
    ax[1].axvline(0.5, linestyle='--', linewidth=1.5, color='grey')
    ax[1].set_xlabel('Orbital phase')  # Time [day]
    ax[1].set_ylabel('Residuals [ppm]')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(-100, 100)
    ax[1].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax[1].tick_params(which="major", length=8, width=1.4, labelsize=16)
    ax[1].tick_params(which="minor", length=4, width=1.0)
    ax[1].minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(out, f'best_fit_inc.png'), format='png', dpi=300,
                bbox_inches='tight')
    plt.show()


def load_parfile(parfile_path):
    with open(parfile_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def cfg_from_yaml(yaml_dict, use_namespace=True):
    if use_namespace:
        def dict_to_namespace(d):
            ns = SimpleNamespace()
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(ns, k, dict_to_namespace(v))
                else:
                    setattr(ns, k, v)
            return ns
        return dict_to_namespace(yaml_dict)
    else:
        return yaml_dict

def save_parfile_to_output(parfile_path, output_folder):
    parfile_path = Path(parfile_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dest_path = output_folder / parfile_path.name

    if dest_path.exists():
        print(f"Parfile already exists in output folder: {dest_path}")
    else:
        shutil.copy2(parfile_path, dest_path)
        print(f"Parfile saved to: {dest_path}")
    return dest_path

def try_load_previous_fit(out_folder):
    npz_path = list(out_folder.glob("*__fit_results.npz"))
    if len(npz_path) == 0:
        return None  # nothing to load

    path = npz_path[0]
    print(f"Loading previous fit: {path}")

    data = np.load(path, allow_pickle=True)

    return (
        data["best_params"],
        data["sigmas"],
        data["best_model"],
        data["samples"],
        data["param_names"],
        data["param_bounds"],
    )


def main():
    ROOT = Path(__file__).resolve().parent
    if path_exists(ROOT.parent / "model"/ "fit_parameters.yaml"):
        parfile_path = ROOT.parent / "model" / "fit_parameters.yaml"

    else:
        print(f"Cannot locate 'fit_parameters.yaml' parfile in {ROOT.parent}")
        sys.exit()
    references_path = ROOT.parent / "references"

    # Load parfile
    params = load_parfile(parfile_path)

    # Configuration
    cfg = cfg_from_yaml(params)

    config = FitConfiguration(
        use_uniform_albedo=cfg.fitting.use_uniform_albedo,
        fit_inclination=cfg.fitting.fit_inclination,
        fixed_inclination=cfg.fitting.fixed_inclination,
        fit_gravitational_effects=cfg.fitting.fit_gravitational_effects,
        add_gravitational_model=cfg.fitting.add_gravitational_model,
        period=cfg.target.period,
        effectivetemperature=cfg.stellar.teff,
        stellarmass=cfg.stellar.mass,
        stellarradius=cfg.stellar.radius,
        planetarymasssini=cfg.target.planetarymasssini,
        target=cfg.target.name,
        fitting_method = cfg.fitting.fitting_method
    )
    # Load grid
    grid_folder = ROOT.parent / cfg.model.folder
    grid_filename = cfg.model.filename

    output_root = ROOT.parent / cfg.output.root_folder
    out_folder, existed = build_output_folder(config, grid_filename, output_root)

    grid, metadata = load_model_grid(grid_folder, grid_filename)

    grid_axes = {
        "Planetary Radius": grid["planetaryradius"],
        "Albedo": grid["albedo"],
        "Redistribution": grid["redistribution"],
        "Inclination": np.asarray(grid["inclination"]),
        "Cloud offset": np.asarray(grid["cloud_offset"])
    }

    if not config.use_uniform_albedo:
        grid_axes["Albedo min"] = np.asarray(grid["albedo_min"])
        grid_axes["Cloud offset"] = np.asarray(grid["cloud_offset"])

    # Drop unused inclinations
    normalized_flux = grid["flux"]

    print(f'Inclination values in the grid: {grid_axes["Inclination"]}')

    if not config.fit_inclination:
        incl = config.fixed_inclination
        idx = np.nanargmin(np.abs(grid_axes["Inclination"] - incl))
        # removing inclination and albedo_min from normalized flux to allow interpolation
        if config.use_uniform_albedo:
            normalized_flux = normalized_flux[:, :, :, idx, 0, :, :]
        else:
            normalized_flux = normalized_flux[:, :, :, idx, :, :, :]
        grid_axes.pop("Inclination")
        #grid_axes.pop("Cloud offset")
    else:
        if config.use_uniform_albedo:
            normalized_flux = normalized_flux[:, :, :, :, 0, :, :]
        else:
            normalized_flux = normalized_flux[:, :, :, :, :, :, :]

    print("normalized_flux.shape before:", normalized_flux.shape)
    for k,v in grid_axes.items():
        print(k, len(v))
    # After slicing:
    print("normalized_flux.shape after:", normalized_flux.shape)
    #assert normalized_flux.ndim == len(grid_axes) - (0 if config.fit_inclination else 1), "Dimension mismatch"
    
    interpolator = RegularGridInterpolator(
        tuple(grid_axes.values()), normalized_flux,
        bounds_error=False, fill_value=np.nan
    )

    # Load photometry data
    if path_exists(references_path / cfg.data.filename):
        data = np.loadtxt(references_path / cfg.data.filename)
    else:
        print('Cannot locate data file. Change configuration file.')
        sys.exit()

    time_array, y = data[:, 0], data[:, 1]

    # Phase-fold data
    tsup_RV = cfg.data.tsup_RV # time of superior conjunction
    t0_Kepler = cfg.data.t0_photometry # first time stamps in Kepler time array
    time_abs = time_array + t0_Kepler

    foldx, foldy = utils.phase_fold(time_abs, y, config.period, tsup_RV)

    phase_obs = ((time_array - tsup_RV) / config.period % 1.0) * 360

    # Uncertainties
    ref_table = pd.read_csv(os.path.join(references_path,  "t0_estimation.csv"))
    target = ref_table.loc[ref_table['KIC'] == int(config.target)]
    yerr = np.repeat(np.asarray(target['noise_jenkins'])[0], len(y))
    config.yerr = yerr

    phase_model = np.linspace(0, 360, 100)

    if existed:
        print("\nFit directory already exists — attempting to load stored results.")
        loaded = try_load_previous_fit(out_folder)
        if loaded is not None:
            best, sigma, best_model, samples, param_name, param_bounds = loaded
            print("Loaded previous fit. Skipping UltraNest.")
        else:
            print("Folder exists, but no previous fit file found. Continuing with fresh fit.")

            if cfg.fitting.fitting_method == 'LS':
                print('Fitting method is Least Square.')
                best, best_model, sigmas, param_name, param_bounds = run_fit_least_square(config, grid_axes,
                                                                                  interpolator, phase_model,
                                                                                  foldx, foldy, phase_obs,
                                                                                  yerr,
                                                                                  output_folder=out_folder)

            elif cfg.fitting.fitting_method == 'NS':
                print('Fitting method is Nested Sampling.')
                best, sigma, best_model, samples, param_name, param_bounds = run_fit_ultranest(config, grid_axes,
                                            interpolator, phase_model,
                                            foldx, foldy, phase_obs,
                                            yerr,
                                            output_folder=out_folder)
            else:
                print('Fitting method is not recognized. Please change it.')
                sys.exit()

    else:
        if cfg.fitting.fitting_method == 'LS':
            print('Fitting method is Least Square.')
            best, best_model, sigmas, param_name, param_bounds = run_fit_least_square(config, grid_axes,
                                                                              interpolator, phase_model,
                                                                              foldx, foldy, phase_obs,
                                                                              yerr,
                                                                              output_folder=out_folder)

        elif cfg.fitting.fitting_method == 'NS':
            print('Fitting method is Nested Sampling.')
            best, sigma, best_model, samples, param_name, param_bounds = run_fit_ultranest(config, grid_axes,
                                                                                           interpolator, phase_model,
                                                                                           foldx, foldy, phase_obs,
                                                                                           yerr,
                                                                                           output_folder=out_folder)
        else:
            print('Fitting method is not recognized. Please change it.')
            sys.exit()

    plot_fit(foldx, time_array, foldy, phase_obs, yerr, phase_model, best_model, out_folder)

    saved_parfile = save_parfile_to_output(parfile_path, out_folder)
    print(f"Stored parfile at: {saved_parfile}")


if __name__ == "__main__":
    main()
    print('done')



