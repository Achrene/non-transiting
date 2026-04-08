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
from scipy.stats import norm
from dataclasses import dataclass
import sys
import os
import pandas as pd

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
            if config.use_uniform_albedo and name in ["Albedo min", "Cloud offset"]:
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
            self.names.append("Delta T0 (days)")
            self.bounds.append((-0.1, 0.1))

        # Gravitational amplitudes (free or fixed)
        if config.fit_gravitational_effects:
            self.names += ["Abeam", "Aellip"]
            self.bounds += [(1e-2, 500), (-500, 500)]


    def prior_transform(self, u):
        values = []

        for ui, (low, high), name in zip(u, self.bounds, self.names):

            # Log-scale for grav amplitudes
            if name in ["Abeam", "Aellip"]:
                val = 10 ** (np.log10(low) + ui * (np.log10(high) - np.log10(low)))

            # Gaussian prior for T0 offset (centered at T0)
            elif name == "Delta T0 (days)":
                val = norm.ppf(ui, loc=0.0, scale=5 * 0.018)
                val = np.clip(val, low, high)

            # Linear priors
            else:
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
        self.config.planetarymasssini = self.config.planetarymasssini * 0.00314558  # conversion to Jupiter mass

        Aellip = (13 * alpha_ellip * np.sin(inc_rad) *
                  self.config.stellarradius ** 3 *
                  self.config.stellarmass ** (-2) *
                  self.config.period ** (-2) *
                  self.config.planetarymasssini)

        Abeam = (2.7 * alpha_beam *
                 self.config.period ** (-1 / 3) *
                 self.config.stellarmass ** (-2 / 3) *
                 self.config.planetarymasssini)
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
            elif self.config.use_uniform_albedo and axis_name in ["Albedo min", "Cloud offset"]:
                continue  # skip
            else:
                raise RuntimeError(f"Axis {axis_name} missing from parameters")

        interp_point = np.array([interp_point])

        # Interpolate planetary flux
        base_flux = self.interpolator(interp_point).squeeze()

        # Shift phase (uniform albedo case only)
        if self.config.use_uniform_albedo:
            dT0 = parameters["Delta T0 (days)"]
            shift = dT0 / self.config.period * 360
            phase_obs = (phase_obs + shift) % 360

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

def run_fit(config: FitConfiguration,
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

    # Log-likelihood
    def loglike(p):
        m = model.flux(p, phase_obs)
        r = (foldy - m) / yerr
        ll = -0.5 * np.sum(r * r + np.log(2 * np.pi * yerr * yerr))
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
        resume='overwrite',
        num_live_points=2000
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
        max_iters=10000,
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

    return best, sigma, best_model, samples

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

    # Target
    target_label = f"target{config.target}"

    folder_name = "__".join([target_label, inc_label, alb_label, grav_label])

    out_folder = root_out / folder_name

    # If exists → stop
    existed = out_folder.exists()
    # Create directories
    (out_folder / "ultranest").mkdir(parents=True, exist_ok=True)

    return out_folder, existed

def build_output_filename(base_folder: Path, base_name: str):
    """
    Use the folder name as a prefix for all output files.
    """
    prefix = base_folder.name
    return base_folder / f"{prefix}__{base_name}"

def plot_fit(foldx, foldy, phase_obs, yerr, phase_model, best_model, out):
    resi = best_model - foldy
    binsize = 50

    bres, _ = utils.bin_data(resi, binsize)
    btime, _ = utils.bin_data(phase_obs, binsize)
    bflux, berr = utils.bin_data(foldy, binsize, err=yerr)

    fig, ax = plt.subplots(2, 1, figsize=(7, 6),
                           gridspec_kw={'height_ratios': [3, 1]},
                           sharex=True)

    ax[0].plot(phase_obs / 360, best_model, color=colors_matter[3], lw=1.3, zorder=100)
    ax[0].errorbar(phase_obs / 360, foldy, yerr=yerr, fmt='.', color='lightgrey', alpha=0.5, zorder=-500)
    ax[0].errorbar(btime / 360, bflux, yerr=berr, fmt='.', color=colors_matter[1], zorder=50)

    ax[0].set_ylabel("Normalized flux [ppm]")

    ax[1].scatter(phase_obs / 360, resi, s=8, color="lightgrey", alpha=0.5)
    ax[1].scatter(btime / 360, bres, s=8, color=colors_matter[1])
    ax[1].axhline(0, color=colors_matter[3], lw=1)

    ax[1].set_xlabel("Orbital phase")
    ax[1].set_ylabel("Residuals [ppm]")

    plt.tight_layout()
    plt.savefig(build_output_filename(out,"fit.png"), dpi=250)
    plt.close(fig)

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
        data["samples"]
    )


def main():
    ROOT = Path(__file__).resolve().parent
    if path_exists(ROOT.parent / "model"/ "fit_parameters.yaml"):
        parfile_path = ROOT.parent / "model" / "fit_parameters.yaml"
        print('I am here')
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
        target=cfg.target.name
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

    if not config.fit_inclination:
        incl = config.fixed_inclination
        idx = np.nanargmin(np.abs(grid_axes["Inclination"] - incl))
        normalized_flux = normalized_flux[:, :, :, idx:idx+1, 0, :]
        grid_axes.pop("Inclination")
        grid_axes.pop("Cloud offset")

    print(grid_axes)
    interpolator = RegularGridInterpolator(
        tuple(grid_axes.values()), normalized_flux,
        bounds_error=False, fill_value=None
    )

    # Load photometry data
    if path_exists(references_path / cfg.data.filename):
        data = np.loadtxt(references_path / cfg.data.filename)
    else:
        print('Cannot locate data file. Change configuration file.')
        sys.exit()

    time_array, y = data[:, 0], data[:, 1]

    # Phase-fold data
    t0_RV = cfg.data.t0_RV
    t0_Kepler = cfg.data.t0_photometry
    ref_time = t0_RV - t0_Kepler

    foldx, foldy = utils.phase_fold(time_array, y, config.period, ref_time)
    phase_obs = ((foldx + 0.5 * config.period) / config.period) * 360

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
            best, sigma, best_model, samples = loaded
            print("Loaded previous fit. Skipping UltraNest.")

        else:
            print("Folder exists, but no previous fit file found. Continuing with fresh fit.")

            # Run fit
            best, sigma, best_model, samples = run_fit(
                config,
                grid_axes,
                interpolator,
                phase_model,
                foldx, foldy, phase_obs, yerr,
                output_folder=out_folder
            )

    # Plot fit
    plot_fit(foldx, foldy, phase_obs, yerr, phase_model, best_model, out_folder)

    saved_parfile = save_parfile_to_output(parfile_path, out_folder)
    print(f"Stored parfile at: {saved_parfile}")

if __name__ == "__main__":
    main()
    print('done')



