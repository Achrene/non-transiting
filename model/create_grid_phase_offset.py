import numpy as  np
import os
import matplotlib.pyplot as plt
import json
import sys
import ExoplanetarySystem_phaseoffset
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

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

colors_matter = cm.cm.matter_r(np.linspace(0,8,10))

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



ROOT = Path(__file__).resolve().parent
references_path = str(ROOT / "references")
model_grid_folder = str(ROOT / "results_grid" / "model_grid" / "phase_offset")
model_name = 'model_grid_phase_offset'

Kepler = True
TESS = False
plot_grid = True
create_grid = True
use_uniform_albedo = True

#  inputs
targetname = '9139163'
period = 0.604734  # in days
nphase = 100  # discretization of phases
phase_model = np.linspace(0, 360., nphase)
wavearray = np.array([0.430, 0.890])  # Kepler bandpass in micron
planetarymasssini = 11.4  # Earth mass, from RV fit
planetarymasssini = planetarymasssini * 0.00314558  # conversion into Jupiter mass -> not used in the mode right now
effectivetemperature = 6358
stellarmass = 1.390
stellarradius = 1.558


if create_grid:
    if os.path.exists(os.path.join(model_grid_folder, f'{model_name}.npz')):
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
    planetaryradius = np.linspace(0.2, 0.3, 3)  #1
    albedo = np.linspace(0.5, 0.99, 2)
    redistribution = np.linspace(0.05, 0.1, 2) # 0.99
    inclination = np.linspace(55, 62, 2)  #1
    if use_uniform_albedo:
        albedo_min = albedo.copy()
    else:
        albedo_min = np.linspace(0.1, 0.5, 3)
    cloud_offset = np.linspace(30, 120, 2)  # degrees


    # creating the grid of model
    def run_model(planetaryradius, albedo, redistribution, inclination, albedo_min, cloud_offset):
        punto_system = ExoplanetarySystem_phaseoffset.ExoplanetarySystem_phaseoffset(orbitalperiod=period,
                                                      effectivetemperature=effectivetemperature,
                                                      stellarmass=stellarmass,
                                                      stellarradius=stellarradius,
                                                      semimajoraxis=None, planetaryradius=planetaryradius,
                                                      planetarymass=planetarymasssini, inclination=inclination,
                                                      redistribution=redistribution,
                                                      albedo=albedo,
                                                      wavearray=wavearray,
                                                      longitudearray=None, latitudearray=None, checking=False,
                                                      internaltemperature=100, area=None, phase=phase_model,
                                                      atmospherictemperature=None,
                                                      totalplanetartintensity=None,
                                                      emittedplanetaryintensity=None,
                                                      reflectedplanetartintensity=None,
                                                      contrast_ppm=None, contrast_ppm_refl=None,
                                                      mission='Kepler', response_nu=None, response_vals=None,
                                                      cloud_offset=cloud_offset,
                                                      albedo_min=albedo_min)
        return punto_system.compute_flux()


    Rp, A, Re, inc, Amin, Coffset = np.meshgrid(planetaryradius, albedo,
                                                redistribution, inclination, albedo_min, cloud_offset, indexing='ij')
    points = list(zip(Rp.flatten(), A.flatten(), Re.flatten(), inc.flatten(), Amin.flatten(), Coffset.flatten()))
    num_points = (len(planetaryradius) * len(albedo) * len(redistribution) *
                  len(inclination) * len(albedo_min) * len(cloud_offset))
    # parallel computation

    # Run in parallel with progress bar
    WHITE = '\033[97m'
    RESET = '\033[0m'
    with tqdm_joblib(tqdm(desc=f"{WHITE}Computing grid{RESET}", total=num_points)) as progress_bar:
        results = Parallel(n_jobs=-1)(delayed(run_model)(r, a, re, i, amin, c) for r, a, re, i, amin, c in points)

    normalized_flux = np.array(results).reshape(len(planetaryradius), len(albedo), len(redistribution),
                                                len(inclination), len(albedo_min), len(cloud_offset), len(phase_model))

    # Saving the model
    np.savez_compressed(os.path.join(model_grid_folder, f"{model_name}.npz"),
                        planetaryradius=planetaryradius,
                        albedo=albedo,
                        redistribution=redistribution,
                        inclination=inclination,
                        albedo_min=albedo_min,
                        cloud_offset=cloud_offset,
                        flux=normalized_flux)

    # Save metadata
    metadata = {
        "planetaryradius": {"name": "Planetary radius", "unit": "Jupiter radii"},
        "albedo": {"name": "albedo", "unit": "dimensionless"},
        "redistribution": {"name": "redistribution", "unit": "dimensionless"},
        "inclination": {"name": "inclination", "unit": "degrees"},
        "albedo_min": {"name": "albedo_min", "unit": "dimensionless"},
        "cloud_offset": {"name": "cloud_offset", "unit": "degrees"},
        "flux": "Normalized flux as a function of the phase angle",
        "author": "A. Dyrek",
        "date_created": datetime.now().isoformat(),
        "description": "Precomputed 3D model grid phase-curve normalised flux with phase-offset.",

        "fixed_parameters": {"target name": f"KIC {targetname}",
                             "period": f"{period} day",
                             "wavelength range": f"{wavearray} micron (Kepler Bandpass)",
                             "phase discretisation": f"{nphase} values (in degrees)",
                             "Planetary mass * sini": f"{planetarymasssini} Jupiter mass * sini"
                             }

    }
    with open(os.path.join(model_grid_folder, f"{model_name}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

else:
    print("'create_grid' is False: reading the model grid.")
    # Load the saved grid
    model_grid = np.load(os.path.join(model_grid_folder, f"{model_name}.npz"))

    planetaryradius = model_grid["planetaryradius"]
    albedo = model_grid["albedo"]
    redistribution = model_grid["redistribution"]
    inclination = model_grid["inclination"]
    inclination = np.asarray(inclination)
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
    inclination = np.asarray(inclination)

    def plot_photometry_model(normalized_flux):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
        for angle in [0, 90, 180, 270, 360]:
            ax.axvline(angle, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)

        # Flatten the parameter combinations
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

        #legend_text = (
            #f"$i = [{np.min(inclination)}, {np.max(inclination)}]^\circ$\n"
            #f"$R_\\mathrm{{p}}$ = [{np.min(planetaryradius):.2f}, {np.max(planetaryradius):.2f}] $R_\\mathrm{{J}}$\n"
            #f"Albedo = [{np.min(albedo):.2f}, {np.max(albedo):.2f}]\n"
            #f"Redist. = [{np.min(redistribution):.2f}, {np.max(redistribution):.2f}]\n"
            #f"Discretisation = [{n_r}, {n_a}, {n_rcl}, {n_inc}]"
        #)
        #ax.text(0.65, 0.98, legend_text,
                #transform=ax.transAxes,
                #fontsize=8,
                #verticalalignment='top',
                #horizontalalignment='left',
                #bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))
        plt.tight_layout()
        plt.savefig(os.path.join(model_grid_folder, f'{model_name}.png'),  format='png', dpi=100, bbox_inches='tight')
        plt.show()


    plot_photometry_model(normalized_flux)
else:
    pass

print('done')


