import matplotlib.pyplot as plt
import numpy as np
import cmocean as cm
import os
import astropy.constants as c
from matplotlib import rc
from mpl_toolkits.basemap import Basemap
import utils
import matplotlib.lines as mlines

rc('image', origin='lower')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 16})
rc('text', usetex=False)
rc('lines', linewidth=0.5)
rc('ytick', right=True, direction='in')
rc('xtick', top=True, direction='in')
rc('axes', axisbelow=False)
rc('mathtext', fontset='cm')

colors_matter = cm.cm.matter_r(np.linspace(0, 1, 10))

results_folder = '/Users/ah258874/PycharmProjects/punto/results_model'


def plot_photometry_model(system, normalized_flux,  planetaryradius, albedo, recirculation):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)

    ax.axvline(0, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)
    ax.axvline(90, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)
    ax.axvline(180, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)
    ax.axvline(270, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)
    ax.axvline(360, -10, 70, color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)
    for i, radius in enumerate(planetaryradius):
        for j, albedovalue in enumerate(albedo):
            for k, recirculationvalue in enumerate(recirculation):
                mask = ~np.isnan(normalized_flux)
                if np.any(mask[i, j, k, :]):
                    ax.plot(system.phase * 180 / np.pi, normalized_flux[i, j, k, :], linewidth=0.8, linestyle='-',
                            label=f'{np.round(radius / (c.R_jup).value, 2)} Rj'
                                  f' Ab {np.round(albedovalue, 1)} r {np.round(recirculationvalue, 1)}',
                            color=colors_matter[i])

    xticks = [0, 90, 180, 270, 360]
    xlabels = [f'{x}°' for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_xlabel('Phase angle')
    ax.set_ylabel('Amplitude [ppm]')

    plt.legend(prop={"size": 7}, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'photometrymodel.png'), format='png', dpi=300)
    plt.show()


def plot_temperature_map(system):
    m = Basemap(projection='mbtfpq', lon_0=0, resolution='c')

    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 20.))
    m.drawmeridians(np.arange(0., 360., 20.))
    m.drawmapboundary(fill_color='white')

    # Convert longitude and latitude to degrees
    lon_centers = system.longitudearray * 180 / np.pi
    lat_centers = system.latitudearray * 180 / np.pi

    # Compute edges
    lon_edges = utils.compute_edges(lon_centers)
    lat_edges = utils.compute_edges(lat_centers)

    # Create meshgrid of edges
    lons, lats = np.meshgrid(lon_edges, lat_edges)

    im1 = m.pcolormesh(lons, lats, system.atmospherictemperature, cmap=cm.cm.matter_r, latlon=True, shading='auto')
    cb = m.colorbar(im1, "bottom", size="5%", pad="2%")
    cb.set_label('Temperature [K]')
    plt.title(f"Temperature map {system.planetaryradius / (c.R_jup).value} Rj, albedo {system.albedo}, "
              f"recirculation {system.redistribution}")
    plt.savefig(os.path.join(results_folder, f'temperaturemap_{system.planetaryradius / (c.R_jup).value}Rj'
                                             f' Ab_{system.albedo}_r_{system.redistribution}.png'),  dpi=300)
    plt.show()


