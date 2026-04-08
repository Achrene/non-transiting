import numpy as np
import astropy.constants as c
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import utils
import sys
from mpl_toolkits.basemap import Basemap
import cmocean as cm
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
from pathlib import Path
import os
import time
import pandas as pd

colors_matter = cm.cm.matter_r(np.linspace(0, 1, 10))

test = False

ROOT = Path(__file__).resolve().parent
references_path = str(ROOT.parent / "references")

def read_response_function(mission):
    if mission == 'Kepler':
        response_function_file = os.path.join(references_path, 'Kepler_Response_Function.txt')
        response_fonction_data = np.loadtxt(response_function_file, skiprows=8)
        response_function_values = response_fonction_data[:, 1]
        response_function_wavelength = response_fonction_data[:, 0] * 10 ** (-3)  # converted into microns

        response_lambda_cm = response_function_wavelength * 1e-4  # micron -> cm
        response_nu_cm1 = 1.0 / response_lambda_cm  # cm^-1

        sort_idx = np.argsort(response_nu_cm1)
        response_nu = response_nu_cm1[sort_idx]
        response_vals = response_function_values[sort_idx]

    elif mission == 'TESS':
        response_function_file = os.path.join(references_path, 'tess-response-function-v2.0.csv')
        response_fonction_data = pd.read_csv(response_function_file, skiprows=6)
        response_function_values = np.asarray(response_fonction_data[' lambda Transmission'])
        response_function_wavelength = np.asarray(response_fonction_data['# Wavelength (nm)']) * 10 ** (-3)  # converted into microns
        response_lambda_cm = response_function_wavelength * 1e-4  # micron -> cm
        response_nu_cm1 = 1.0 / response_lambda_cm  # cm^-1

        sort_idx = np.argsort(response_nu_cm1)
        response_nu = response_nu_cm1[sort_idx]
        response_vals = response_function_values[sort_idx]

    elif mission == 'None':
        print('No mission name provided: response function will be squared.')
        response_nu = None
        response_vals = None
    else:
        print('Mission name not recognised, response function cannot be processed.')
        sys.exit()

    return response_nu, response_vals


def integrate_with_response_function(wavelengths, response, B_lambda):
    """Integrate the Planck function weighted by the response function."""
    A = integrate.simpson(y=B_lambda * response, x=wavelengths)
    B = integrate.simpson(y=response, x=wavelengths)
    return A / B

def convert_mu_to_cm(wavearray):
    wavearray = 1 / (wavearray * 1e-4)  # from micron to cm-1
    wavearray = np.flip(wavearray)

    return wavearray

def convert_deg_to_radian(x):
    return x * (np.pi/180)

def convert_orbital_parameters(stellarmass, stellarradius, planetaryradius, orbitalperiod):
    stellarmass = (stellarmass * c.M_sun).value  # in kg
    stellarradius = (stellarradius * c.R_sun).value  # in m
    planetaryradius = (planetaryradius * c.R_jup).value  # in m
    orbitalperiod = orbitalperiod * 24 * 3600  # in seconds

    return stellarmass, stellarradius, planetaryradius, orbitalperiod

def compute_semi_major_axis(orbitalperiod, stellarmass, checking=False):
    semimajoraxis = ((orbitalperiod ** 2 * c.G.value * stellarmass) /
                          (4 * (np.pi ** 2))) ** (1 / 3)  # in m

    if checking is True:
        print(f'Semi-major axis is: {semimajoraxis * 6.68459e-12} UA.')

    return semimajoraxis

def compute_longitude_latitude():
    # Sphere discretisation

    numberoflongitude = 180 #90
    numberoflatitude = 90 #60

    lon_deg_limits = np.linspace(-180., 180., numberoflongitude + 1)
    lon_deg = (lon_deg_limits[0:numberoflongitude] + lon_deg_limits[1:numberoflongitude + 1]) * 0.5
    lat_deg_limits = np.linspace(-90., 90., numberoflatitude + 1)
    lat_deg = (lat_deg_limits[0:numberoflatitude] + lat_deg_limits[1:numberoflatitude + 1]) * 0.5

    longitudearray = convert_deg_to_radian(lon_deg)  # in radian
    latitudearray = convert_deg_to_radian(lat_deg)  # in radian

    return longitudearray, latitudearray

def check_area(latitudearray, longitudearray, checking=False):
    area = np.zeros((len(latitudearray), len(longitudearray)))
    dlat = latitudearray[1] - latitudearray[0];
    dlon = longitudearray[1] -longitudearray[0]

    area[-1, :] = dlon * (1. - np.sin(latitudearray[-1] - dlat / 2.))
    area[0, :] = area[len(latitudearray) - 1, :]

    for ilat in range(len(latitudearray) - 2):
        area[ilat + 1, :] = dlon * (np.sin(latitudearray[ilat + 1] + dlat / 2.) -
                                    np.sin(latitudearray[ilat + 1] - dlat / 2.))
        # area in planetary radius unit of surface cells
        # depends on latitude only
    if checking is True:
        print("Area must be around 1 :", area.sum() / (4 * np.pi))

    return area

def compute_Planck_law(nu_edges, temperature, numberofterms=30):
    """from exo_k Bnu_integral_num()
    https://perso.astrophy.u-bordeaux.fr/~jleconte/exo_k-doc/_modules/exo_k/util/radiation.html#Bnu_integral_num
    """

    # c.h Planck constant J s
    # c.c speed of light m/s
    # c.k_B Boltzmann constant J/K

    c1 = (2. * c.h * c.c ** 2).value  # first radiation constant
    c2 = ((c.h * c.c) / c.k_B).value  # second radiation constant
    kp = c2 / temperature

    blackbodyintensity = np.zeros(nu_edges.size)
    edges_size = nu_edges.size

    for i in range(edges_size):
        kpnu = kp * nu_edges[i] * 1.e2
        for n in range(1, numberofterms + 1):
            kpnun = kpnu * n
            blackbodyintensity[i] += np.exp(-kpnun) * (6. + 6. * kpnun + 3. * kpnun ** 2 + kpnun ** 3) / n ** 4

    for i in range(edges_size - 1):
        blackbodyintensity[i] -= blackbodyintensity[i + 1]

    return (blackbodyintensity * c1) / kp ** 4   # blackbodyintensity[:-1]

def fun(xi, P, epsilon=1):
    """
    The right hand term of the differential
    equation.

    Parameters
    ----------
    xi : ndarray
      the longitude, counted to be -pi/2 at the
      dawn terminator, 0 at the substellar point
      and pi/2 at the dusk terminator.

    P : ndarray
      the thermal phase.

    epsilon : float
      the redistribution factor
    """
    return 1 / epsilon * (.5 * (np.cos(xi) + np.abs(np.cos(xi))) - P ** 4)


def compute_thermal_phase(xi, epsilon=1, checking=False):
    """
    Parameters
    ----------
    xi : ndarray
      the longitude, counted to be -pi/2 at the
      dawn terminator, 0 at the substellar point
      and pi/2 at the dusk terminator.

    epsilon : float
      the redistribution factor
    """
    Pdawn = (np.pi + (3 * np.pi / epsilon) ** (4 / 3)) ** (-1 / 4)
    if xi[0] != - np.pi / 2:
        raise Exception("Integration must start at dawn terminator xi=-pi/2.")
    xi_span = (xi[0], xi[-1])
    result = solve_ivp(fun, xi_span, np.atleast_1d(Pdawn),
                       t_eval=xi, args=(epsilon,))
    if checking is True:
        print(result.status)
        print(result.message)
        print(result.success)
    return result.y[0]

def albedo_map(lon_array, albedo, albedo_min, cloud_offset):
    A_max = albedo  # from grid
    A_min = albedo_min  # from Webber+, 2015, figure 5
    theta_c = np.radians(cloud_offset)  # cloud offset
    sigma = np.radians(40)  # cloud width

    delta = (lon_array + theta_c + np.pi) % (2 * np.pi) - np.pi
    return A_min + (A_max - A_min) * np.exp(-delta ** 2 / (2 * sigma ** 2))

def plot_albedo_map(longitudearray, A_lon):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
    ax.plot(longitudearray * 180/np.pi, A_lon, color='green', linewidth=2)
    ax.axhline(A_lon[A_lon.shape[0]//2], color='lightpink', linestyle='--', linewidth=1)
    ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Longitude [$^{\circ}$]')
    ax.set_ylabel('Albedo $A_B$')
    ax.set_xlim(-180, 180)
    plt.tight_layout()
    plt.show()

def plot_reflected_radiance(longitudearray, latitudearray, reflectedplanetartintensity, reflection_mode):
    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            # Create a circle at the center of the legend box
            center = (xdescent + width / 2., ydescent + height / 2.)
            p = Circle(center, radius=min(width, height) / 2)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    m = Basemap(projection='mbtfpq', lon_0=0, resolution='c')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 20.))
    m.drawmeridians(np.arange(0., 360., 20.))
    m.drawmapboundary(fill_color='white')

    # Convert longitude and latitude to degrees
    lon_centers = longitudearray * 180 / np.pi
    lat_centers = latitudearray * 180 / np.pi

    # Compute edges
    lon_edges = utils.compute_edges(lon_centers)
    lat_edges = utils.compute_edges(lat_centers)

    # Create meshgrid of edges
    lons, lats = np.meshgrid(lon_edges, lat_edges)

    im1 = m.pcolormesh(lons, lats, reflectedplanetartintensity, cmap=cm.cm.matter_r, latlon=True, shading='auto')
    cb = m.colorbar(im1, "bottom", size="5%", pad="2%")
    cb.set_label(r'Reflected radiance $\mathcal{I}_{\rm refl}$ [W m$^2$ sr$^{-1}$]', fontsize=12)

    lon_labels = [lon_edges[0], lon_edges[len(lon_edges) // 2], lon_edges[-1]]
    for lon in lon_labels:
        x, y = m(lon, lat_edges[-1])  # top edge
        plt.text(x, y + 0.02 * (m.urcrnrx - m.llcrnrx), f"{lon:.0f}°",
                 ha='center', va='bottom', fontsize=10)

    plt.text(0.5, 1.15, "Longitude", transform=plt.gca().transAxes,
             ha='center', va='bottom', fontsize=12)

    plt.annotate(
        "West", xy=(0.25, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
        xytext=(0.35, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
        ha='left', va='center', fontsize=12, color='turquoise', fontfamily='sans-serif', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='turquoise', lw=1, shrinkA=0, shrinkB=0)
    )

    plt.annotate(
        "East", xy=(0.75, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
        xytext=(0.6, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
        ha='left', va='center', fontsize=12, color='turquoise', fontfamily='sans-serif', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='turquoise', lw=1, shrinkA=0, shrinkB=0)
    )

    # substellar point
    x0, y0 = m(0, 0)
    substellar_circle = Circle((x0, y0), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                               facecolor='cyan', edgecolor='black', linewidth=0.5, label='Substellar point',
                               zorder=5)
    plt.gca().add_patch(substellar_circle)

    # temperature maximum
    max_idx = np.unravel_index(np.argmax(reflectedplanetartintensity), reflectedplanetartintensity.shape)
    max_lon = lon_centers[max_idx[1]]
    if reflection_mode == 'grid':
        max_lat = lat_centers[max_idx[0]]
    elif reflection_mode == 'global':
        max_lat = 0
    x_max, y_max = m(max_lon, max_lat)
    max_circle = Circle((x_max, y_max), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                        facecolor='red', edgecolor='yellow', linewidth=0.5, label=r'$\mathcal{I}_{\rm{refl}, '
                                                                                  r'\rm{max}}$', zorder=5)
    plt.gca().add_patch(max_circle)

    plt.legend(handles=[substellar_circle, max_circle], loc='upper right',
               bbox_to_anchor=(1.05, 1.25), fontsize=10, handler_map={Circle: HandlerCircle()})

    # plt.savefig('/System/Volumes/Data/user/adyrek/non-transiting/'
    # 'results_grid/model_grid/phase_offset/reflection_global_map.png', format='png', dpi=300)

    plt.show()

def check_intensities_units(totalplanetartintensity, area, planetaryradius):
    # Radiance of specific intensity
    print(f'Radiance or specific intensity: self.totalplanetartintensity in W/m^2/sr.')
    # Power per steradian
    print(f'Power per steradian: self.totalplanetartintensity *'
          f' self.planetaryradius ** 2 in W/sr.')
    # Flux or irradiance
    print(f'Flux or irradiance: self.totalplanetartintensity *'
          f' self.area in W/m^2.')
    # Total power
    print(f'Total power: Sum str(self.totalplanetartintensity *'
          f' self.area * self.planetaryradius ** 2) in W.')
    total_rad = np.sum(totalplanetartintensity[:, :] * area[:, :]) * (planetaryradius ** 2)
    print(f'Total power: {total_rad} W.')

def prepare_grids():
    longitudearray, latitudearray = compute_longitude_latitude()
    area = check_area(latitudearray, longitudearray)
    return longitudearray, latitudearray, area

def plot_temperature_map(longitudearray, latitudearray, atmospherictemperature):

    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            # Create a circle at the center of the legend box
            center = (xdescent + width / 2., ydescent + height / 2.)
            p = Circle(center, radius=min(width, height) / 2)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    m = Basemap(projection='mbtfpq', lon_0=0, resolution='c')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 20.))
    m.drawmeridians(np.arange(0., 360., 20.))
    m.drawmapboundary(fill_color='white')

    # Convert longitude and latitude to degrees
    lon_centers = longitudearray * 180 / np.pi
    lat_centers = latitudearray * 180 / np.pi

    # Compute edges
    lon_edges = utils.compute_edges(lon_centers)
    lat_edges = utils.compute_edges(lat_centers)

    # Create meshgrid of edges
    lons, lats = np.meshgrid(lon_edges, lat_edges)

    im1 = m.pcolormesh(lons, lats, atmospherictemperature, cmap=cm.cm.matter_r, latlon=True, shading='auto')
    cb = m.colorbar(im1, "bottom", size="5%", pad="2%")
    cb.set_label(r'$T_{\mathrm{atm}}$ [K]', fontsize=12)

    lon_labels = [lon_edges[0], lon_edges[len(lon_edges) // 2], lon_edges[-1]]
    for lon in lon_labels:
        x, y = m(lon, lat_edges[-1])  # top edge
        plt.text(x, y + 0.02 * (m.urcrnrx - m.llcrnrx), f"{lon:.0f}°",
                 ha='center', va='bottom', fontsize=10)

    plt.text(0.5, 1.15, "Longitude", transform=plt.gca().transAxes,
             ha='center', va='bottom', fontsize=12)

    # substellar point
    x0, y0 = m(0, 0)
    substellar_circle = Circle((x0, y0), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                               facecolor='cyan', edgecolor='black', linewidth=0.5, label='Substellar point',
                               zorder=5)
    plt.gca().add_patch(substellar_circle)

    # temperature maximum
    max_idx = np.unravel_index(np.argmax(atmospherictemperature), atmospherictemperature.shape)
    max_lon = lon_centers[max_idx[1]]  # longitude of max temperature
    max_lat = lat_centers[max_idx[0]]  # latitude of max temperature
    x_max, y_max = m(max_lon, max_lat)
    max_circle = Circle((x_max, y_max), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                        facecolor='red', edgecolor='yellow', linewidth=0.5, label=r'$T_{\rm max}$', zorder=5)
    plt.gca().add_patch(max_circle)

    plt.annotate(
        "West", xy=(0.25, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
        xytext=(0.35, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
        ha='left', va='center', fontsize=12, color='turquoise', fontfamily='sans-serif', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='turquoise', lw=1, shrinkA=0, shrinkB=0)
    )

    plt.annotate(
        "East", xy=(0.75, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
        xytext=(0.6, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
        ha='left', va='center', fontsize=12, color='turquoise', fontfamily='sans-serif', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='turquoise', lw=1, shrinkA=0, shrinkB=0)
    )

    plt.legend(handles=[substellar_circle, max_circle], loc='upper right',
               bbox_to_anchor=(1.05, 1.25), fontsize=10, handler_map={Circle: HandlerCircle()})

    # plt.savefig('/System/Volumes/Data/user/adyrek/non-transiting/'
    # 'results_grid/model_grid/phase_offset/emission_global_map.png', format='png', dpi=300)
    plt.show()


def make_thermal_phase_array(longitudearray, redistribution):
    numberoflongitude = len(longitudearray)

    xi = np.linspace(-np.pi / 2, 3 * np.pi / 2, numberoflongitude)

    # Compute dimensionless thermal phase
    P = compute_thermal_phase(xi, epsilon=redistribution, checking=False)

    # sorting xi and P values
    xi_phase_0_2pi = (xi) % (2 * np.pi)  # - np.pi /2 + 2 * np.pi
    sort_idx = np.argsort(xi_phase_0_2pi)
    xi_sorted = xi_phase_0_2pi[sort_idx]
    P_sorted = P[sort_idx]

    # interpolation
    lon_0_2pi = (longitudearray) % (2 * np.pi)  # + 2 * np.pi
    P_lon = np.interp(lon_0_2pi, xi_sorted, P_sorted)

    return P, P_lon

def plot_thermal_phase_array(longitudearray, redistribution):
    numberoflongitude = len(longitudearray)
    xi = np.linspace(-np.pi / 2, 3 * np.pi / 2, numberoflongitude)

    P, P_lon = make_thermal_phase_array(longitudearray, redistribution)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
    ax.plot(xi * 180 / np.pi, P, color='orange', linewidth=2)
    ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Longitude $\mathcal{\xi}$ [$^{\circ}$]')
    ax.set_ylabel(r"$\mathcal{P}$ (before interp.)")
    ax.set_xlim(-np.pi / 2 * 180 / np.pi, 3 * np.pi / 2 * 180 / np.pi)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
    ax.plot(longitudearray * 180 / np.pi, P_lon, color='orange', linewidth=2)
    ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Longitude $\mathcal{\Phi}$ [$^{\circ}$]')
    ax.set_ylabel(r'$\mathcal{P}$')
    ax.set_xlim(-180, 180)
    plt.tight_layout()
    plt.show()


def plot_contrast(phase, contrast_ppm_refl, contrast_ppm, contrast_ppm_em):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
    ax.plot(phase * 180 / np.pi,
            contrast_ppm_refl, color='lightblue', linewidth=2, label='Reflection')
    ax.plot(phase * 180 / np.pi,
            contrast_ppm, color='pink', linewidth=2, label='Composite')
    ax.plot(phase * 180 / np.pi,
            contrast_ppm_em, color='green', linewidth=2, label='Emission')
    ax.axvline(180, color='lightgrey', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Phase [$^{\circ}$]')
    ax.set_ylabel(r'Planet-star contrast $F_p / F_{\star} \times 10^6$ [ppm]')
    ax.set_xlim(0, 360)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_temperature(longitudearray, latitudearray, internaltemperature, redistribution,
                                      effectivetemperature,
                                      stellarradius,
                                      semimajoraxis,
                                      albedo,
                                      cloud_offset,
                                      albedo_min):

    numberoflongitude = len(longitudearray)
    numberoflatitude = len(latitudearray)
    atmospherictemperature = np.zeros((numberoflatitude, numberoflongitude))

    # Internal flux
    if internaltemperature is None:
        internaltemperature = 0
    else:
        pass

    # albedo
    A_lon = albedo_map(longitudearray, albedo, albedo_min, cloud_offset)
    P, P_lon = make_thermal_phase_array(longitudearray, redistribution)

    # getting the cosine of latitudes
    coslat_1D = np.cos(latitudearray[:])

    # convert to temperature
    T_0 = (effectivetemperature * np.sqrt(stellarradius / semimajoraxis) *
           (1 - A_lon[A_lon.shape[0]//2]) ** 0.25)

    # reference temperature at substellar point
    for ilon in range(numberoflongitude):
        atmospherictemperature[:, ilon] = ((P_lon[ilon] * T_0  * coslat_1D ** 0.25) ** 4
                                                + internaltemperature ** 4) ** 0.25


    return atmospherictemperature

def compute_intensities(response_nu, response_vals, wavearray, atmospherictemperature, effectivetemperature,
                        stellarradius, semimajoraxis, longitudearray, latitudearray, area, planetaryradius,
                        albedo, albedo_min, cloud_offset, checking=False, reflection_mode="global"):
    """
       ``reflection_mode`` can be either ``"global"`` or ``"grid``"
    """

    numberoflongitude = len(longitudearray)
    numberoflatitude = len(latitudearray)
    coszen = np.zeros((numberoflatitude, numberoflongitude))
    reflectedplanetartintensity = np.zeros((numberoflatitude, numberoflongitude))
    Iemis = np.zeros((numberoflatitude, numberoflongitude))

    # Interpolate response onto same nu-edge grid (self.wavearray is cm^-1)
    if (response_nu is not None) and (response_vals is not None):
        resp_at_edges = np.interp(wavearray, response_nu, response_vals, left=0.0, right=0.0)
    else:
        resp_at_edges = np.ones_like(wavearray)  # flat (square) response = 1 across band


    for ilat in range(numberoflatitude):
        coszen[ilat, :] = np.cos(latitudearray[ilat]) * np.cos(longitudearray[:])

    coszen[coszen < 0.] = 0.


    # Planetary Emission
    for ilat in range(numberoflatitude):
        for ilon in range(numberoflongitude):
            B_lambda = compute_Planck_law(nu_edges=wavearray,
                                                        temperature=atmospherictemperature[ilat, ilon],
                                                        numberofterms=50) # W/m2/band/str
            Iemis[ilat, ilon] = integrate_with_response_function(wavearray, resp_at_edges,
                                                                       B_lambda)

    # Planetary reflexion
    # Integrated stellar flux in Kepler bandpass
    Bnu_integrated_star = compute_Planck_law(nu_edges=wavearray, temperature=effectivetemperature,
                                         numberofterms=60)  # W/m2/band/str

    tau_bin = 0.5 * (resp_at_edges[:-1] + resp_at_edges[1:])
    Istar_per_bin = Bnu_integrated_star * tau_bin
    Istar_band = np.sum(Istar_per_bin)
    ISS_band = Istar_band * np.pi * (stellarradius / semimajoraxis) ** 2  # added a missing pi

    # albedo
    A_lon = albedo_map(longitudearray, albedo, albedo_min, cloud_offset)

    if checking:
        plot_albedo_map(longitudearray, A_lon)

    A_grid = np.tile(A_lon, (len(latitudearray), 1))
    if reflection_mode=="grid":
        reflectedplanetartintensity = coszen * A_grid * ISS_band
    elif reflection_mode=="global":
        reflectedplanetartintensity = np.tile(albedo * np.maximum(0, np.cos (longitudearray
                                                                                       + np.radians(cloud_offset)))
                                                   * ISS_band, (len(latitudearray), 1))


    emittedplanetaryintensity = Iemis[:, :]

    # Total intensity
    totalplanetartintensity = reflectedplanetartintensity + emittedplanetaryintensity

    if checking:
        plot_reflected_radiance(longitudearray, latitudearray, reflectedplanetartintensity, reflection_mode)
        check_intensities_units(totalplanetartintensity, area, planetaryradius)


    return reflectedplanetartintensity, emittedplanetaryintensity, totalplanetartintensity

def compute_phasecurve(longitudearray, latitudearray, phase, reflectedplanetartintensity, totalplanetartintensity,
                       area, emittedplanetaryintensity, inclination, checking=False):

    nlon = len(longitudearray)
    nlat = len(latitudearray)

    observedlatitude = 90. - inclination
    observedlatitude = convert_deg_to_radian(observedlatitude)

    phase = convert_deg_to_radian(phase)
    nphase = len(phase)

    fobs = np.zeros(nphase)
    fobs_refl = np.zeros(nphase)
    fobs_em = np.zeros(nphase)

    xyz_obs = np.zeros((3, nphase))
    xyz = np.zeros((3, nlat, nlon))

    for ilat in range(nlat):
        xyz[0, ilat, :] = np.cos(latitudearray[ilat]) * np.cos(longitudearray[:])
        xyz[1, ilat, :] = np.cos(latitudearray[ilat]) * np.sin(longitudearray[:])
        xyz[2, ilat, :] = np.sin(latitudearray[ilat])

    xyz_obs[0, :] = np.cos(observedlatitude) * np.cos(- phase[:] + np.pi)
    xyz_obs[1, :] = np.cos(observedlatitude) * np.sin(- phase[:] + np.pi)
    xyz_obs[2, :] = np.sin(observedlatitude)

    cos_theta = np.zeros((nlat, nlon))

    for iphase in range(nphase):
        for ilat in range(nlat):
            cos_theta[ilat, :] = xyz[0, ilat, :] * xyz_obs[0, iphase] + \
                                 xyz[1, ilat, :] * xyz_obs[1, iphase] + xyz[2, ilat, :] * xyz_obs[2, iphase]

        visible = np.where(cos_theta > 0.)

        fobs[iphase] = np.sum(totalplanetartintensity[visible] * area[visible] * cos_theta[visible])
        fobs_refl[iphase] = np.sum(reflectedplanetartintensity[visible] * area[visible]
                                   * cos_theta[visible])
        fobs_em[iphase] = np.sum(emittedplanetaryintensity[visible] * area[visible] * cos_theta[visible])

    if checking:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
        ax.plot(phase * 180 / np.pi, fobs_em, color='orange',
                linewidth=2, label=r'Emitted $\mathcal{I}_{\rm R, em}$')
        ax.plot(phase * 180 / np.pi, fobs_refl, color='red',
                linewidth=2, label=r'Reflected $\mathcal{I}_{\rm R, refl}$')
        ax.axvline(180, color='lightgrey', linestyle='--', linewidth=1)
        ax.set_xlabel(r'Phase [$^{\circ}$]')
        ax.set_ylabel(r'Irradiance $\mathcal{I}_{\rm R}$ [W m$^2$]')
        ax.set_xlim(0, 360)
        plt.legend()
        plt.tight_layout()
        plt.show()


    return fobs, fobs_refl, fobs_em

def compute_contrast(phase, wavearray, effectivetemperature, response_nu, response_vals, planetaryradius,
                     stellarradius, longitudearray, latitudearray, internaltemperature, redistribution, semimajoraxis,
                     albedo, cloud_offset, area, inclination, albedo_min, reflection_mode, checking=False):

    atmospherictemperature = compute_temperature(longitudearray, latitudearray, internaltemperature, redistribution,
                                      effectivetemperature,
                                      stellarradius,
                                      semimajoraxis,
                                      albedo,
                                      cloud_offset,
                                      albedo_min)

    reflectedplanetartintensity, emittedplanetaryintensity, totalplanetartintensity = (
        compute_intensities(response_nu, response_vals, wavearray, atmospherictemperature, effectivetemperature,
                        stellarradius, semimajoraxis, longitudearray, latitudearray, area, planetaryradius,
                        albedo, albedo_min, cloud_offset, checking, reflection_mode))


    fobs, fobs_refl, fem = compute_phasecurve(longitudearray, latitudearray, phase,
                                              reflectedplanetartintensity, totalplanetartintensity,
                       area, emittedplanetaryintensity, inclination, checking=checking)

    phase = convert_deg_to_radian(phase)

    # Integrated stellar flux in Kepler bandpass
    Bnu_integrated_star = compute_Planck_law(nu_edges=wavearray, temperature=effectivetemperature,
                                         numberofterms=60)  # W/m2/band/str

    # Interpolate response onto same nu-edge grid (self.wavearray is cm^-1)
    if (response_nu is not None) and (response_vals is not None):
        resp_at_edges = np.interp(wavearray, response_nu, response_vals, left=0.0, right=0.0)
    else:
        resp_at_edges = np.ones_like(wavearray)  # flat (square) response = 1 across band

    tau_bin = 0.5 * (resp_at_edges[:-1] + resp_at_edges[1:])
    Istar_per_bin = Bnu_integrated_star * tau_bin
    Istar_band = np.sum(Istar_per_bin)

    Fstar_band = Istar_band * np.pi


    # Total contrast reflected + emitted
    contrast_ppm = (fobs * 1.e6 / Fstar_band) * \
                                        (planetaryradius / stellarradius) ** 2

    contrast_ppm_em = (fem * 1.e6 / Fstar_band) * (planetaryradius / stellarradius) ** 2
    # Only reflected contrast
    contrast_ppm_refl = (fobs_refl * 1.e6 / Fstar_band) * (planetaryradius / stellarradius) ** 2

    if checking:
        plot_contrast(phase, contrast_ppm_refl, contrast_ppm, contrast_ppm_em)

    return contrast_ppm

def compute_flux(phase, wavearray, effectivetemperature, semimajoraxis, mission, planetaryradius,
                     stellarradius, internaltemperature, redistribution,
                     albedo, cloud_offset, inclination, albedo_min, reflection_mode, checking=False):
    wavearray = convert_mu_to_cm(wavearray)

    response_nu, response_vals = read_response_function(mission)

    longitudearray, latitudearray = compute_longitude_latitude()
    area = check_area(latitudearray, longitudearray, checking=checking)

    contrast = compute_contrast(phase, wavearray, effectivetemperature, response_nu, response_vals, planetaryradius,
                     stellarradius, longitudearray, latitudearray, internaltemperature, redistribution, semimajoraxis,
                     albedo, cloud_offset, area, inclination, albedo_min, reflection_mode, checking=checking)

    return contrast

if test:
    #  Test
    start = time.time()

    def run_model():
        #  inputs
        planetaryradius = 0.22  # Jup radius
        albedo = 0.7
        albedo_min = 0.7
        redistribution = 0.05

        targetname = '9139163'
        period = 0.604734  # in days
        nphase = 100  # discretization of phases
        phase_model = np.linspace(0, 360., nphase)
        planetarymasssini = 11.4  # Earth mass, from RV fit
        planetarymasssini = planetarymasssini * 0.00314558  # conversion into Jupiter mass
        inclination = 62  # degrees
        effectivetemperature = 6358
        stellarmass = 1.390
        stellarradius = 1.558
        cloud_offset = 120
        internaltemperature=100
        mission='TESS'
        if mission == 'Kepler':
            wavearray = np.array([0.430, 0.890])  # Kepler bandpass in micron
        if mission == 'TESS':
            wavearray = np.array([0.6, 1])  # TESS bandpass in micron
        reflection_mode = "global"
        checking = True


        if reflection_mode in ["global", "grid"]:
            pass
        else:
            print(f'Reflection mode should be "global" or "grid".')
            sys.exit()

        if reflection_mode == 'grid':
            if albedo_min >= albedo:
                print('For the reflection "map" to work, "albedo_min" and "albedo" should be different, '
                      'and "albedo_min" < "albedo".')
                sys.exit()

        stellarmass, stellarradius, planetaryradius, orbitalperiod = convert_orbital_parameters(stellarmass,
                                                                                                stellarradius,
                                                                                                planetaryradius,
                                                                                                period)
        semimajoraxis = compute_semi_major_axis(orbitalperiod, stellarmass, checking=checking)


        contrast = compute_flux(phase=phase_model, wavearray=wavearray, effectivetemperature=effectivetemperature,
                     semimajoraxis=semimajoraxis, mission=mission,
                     planetaryradius=planetaryradius,
                    stellarradius=stellarradius, internaltemperature=internaltemperature,
                     redistribution=redistribution, albedo=albedo, cloud_offset=cloud_offset,
                     inclination=inclination, albedo_min=albedo_min, reflection_mode=reflection_mode, checking=checking)



    run_model()
    end = time.time()
    elapsed_time = end - start
    print(f'Model takes {elapsed_time} seconds to complete.')




