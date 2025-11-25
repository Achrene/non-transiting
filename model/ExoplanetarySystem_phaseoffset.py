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

colors_matter = cm.cm.matter_r(np.linspace(0, 1, 10))

test = False

ROOT = Path(__file__).resolve().parent
references_path = str(ROOT / "references")


class ExoplanetarySystem_phaseoffset:
    def __init__(self, orbitalperiod,
                 effectivetemperature, stellarmass, stellarradius, semimajoraxis, planetaryradius, planetarymass,
                 inclination, redistribution, albedo, wavearray, longitudearray, latitudearray, checking,
                 internaltemperature, area, phase, atmospherictemperature,
                 totalplanetartintensity, emittedplanetaryintensity, reflectedplanetartintensity,
                 contrast_ppm, contrast_ppm_refl, mission, response_nu, response_vals, cloud_offset, albedo_min):

        self.orbitalperiod = orbitalperiod  # days
        self.effectivetemperature = effectivetemperature  # stellar temperature K
        self.stellarmass = stellarmass  # Solar mass
        self.stellarradius = stellarradius  # Solar mass
        self.semimajoraxis = semimajoraxis  # UA
        self.planetaryradius = planetaryradius  # Jupiter radii
        self.planetarymass = planetarymass  # Jupiter mass sin i
        self.inclination = inclination  # degrees
        self.redistribution = redistribution  # value between 0 and 1, no unit
        self.albedo = albedo  # value between 0 and 1, no unit
        self.wavearray = wavearray  # wave bandpass array micron
        self.longitudearray = longitudearray  # longitude array radian
        self.latitudearray = latitudearray  # latitude array radian
        self.checking = checking  # True or false
        self.internaltemperature = internaltemperature  # K
        self.area = area  # in unit of cell (latitude, longitude) of Jupiter radius
        self.phase = phase  # in degrees
        self.atmospherictemperature = atmospherictemperature
        self.totalplanetartintensity = totalplanetartintensity  # total planetary flux
        self.emittedplanetaryintensity  = emittedplanetaryintensity  # total emitted flux
        self.reflectedplanetartintensity = reflectedplanetartintensity  # total reflected flux
        self.contrast_ppm = contrast_ppm  # normalized total flux in ppm
        self.contrast_ppm_refl = contrast_ppm_refl  # normalized reflected flux in ppm
        self.mission = mission
        self.response_nu = response_nu
        self.response_vals = response_vals
        self.cloud_offset = cloud_offset
        self.albedo_min = albedo_min

    def read_response_function(self):
        if self.mission == 'Kepler':
            response_function_file = os.path.join(references_path, 'Kepler_Response_Function.txt')
            response_fonction_data = np.loadtxt(response_function_file, skiprows = 8)
            response_function_values = response_fonction_data[:, 1]
            response_function_wavelength = response_fonction_data[:, 0] * 10**(-3)  # converted into microns

            response_lambda_cm = response_function_wavelength * 1e-4  # micron -> cm
            response_nu_cm1 = 1.0 / response_lambda_cm  # cm^-1

            sort_idx = np.argsort(response_nu_cm1)
            self.response_nu = response_nu_cm1[sort_idx]
            self.response_vals = response_function_values[sort_idx]

        elif self.mission == 'TESS':
            print('The TESS response function has not been implemented yet. Using a square function instead.')
            self.response_nu = None
            self.response_vals = None

        elif self.mission == 'None':
            print('No mission name provided: response function will be squared.')
            self.response_nu = None
            self.response_vals = None
        else:
            print('Mission name not recognised, response function cannot be processed.')
            sys.exit()

        return None

    def integrate_with_response_function(self, wavelengths, response, B_lambda):
        """Integrate the Planck function weighted by the response function."""
        A = integrate.simpson(y=B_lambda * response, x=wavelengths)
        B = integrate.simpson(y=response, x=wavelengths)
        return A / B

    def convert_mu_to_cm(self):
        self.wavearray = 1 / (self.wavearray * 1e-4)  # from micron to cm-1
        self.wavearray = np.flip(self.wavearray)

    def convert_orbital_parameters(self):
        self.stellarmass = (self.stellarmass * c.M_sun).value  # in kg
        self.stellarradius = (self.stellarradius * c.R_sun).value  # in m
        self.planetaryradius = (self.planetaryradius * c.R_jup).value  # in m
        self.orbitalperiod = self.orbitalperiod * 24 * 3600  # in seconds

    def compute_semi_major_axis(self):
        self.semimajoraxis = ((self.orbitalperiod ** 2 * c.G.value * self.stellarmass) /
                              (4 * (np.pi ** 2))) ** (1 / 3)  # in m

        if self.checking is True:
            print(f'Semi-major axis is: {self.semimajoraxis * 6.68459e-12} UA.')

        return self.semimajoraxis

    def compute_longitude_latitude(self):
        # Sphere discretisation

        numberoflongitude = 180
        numberoflatitude = 90

        lon_deg_limits = np.linspace(-180., 180., numberoflongitude + 1)
        lon_deg = (lon_deg_limits[0:numberoflongitude] + lon_deg_limits[1:numberoflongitude + 1]) * 0.5
        lat_deg_limits = np.linspace(-90., 90., numberoflatitude + 1)
        lat_deg = (lat_deg_limits[0:numberoflatitude] + lat_deg_limits[1:numberoflatitude + 1]) * 0.5

        self.longitudearray = utils.convert_deg_to_radian(lon_deg)  # in radian
        self.latitudearray = utils.convert_deg_to_radian(lat_deg)  # in radian

    def check_area(self):
        self.area = np.zeros((len(self.latitudearray), len(self.longitudearray)))
        dlat = self.latitudearray[1] - self.latitudearray[0];
        dlon = self.longitudearray[1] - self.longitudearray[0]

        self.area[-1, :] = dlon * (1. - np.sin(self.latitudearray[-1] - dlat / 2.))
        self.area[0, :] = self.area[len(self.latitudearray) - 1, :]

        for ilat in range(len(self.latitudearray) - 2):
            self.area[ilat + 1, :] = dlon * (np.sin(self.latitudearray[ilat + 1] + dlat / 2.) -
                                        np.sin(self.latitudearray[ilat + 1] - dlat / 2.))
            # area in planetary radius unit of surface cells
            # depends on latitude only
        if self.checking is True:
            print("Area must be around 1 :", self.area.sum() / (4 * np.pi))

    def compute_Planck_law(self, nu_edges, temperature, numberofterms=30):
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

    def fun(self, xi, P, epsilon=1):
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

    def compute_thermal_phase(self, xi, epsilon=1, verbose=False):
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
        result = solve_ivp(self.fun, xi_span, np.atleast_1d(Pdawn),
                           t_eval=xi, args=(epsilon,))
        if verbose:
            print(result.status)
            print(result.message)
            print(result.success)
        return result.y[0]

    def albedo_map(self, lon_array):
        A_max = self.albedo  # from grid
        A_min = self.albedo_min  # from Webber+, 2015, figure 5
        theta_c = np.radians(self.cloud_offset)  # cloud offset
        sigma = np.radians(40)  # cloud width

        delta = (lon_array + theta_c + np.pi) % (2 * np.pi) - np.pi
        return A_min + (A_max - A_min) * np.exp(-delta ** 2 / (2 * sigma ** 2))

    def compute_temperature_and_intensity(self):

        numberoflongitude = len(self.longitudearray)
        numberoflatitude = len(self.latitudearray)
        coszen = np.zeros((numberoflatitude, numberoflongitude))
        self.reflectedplanetartintensity = np.zeros((numberoflatitude, numberoflongitude))
        Iemis = np.zeros((numberoflatitude, numberoflongitude))
        self.totalplanetartintensity = np.zeros((numberoflatitude, numberoflongitude))
        self.atmospherictemperature = np.zeros((numberoflatitude, numberoflongitude))

        for ilat in range(numberoflatitude):
            coszen[ilat, :] = np.cos(self.latitudearray[ilat]) * np.cos(self.longitudearray[:])

        coszen[coszen < 0.] = 0.

        # albedo
        A_lon = self.albedo_map(self.longitudearray)

        # Internal flux
        if self.internaltemperature is None:
            self.internaltemperature = 0
        else:
            pass

        xi = np.linspace(-np.pi/2, 3*np.pi/2, numberoflongitude)

        # Compute dimensionless thermal phase
        P = self.compute_thermal_phase(xi, epsilon=self.redistribution, verbose=False)

        if test:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
            ax.plot(xi * 180 / np.pi, P, color='orange', linewidth=2)
            ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1)
            ax.set_xlabel(r'Longitude $\mathcal{\xi}$ [$^{\circ}$]')
            ax.set_ylabel(r"$\mathcal{P}$ (before interp.)")
            ax.set_xlim(-np.pi / 2 * 180 / np.pi, 3 * np.pi / 2 * 180 / np.pi)
            plt.tight_layout()
            plt.show()

        # sorting xi and P values
        xi_phase_0_2pi = (xi) % (2 * np.pi) # - np.pi /2 + 2 * np.pi
        sort_idx = np.argsort(xi_phase_0_2pi)
        xi_sorted = xi_phase_0_2pi[sort_idx]
        P_sorted = P[sort_idx]

        # interpolation
        lon_0_2pi = (self.longitudearray) % (2 * np.pi) # + 2 * np.pi
        P_lon = np.interp(lon_0_2pi, xi_sorted, P_sorted)


        if test:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
            ax.plot(self.longitudearray * 180 / np.pi, P_lon, color='orange', linewidth=2)
            ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1)
            ax.set_xlabel(r'Longitude $\mathcal{\Phi}$ [$^{\circ}$]')
            ax.set_ylabel(r'$\mathcal{P}$')
            ax.set_xlim(-180, 180)
            plt.tight_layout()
            plt.show()

        # getting the cosine of latitudes
        coslat_1D = np.cos(self.latitudearray[:])
        # convert to temperature
        T_0 = (self.effectivetemperature * np.sqrt(self.stellarradius / self.semimajoraxis) *
               (1 - A_lon[A_lon.shape[0]//2]) ** 0.25)
        # reference temperature at substellar point
        for ilon in range(numberoflongitude):
            self.atmospherictemperature[:, ilon] = ((P_lon[ilon] * T_0  * coslat_1D ** 0.25) ** 4
                                                    + self.internaltemperature ** 4) ** 0.25

        if test:
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
            lon_centers = self.longitudearray * 180 / np.pi
            lat_centers = self.latitudearray * 180 / np.pi

            # Compute edges
            lon_edges = utils.compute_edges(lon_centers)
            lat_edges = utils.compute_edges(lat_centers)

            # Create meshgrid of edges
            lons, lats = np.meshgrid(lon_edges, lat_edges)

            im1 = m.pcolormesh(lons, lats, self.atmospherictemperature, cmap=cm.cm.matter_r, latlon=True, shading='auto')
            cb = m.colorbar(im1, "bottom", size="5%", pad="2%")
            cb.set_label(r'$T_{\mathrm{atm}}$ [K]')

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
                                       facecolor='cyan', edgecolor='black', linewidth=0.5, label='Substellar point', zorder=5)
            plt.gca().add_patch(substellar_circle)

            # temperature maximum
            max_idx = np.unravel_index(np.argmax(self.atmospherictemperature), self.atmospherictemperature.shape)
            max_lon = lon_centers[max_idx[1]]  # longitude of max temperature
            max_lat = lat_centers[max_idx[0]]  # latitude of max temperature
            x_max, y_max = m(max_lon, max_lat)
            max_circle = Circle((x_max, y_max), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                                facecolor='red', edgecolor='black', linewidth=0.5, label=r'$T_{\rm max}$', zorder=5)
            plt.gca().add_patch(max_circle)

            plt.annotate(
                "West", xy=(0.25, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
                xytext=(0.3, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
                ha='left', va='center', fontsize=9, color='white', fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='white', lw=1, shrinkA=0, shrinkB=0)
            )

            plt.annotate(
                "East", xy=(0.75, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
                xytext=(0.65, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
                ha='left', va='center', fontsize=9, color='white', fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='white', lw=1, shrinkA=0, shrinkB=0)
            )


            plt.legend(handles=[substellar_circle, max_circle], loc='upper right',
                       bbox_to_anchor=(1.03, 1.1), fontsize=7, handler_map={Circle: HandlerCircle()})

            plt.show()

        # get response function
        self.read_response_function()
        # Interpolate response onto same nu-edge grid (self.wavearray is cm^-1)
        if (self.response_nu is not None) and (self.response_vals is not None):
            resp_at_edges = np.interp(self.wavearray, self.response_nu, self.response_vals, left=0.0, right=0.0)
        else:
            resp_at_edges = np.ones_like(self.wavearray)  # flat (square) response = 1 across band

        # Planetary Emission
        for ilat in range(numberoflatitude):
            for ilon in range(numberoflongitude):
                B_lambda = self.compute_Planck_law(nu_edges=self.wavearray,
                                                            temperature=self.atmospherictemperature[ilat, ilon],
                                                            numberofterms=50) # W/m2/band/str
                Iemis[ilat, ilon] = self.integrate_with_response_function(self.wavearray, resp_at_edges,
                                                                           B_lambda)

        # Planetary reflexion
        # Integrated stellar flux in Kepler bandpass
        Bnu_integrated_star = self.compute_Planck_law(nu_edges=self.wavearray, temperature=self.effectivetemperature,
                                             numberofterms=60)  # W/m2/band/str

        tau_bin = 0.5 * (resp_at_edges[:-1] + resp_at_edges[1:])
        Istar_per_bin = Bnu_integrated_star * tau_bin
        Istar_band = np.sum(Istar_per_bin)
        ISS_band = Istar_band * np.pi * (self.stellarradius / self.semimajoraxis) ** 2  # added a missing pi

        if test:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
            ax.plot(self.longitudearray * 180/np.pi, A_lon, color='green', linewidth=2)
            ax.axhline(A_lon[A_lon.shape[0]//2], color='lightpink', linestyle='--', linewidth=1)
            ax.axvline(0, color='lightgrey', linestyle='--', linewidth=1)
            ax.set_xlabel(r'Longitude [$^{\circ}$]')
            ax.set_ylabel('Albedo $A_B$')
            ax.set_xlim(-180, 180)
            plt.tight_layout()
            plt.show()

        A_grid = np.tile(A_lon, (len(self.latitudearray), 1))
        self.reflectedplanetartintensity[:, :] = coszen[:, :] * A_grid * ISS_band

        self.emittedplanetaryintensity = Iemis[:, :]

        # Total intensity
        self.totalplanetartintensity[:, :] = self.reflectedplanetartintensity[:, :] + self.emittedplanetaryintensity

        if test:
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
            lon_centers = self.longitudearray * 180 / np.pi
            lat_centers = self.latitudearray * 180 / np.pi

            # Compute edges
            lon_edges = utils.compute_edges(lon_centers)
            lat_edges = utils.compute_edges(lat_centers)

            # Create meshgrid of edges
            lons, lats = np.meshgrid(lon_edges, lat_edges)

            im1 = m.pcolormesh(lons, lats, self.reflectedplanetartintensity, cmap=cm.cm.matter_r, latlon=True, shading='auto')
            cb = m.colorbar(im1, "bottom", size="5%", pad="2%")
            cb.set_label(r'Reflected radiance $\mathcal{I}_{\rm refl}$ [W m$^2$ sr$^{-1}$]')

            lon_labels = [lon_edges[0], lon_edges[len(lon_edges) // 2], lon_edges[-1]]
            for lon in lon_labels:
                x, y = m(lon, lat_edges[-1])  # top edge
                plt.text(x, y + 0.02 * (m.urcrnrx - m.llcrnrx), f"{lon:.0f}°",
                         ha='center', va='bottom', fontsize=10)

            plt.text(0.5, 1.15, "Longitude", transform=plt.gca().transAxes,
                     ha='center', va='bottom', fontsize=12)

            plt.annotate(
                "West", xy=(0.25, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
                xytext=(0.3, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
                ha='left', va='center', fontsize=9, color='white', fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='white', lw=1, shrinkA=0, shrinkB=0)
            )

            plt.annotate(
                "East", xy=(0.75, 0.9), xycoords='axes fraction',  # position (10% from left, just above plot)
                xytext=(0.65, 0.9), textcoords='axes fraction',  # text slightly to the right of arrow
                ha='left', va='center', fontsize=9, color='white', fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='white', lw=1, shrinkA=0, shrinkB=0)
            )

            # substellar point
            x0, y0 = m(0, 0)
            substellar_circle = Circle((x0, y0), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                                       facecolor='cyan', edgecolor='black', linewidth=0.5, label='Substellar point',
                                       zorder=5)
            plt.gca().add_patch(substellar_circle)

            # temperature maximum
            max_idx = np.unravel_index(np.argmax(self.reflectedplanetartintensity), self.reflectedplanetartintensity.shape)
            max_lon = lon_centers[max_idx[1]]
            max_lat = lat_centers[max_idx[0]]
            x_max, y_max = m(max_lon, max_lat)
            max_circle = Circle((x_max, y_max), radius=0.01 * (m.urcrnrx - m.llcrnrx),
                                facecolor='red', edgecolor='black', linewidth=0.5, label=r'$\mathcal{I}_{\rm{refl}, \rm{max}}$', zorder=5)
            plt.gca().add_patch(max_circle)

            plt.legend(handles=[substellar_circle, max_circle], loc='upper right',
                       bbox_to_anchor=(1.03, 1.1), fontsize=7, handler_map={Circle: HandlerCircle()})

            plt.show()

        if self.checking is True:
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
            total_rad = np.sum(self.totalplanetartintensity[:, :] * self.area[:, :]) * (self.planetaryradius ** 2)
            print(f'Total power: {total_rad} W.')

    def compute_phasecurve(self):

        nlon = len(self.longitudearray)
        nlat = len(self.latitudearray)

        observedlatitude = 90. - self.inclination
        observedlatitude = utils.convert_deg_to_radian(observedlatitude)

        self.phase = utils.convert_deg_to_radian(self.phase)
        nphase = len(self.phase)

        fobs = np.zeros(nphase)
        fobs_refl = np.zeros(nphase)
        fobs_em = np.zeros(nphase)

        xyz_obs = np.zeros((3, nphase))
        xyz = np.zeros((3, nlat, nlon))

        for ilat in range(nlat):
            xyz[0, ilat, :] = np.cos(self.latitudearray[ilat]) * np.cos(self.longitudearray[:])
            xyz[1, ilat, :] = np.cos(self.latitudearray[ilat]) * np.sin(self.longitudearray[:])
            xyz[2, ilat, :] = np.sin(self.latitudearray[ilat])

        xyz_obs[0, :] = np.cos(observedlatitude) * np.cos(- self.phase[:] + np.pi)
        xyz_obs[1, :] = np.cos(observedlatitude) * np.sin(- self.phase[:] + np.pi)
        xyz_obs[2, :] = np.sin(observedlatitude)

        cos_theta = np.zeros((nlat, nlon))

        for iphase in range(nphase):
            for ilat in range(nlat):
                cos_theta[ilat, :] = xyz[0, ilat, :] * xyz_obs[0, iphase] + \
                                     xyz[1, ilat, :] * xyz_obs[1, iphase] + xyz[2, ilat, :] * xyz_obs[2, iphase]

            visible = np.where(cos_theta > 0.)

            fobs[iphase] = np.sum(self.totalplanetartintensity[visible] * self.area[visible] * cos_theta[visible])
            fobs_refl[iphase] = np.sum(self.reflectedplanetartintensity[visible] * self.area[visible]
                                       * cos_theta[visible])
            fobs_em[iphase] = np.sum(self.emittedplanetaryintensity[visible] * self.area[visible] * cos_theta[visible])

        if test:

            fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
            ax.plot(self.phase * 180 / np.pi, fobs_em, color='orange',
                    linewidth=2, label=r'Emitted $\mathcal{I}_{\rm R, em}$')
            ax.plot(self.phase * 180 / np.pi, fobs_refl, color='red',
                    linewidth=2, label=r'Reflected $\mathcal{I}_{\rm R, refl}$')
            ax.axvline(180, color='lightgrey', linestyle='--', linewidth=1)
            ax.set_xlabel(r'Phase [$^{\circ}$]')
            ax.set_ylabel(r'Irradiance $\mathcal{I}_{\rm R}$ [W m$^2$]')
            ax.set_xlim(0, 360)
            plt.legend()
            plt.tight_layout()
            plt.show()


        return fobs, fobs_refl, fobs_em

    def compute_contrast(self):
        # Reflected and emitted light computation
        nphase = len(self.phase)
        self.contrast_ppm = np.zeros(nphase)
        self.contrast_ppm_refl = np.zeros(nphase)
        contrast_ppm_em = np.zeros(nphase)

        self.compute_temperature_and_intensity()

        fobs, fobs_refl, fem = self.compute_phasecurve()

        self.read_response_function()

        # Integrated stellar flux in Kepler bandpass
        Bnu_integrated_star = self.compute_Planck_law(nu_edges=self.wavearray, temperature=self.effectivetemperature,
                                             numberofterms=60)  # W/m2/band/str

        # Interpolate response onto same nu-edge grid (self.wavearray is cm^-1)
        if (self.response_nu is not None) and (self.response_vals is not None):
            resp_at_edges = np.interp(self.wavearray, self.response_nu, self.response_vals, left=0.0, right=0.0)
        else:
            resp_at_edges = np.ones_like(self.wavearray)  # flat (square) response = 1 across band

        tau_bin = 0.5 * (resp_at_edges[:-1] + resp_at_edges[1:])
        Istar_per_bin = Bnu_integrated_star * tau_bin
        Istar_band = np.sum(Istar_per_bin)

        Fstar_band = Istar_band * np.pi

        # Total contrast reflected + emitted
        self.contrast_ppm = (fobs * 1.e6 / Fstar_band) * \
                                            (self.planetaryradius / self.stellarradius) ** 2

        contrast_ppm_em = (fem * 1.e6 / Fstar_band) * (self.planetaryradius / self.stellarradius) ** 2
        # Only reflected contrast
        self.contrast_ppm_refl = (fobs_refl * 1.e6 / Fstar_band) * (self.planetaryradius / self.stellarradius) ** 2

        if test:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)
            ax.plot(self.phase * 180 / np.pi,
                    self.contrast_ppm_refl, color='lightblue', linewidth=2, label='Reflection')
            ax.plot(self.phase * 180 / np.pi,
                    self.contrast_ppm, color='pink', linewidth=2, label='Composite')
            ax.plot(self.phase * 180 / np.pi,
                    contrast_ppm_em, color='green', linewidth=2, label='Emission')
            ax.axvline(180, color='lightgrey', linestyle='--', linewidth=1)
            ax.set_xlabel(r'Phase [$^{\circ}$]')
            ax.set_ylabel(r'Planet-star contrast $F_p / F_{\star} \times 10^6$ [ppm]')
            ax.set_xlim(0, 360)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return self.contrast_ppm

    def compute_flux(self):
        self.convert_mu_to_cm()
        self.convert_orbital_parameters()
        self.compute_semi_major_axis()
        self.compute_longitude_latitude()
        self.check_area()

        contrast = self.compute_contrast()

        # plots.plot_temperature_map(self)
        # contrast_for_plotting = contrast[np.newaxis, np.newaxis, np.newaxis, :]
        # plots.plot_photometry_model(self, contrast_for_plotting, [self.planetaryradius],
                                    # [self.albedo], [self.redistribution])
        return contrast


if test:
    #  Test
    def run_model():
        #  inputs
        planetaryradius = 0.3  # Jup radius
        albedo = 0.5
        albedo_min = 0.5
        redistribution = 0.95

        targetname = '9139163'
        period = 0.604734  # in days
        nphase = 100  # discretization of phases
        phase_model = np.linspace(0, 360., nphase)
        wavearray = np.array([0.430, 0.890])  # Kepler bandpass in micron
        planetarymasssini = 12.5  # Earth mass, from RV fit
        planetarymasssini = planetarymasssini * 0.00314558  # conversion into Jupiter mass
        inclination = 62  # degrees
        effectivetemperature = 6358
        stellarmass = 1.390
        stellarradius = 1.558
        cloud_offset = 80

        punto_system = ExoplanetarySystem_phaseoffset(orbitalperiod=period,
                                                             effectivetemperature=effectivetemperature,
                                                             stellarmass=stellarmass,
                                                             stellarradius=stellarradius,
                                                             semimajoraxis=None, planetaryradius=planetaryradius,
                                                             planetarymass=planetarymasssini, inclination=inclination,
                                                             redistribution=np.round(redistribution, 1),
                                                             albedo=np.round(albedo, 1),
                                                             wavearray=wavearray,
                                                             longitudearray=None, latitudearray=None, checking=True,
                                                             internaltemperature=100, area=None, phase=phase_model,
                                                             atmospherictemperature=None,
                                                             totalplanetartintensity=None,
                                                             emittedplanetaryintensity=None,
                                                             reflectedplanetartintensity=None,
                                                             contrast_ppm=None, contrast_ppm_refl=None,
                                                             mission='Kepler', response_nu=None, response_vals=None,
                                                             cloud_offset = cloud_offset,
                                                             albedo_min = albedo_min)

        return punto_system.compute_flux()

    run_model()

