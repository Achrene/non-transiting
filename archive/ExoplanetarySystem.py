import numpy as np
import pandas as pd
import os
import astropy.constants as c
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import plots
import utils
import sys

test = False


class ExoplanetarySystem:
    def __init__(self, orbitalperiod,
                 effectivetemperature, stellarmass, stellarradius, semimajoraxis, planetaryradius, planetarymass,
                 inclination, redistribution, albedo, wavearray, longitudearray, latitudearray, checking,
                 internaltemperature, area, phase, atmospherictemperature,
                 totalplanetartintensity, emittedplanetaryintensity, reflectedplanetartintensity,
                 contrast_ppm, contrast_ppm_refl, mission, response_nu, response_vals):

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

    def read_response_function(self):
        if self.mission == 'Kepler':
            response_function_file = '/references/Kepler_Response_Function.txt'
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

    def compute_temperature_and_intensity(self):

        numberoflongitude = len(self.longitudearray)
        numberoflatitude = len(self.latitudearray)
        coszen = np.zeros((numberoflatitude, numberoflongitude))
        self.reflectedplanetartintensity = np.zeros((numberoflatitude, numberoflongitude))
        Iemis = np.zeros((numberoflatitude, numberoflongitude))
        self.totalplanetartintensity = np.zeros((numberoflatitude, numberoflongitude))
        self.atmospherictemperature = np.zeros((numberoflatitude, numberoflongitude))

        for ilat in range(numberoflatitude):
            coszen[ilat, :] = np.cos(self.longitudearray[:]) * np.cos(self.latitudearray[ilat])

        coszen[coszen < 0.] = 0.

        # Bolometric flux at effective temperature
        FSS_bolo = (c.sigma_sb.value * self.effectivetemperature ** 4) * (self.stellarradius / self.semimajoraxis) ** 2
        # W / m2

        stellar_unif = (1/4) * FSS_bolo * self.redistribution
        stellar_local = FSS_bolo * (1. - self.redistribution)

        # Internal flux
        if self.internaltemperature is None:
            self.internaltemperature = 0
        else:
            pass

        Fint = c.sigma_sb.value * self.internaltemperature ** 4  # internal heat flux [W/m2]

        self.atmospherictemperature[:, :] = ((stellar_unif * (1. - self.albedo) + stellar_local * (1. - self.albedo)
                        * coszen[:, :] + Fint) / c.sigma_sb.value) ** 0.25

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

        self.reflectedplanetartintensity[:, :] = coszen[:, :] * self.albedo * ISS_band

        self.emittedplanetaryintensity = Iemis[:, :]

        # Total intensity
        self.totalplanetartintensity[:, :] = self.reflectedplanetartintensity[:, :] + self.emittedplanetaryintensity

        if self.checking is True:
            # Total irradiance
            total_rad = np.sum(self.totalplanetartintensity[:, :] * self.area[:, :]) * (self.planetaryradius ** 2)
            print(f'Total Radiation: {total_rad} W.')

    def compute_phasecurve(self):

        nlon = len(self.longitudearray)
        nlat = len(self.latitudearray)

        observedlatitude = 90. - self.inclination
        observedlatitude = utils.convert_deg_to_radian(observedlatitude)

        self.phase = utils.convert_deg_to_radian(self.phase)
        nphase = len(self.phase)

        fobs = np.zeros(nphase)
        fobs_refl = np.zeros(nphase)

        xyz_obs = np.zeros((3, nphase))
        xyz = np.zeros((3, nlat, nlon))

        for ilat in range(nlat):
            xyz[0, ilat, :] = np.cos(self.latitudearray[ilat]) * np.cos(self.longitudearray[:])
            xyz[1, ilat, :] = np.cos(self.latitudearray[ilat]) * np.sin(self.longitudearray[:])
            xyz[2, ilat, :] = np.sin(self.latitudearray[ilat])

        xyz_obs[0, :] = np.cos(observedlatitude) * np.cos(self.phase[:] + np.pi)
        xyz_obs[1, :] = np.cos(observedlatitude) * np.sin(self.phase[:] + np.pi)
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

        return fobs, fobs_refl

    def compute_contrast(self):
        # Reflected and emitted light computation
        nphase = len(self.phase)
        self.contrast_ppm = np.zeros(nphase)
        self.contrast_ppm_refl = np.zeros(nphase)

        self.compute_temperature_and_intensity()

        fobs, fobs_refl = self.compute_phasecurve()

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

        # Only reflected contrast
        self.contrast_ppm_refl = (fobs_refl * 1.e6 / Fstar_band) * (self.planetaryradius / self.stellarradius) ** 2


        #fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=100)
        #ax.plot(np.linspace(0, 1, len(self.contrast_ppm_refl)), self.contrast_ppm_refl)
        #ax.set_title('Reflected component')
        #plt.show()


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
        albedo = 0.1
        redistribution = 0.1

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

        punto_system = ExoplanetarySystem(orbitalperiod=period,
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
                                                             mission='Kepler', response_nu=None, response_vals=None)

        return punto_system.compute_flux()

    run_model()

