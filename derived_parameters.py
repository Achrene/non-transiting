import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt

import os
import pandas as pd
import cmocean as cm

from matplotlib import rc
rc('image', origin='lower')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 16})
rc('text', usetex=True)
rc('lines', linewidth=0.5)
rc('ytick', right=True, direction = 'in')
rc('xtick', top = True, direction = 'in')
rc('axes', axisbelow = False)
rc('mathtext', fontset = 'cm')


class PlanetarySystem:
    def __init__(self, stellarmass, stellarradius, orbitalperiod, effectivetemperature, planetarytemperature,
                 planetaryradius, inclination, Abeam, Arefl, Aellip, alpha_refl, alpha_ellip, alpha_beam,
                 planetarymasssini, semimajoraxis):
        self.stellarmass = stellarmass  # solar mass
        self.stellarradius = stellarradius  # solar radius
        self.orbitalperiod = orbitalperiod  # days
        self.effectivetemperature = effectivetemperature  # K Stellar effective temperature
        self.planetarytemperature = planetarytemperature  # K Teq
        self.planetaryradius = planetaryradius  # Jupiter radius
        self.inclination = inclination  # degrees
        self.Abeam = Abeam  # ppm
        self.Arefl = Arefl  # ppm
        self.Aellip = Aellip  # ppm
        self.alpha_refl = alpha_refl  # no unit
        self.alpha_ellip = alpha_ellip   # no unit
        self.alpha_beam = alpha_beam   # no unit
        self.planetarymasssini = planetarymasssini  # in Jupiter mass
        self.semimajoraxis = semimajoraxis  # UA

    def compute_sini(self):
        return np.sin(self.inclination * 2 * np.pi / 360)

    def compute_alpha_ellip(self):
        return -2.2 * (10 ** -4) * self.effectivetemperature + 2.6

    def compute_alpha_beam(self):
        return -6 * (10 ** -4) * self.effectivetemperature + 7.2

    def compute_alpha_refl(self):

        sini = self.compute_sini()

        alpha_refl = self.Arefl / (57 * sini * self.stellarmass ** (-2 / 3) * (self.orbitalperiod / 1) ** (-4 / 3)
                                   * self.planetaryradius ** 2)
        return alpha_refl

    def estimate_semi_major_axis(self):

        orbitalperiod_seconds = self.orbitalperiod * 24 * 3600
        stellarmass_kilo = self.stellarmass * c.M_sun

        self.semimajoraxis = ((orbitalperiod_seconds ** 2 * c.G.value * stellarmass_kilo.value) /
                              (4 * (np.pi ** 2))) ** (1/3)
        self.semimajoraxis = self.semimajoraxis * 6.68459e-12  # conversion into UA

        return self.semimajoraxis

    def estimate_grazing_inclination(self):
        stellarradius_meter = self.stellarradius * c.R_sun
        semimajoraxis_meter = self.semimajoraxis / 6.68459e-12
        imin = np.arccos(stellarradius_meter.value / semimajoraxis_meter) * 360 / (2 * np.pi)  # in degrees

        return imin

    def estimate_mass_Abeam(self):
        alpha_beam = self.compute_alpha_beam()

        Mp_sini = self.Abeam / (2.7 * alpha_beam * (self.orbitalperiod / 1) ** (-1 / 3) * self.stellarmass ** (-2 / 3))
        # Lillo-Box, 2021

        return Mp_sini   # in Jupiter mass

    def estimate_mass_Aellip(self):
        alpha_ellip = self.compute_alpha_ellip()

        sini = self.compute_sini()

        Mp_sini = self.Aellip / (13 * alpha_ellip * sini * self.stellarradius ** 3 * self.stellarmass ** (-2)
                            * (self.orbitalperiod / 1) ** (-2))  # Lillo-Box, 2021
        return Mp_sini   # in Jupiter mass

    def estimate_Abeam(self):
        alpha_beam = self.compute_alpha_beam()

        self.Abeam = 2.7 * alpha_beam * (self.orbitalperiod / 1) ** (-1 / 3) * self.stellarmass ** (-2 / 3) \
                     * self.planetarymasssini

        return self.Abeam

    def estimate_Aellip(self):
        alpha_ellip = self.compute_alpha_ellip()

        sini = self.compute_sini()

        self.Aellip = 13 * alpha_ellip * sini * self.stellarradius ** 3 * self.stellarmass ** (-2) \
                      * (self.orbitalperiod / 1) ** (-2) * self.planetarymasssini

        return self.Aellip

    def estimate_radius_Arefl(self):
        # ceoff_refl is alpha_refl * (Rp / R_jup) ** 2 where Rp is the planet radius and i the orbit inclination

        sini = self.compute_sini()

        coeff_refl = self.Arefl / (57 * sini * self.stellarmass ** (-2 / 3) * (self.orbitalperiod / 1) ** (-4 / 3))
        # Lillo-Box, 2021

        Rp = np.sqrt(coeff_refl / self.alpha_refl)
        return Rp  # in Jupiter radius

    def estimate_beam_ellip_ratio(self):
        expected_ratio = 5 * (self.compute_alpha_ellip()/self.compute_alpha_beam()) \
                         * self.stellarradius ** 3 * self.stellarmass ** (-4/3) \
                         * (self.orbitalperiod/1) ** (-5/3) * self.compute_sini()
        obtained_ratio = self.Aellip / self.Abeam
        return expected_ratio, obtained_ratio

    def get_sini(self):
        """
        This function is a diagnostic for the system inclination based on the Allip and Abeam fitted params.
        Comes from Eq. 12 in Shporer 2017
        :rtype: object
        """
        obtained_ratio = self.Aellip / self.Abeam
        A = 5 * (self.compute_alpha_ellip() / self.compute_alpha_beam()) \
                         * self.stellarradius ** 3 * self.stellarmass ** (-4 / 3) \
                         * (self.orbitalperiod / 1) ** (-5 / 3)
        sini = obtained_ratio * (1 / A)
        return sini


if __name__ == '__main__':

    input_file = '/Users/ah258874/Documents/STScI_fellowship/Punto/fits/photometric_fit/' \
                 'refl_ellip_beam_T0/results/params.txt'
    params = np.loadtxt(input_file, dtype=object)

    A0 = float(params[0][1])
    Arefl = float(params[1][1])
    T0 = float(params[2][1])
    Aellip = np.abs(float(params[3][1]))
    Abeam = float(params[4][1])

    # proxy for Punto, let's take WASP-18b or WASP-121b

    # Arefl_wasp18 = 174  # Shporer 2019 Kepler
    Arefl_wasp18 = 267  # Cullen 2024 TESS

    wasp_18 = PlanetarySystem(stellarmass=1.294, stellarradius=1.319, orbitalperiod=0.94,
                              effectivetemperature=6226, planetarytemperature=2429, planetaryradius=1.16,
                              inclination=83.5, Abeam=None, Arefl=Arefl_wasp18,
                              Aellip=None, alpha_refl=None, alpha_ellip=None, alpha_beam=None, planetarymasssini=None,
                              semimajoraxis=None)

    alpha_refl_wasp18 = wasp_18.compute_alpha_refl()
    wasp_18.alpha_refl = alpha_refl_wasp18

    A_refl_wasp121 = 214  # Wong 2020, Cullen 2024 TESS

    wasp_121 = PlanetarySystem(stellarmass=1.358, stellarradius=1.437, orbitalperiod=1.2749,
                               effectivetemperature=6776, planetarytemperature=2358, planetaryradius=1.753,
                               inclination=88.49, Abeam=None, Arefl=A_refl_wasp121,
                               Aellip=None, alpha_refl=None, alpha_ellip=None, alpha_beam=None,
                               planetarymasssini=None, semimajoraxis=None)

    alpha_refl_wasp121 = wasp_121.compute_alpha_refl()
    wasp_121.alpha_refl = alpha_refl_wasp121

    # let's use this alpha refl coefficient as a proxy for Punto
    punto = PlanetarySystem(stellarmass=1.36, stellarradius=1.54, orbitalperiod=0.604734,
                            effectivetemperature=6396, planetarytemperature=None, planetaryradius=None,
                            inclination=None, Abeam=Abeam, Arefl=Arefl,
                            Aellip=Aellip, alpha_refl=alpha_refl_wasp18, alpha_ellip=None, alpha_beam=None,
                            planetarymasssini=None, semimajoraxis=None)

    # let's find the inclination based on the Abeam and Aellip fitted params
    sini = punto.get_sini()
    i = np.arcsin(sini) / (2 * np.pi) * 360

    # let's replace the inclination with the obtained value
    punto.inclination = i

    # let's find the mass and the radius of the candidate
    mass_Abeam = punto.estimate_mass_Abeam()  # Jupiter mass
    mass_Aellip = punto.estimate_mass_Aellip()  # Jupiter mass
    radius = punto.estimate_radius_Arefl()  # Jupiter radius

    # Let's compute Abeam and Aellip based on the mass coming from the RVs
    punto.planetarymasssini = 0.0078  # Jupiter mass, from Hans' analysis
    semimajoraxis_estimation = punto.estimate_semi_major_axis()  # AU
    punto.semimajoraxis = semimajoraxis_estimation
    imin = punto.estimate_grazing_inclination()  # degrees

    punto.inclination = 5 # degrees

    Abeam_estimation = punto.estimate_Abeam()  # ppm
    Aellip_estimation = punto.estimate_Aellip()  # ppm

    print('done')