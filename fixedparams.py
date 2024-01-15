import numpy as np

from astropy import units as u
from astropy import constants as const

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from .utils import th2ma


class Target:
    """Fixed parameters of the triple system."""

    def __init__(
        self,
        ref_all=None,
        ref_position=None,
        ref_proper_motion=None,
        ref_orbital=None,
        ref_parallax=None,
        ref_inclination=None,
        ref_omega_p=None,
    ):
        if ref_all == "Archibald":
            ref_position = ref_position or "Archibald"
            ref_proper_motion = ref_proper_motion or "Deller"
            ref_orbital = ref_orbital or "Archibald"
            ref_parallax = ref_parallax or "Deller"
            ref_inclination = ref_inclination or "Archibald CW"
            ref_omega_p = ref_omega_p or "Voisin"
        elif ref_all == "Voisin":
            ref_position = ref_position or "Voisin"
            ref_proper_motion = ref_proper_motion or "Deller"
            ref_orbital = ref_orbital or "Voisin"
            ref_parallax = ref_parallax or "Deller"
            ref_inclination = ref_inclination or "Voisin"
            ref_omega_p = ref_omega_p or "Voisin"

        self.ref_position = ref_position
        self.ref_proper_motion = ref_proper_motion
        self.ref_orbital = ref_orbital
        self.ref_parallax = ref_parallax
        self.ref_inclination = ref_inclination
        self.ref_omega_p = ref_omega_p

        # proper motion - Deller, Archibald, Voisin
        if ref_proper_motion == "Deller":
            self.mu_alpha_star = 5.90 * u.mas / u.yr
            self.mu_delta = -4.02 * u.mas / u.yr
        elif ref_proper_motion == "Archibald":
            self.mu_alpha_star = 4.51 * u.mas / u.yr
            self.mu_delta = 2.2 * u.mas / u.yr
        elif ref_proper_motion == "Voisin":
            self.mu_alpha_star = 5.01 * u.mas / u.yr
            self.mu_delta = -0.85 * u.mas / u.yr
        else:
            raise ValueError(f"Unrecognized ref_prorer_motion {ref_proper_motion}")

        # sky position - Archibald, Voisin
        if ref_position == "Archibald":
            self.psr_coord = SkyCoord(
                "03 37 43.82589 +17 15 14.8280",
                unit=("hourangle", "degree"),
                pm_ra_cosdec=self.mu_alpha_star,
                pm_dec=self.mu_delta,
            )
        elif ref_position == "Voisin":
            self.psr_coord = SkyCoord(
                "03 37 43.8270392 +17 15 14.81843",
                unit=("hourangle", "degree"),
                pm_ra_cosdec=self.mu_alpha_star,
                pm_dec=self.mu_delta,
            )
        else:
            raise ValueError(f"Unrecognized ref_position {ref_position}")

        # orbital parameters - Archibald, Voisin
        if ref_orbital == "Archibald":
            self.p_orb_i = 1.6293932 * u.day
            self.p_orb_o = 327.25685 * u.day

            self.t_asc_i = Time(55920.40771662, format="mjd")
            self.t_asc_o = Time(56233.93512, format="mjd")

            self.asini_i = 1.2175252 * const.c * u.s
            self.asini_o = 74.672629 * const.c * u.s

            eps1_i = +6.8833e-4 * u.dimensionless_unscaled
            eps2_i = -0.91401e-4 * u.dimensionless_unscaled

            eps1_o = +3.518595e-2 * u.dimensionless_unscaled
            eps2_o = -0.346313e-2 * u.dimensionless_unscaled

            self.ecc_i = np.sqrt(eps1_i**2 + eps2_i**2)
            self.ecc_o = np.sqrt(eps1_o**2 + eps2_o**2)

            self.arg_per_i = np.arctan2(eps1_i, eps2_i).to(u.deg) % (360 * u.deg)
            self.arg_per_o = np.arctan2(eps1_o, eps2_o).to(u.deg) % (360 * u.deg)

            ma_asc_i = -th2ma(self.arg_per_i, self.ecc_i)
            ma_asc_o = -th2ma(self.arg_per_o, self.ecc_o)

            self.t_per_i = self.t_asc_i - ma_asc_i.to_value(u.cycle) * self.p_orb_i
            self.t_per_o = self.t_asc_o - ma_asc_o.to_value(u.cycle) * self.p_orb_o

            self.k_i = (
                2 * np.pi * self.asini_i / self.p_orb_i / np.sqrt(1 - self.ecc_i**2)
            )
            self.k_o = (
                2 * np.pi * self.asini_o / self.p_orb_o / np.sqrt(1 - self.ecc_o**2)
            )
        elif ref_orbital == "Voisin":
            self.p_orb_i = 1.6294006 * u.day
            self.p_orb_o = 327.25539 * u.day

            # self.t_asc_i = Time(55917.1584, format="mjd")  # WRONG: differently defined
            # self.t_asc_o = Time(56230.19511, format="mjd") # WRONG: differently defined

            self.asini_i = 1.2175280 * const.c * u.s
            self.asini_o = 74.672374 * const.c * u.s

            eps1_i = +6.9365e-4 * u.dimensionless_unscaled
            eps2_i = -0.8544e-4 * u.dimensionless_unscaled

            eps1_o = +3.511431e-2 * u.dimensionless_unscaled
            eps2_o = -0.352480e-2 * u.dimensionless_unscaled

            self.ecc_i = np.sqrt(eps1_i**2 + eps2_i**2)
            self.ecc_o = np.sqrt(eps1_o**2 + eps2_o**2)

            self.arg_per_i = np.arctan2(eps1_i, eps2_i).to(u.deg) % (360 * u.deg)
            self.arg_per_o = np.arctan2(eps1_o, eps2_o).to(u.deg) % (360 * u.deg)

            self.t_per_i = Time(55917.5975, format="mjd")
            self.t_per_o = Time(56317.21976, format="mjd")

            ma_asc_i = -th2ma(self.arg_per_i, self.ecc_i)
            ma_asc_o = -th2ma(self.arg_per_o, self.ecc_o)

            self.t_asc_i = self.t_per_i + ma_asc_i.to_value(u.cycle) * self.p_orb_i
            self.t_asc_o = self.t_per_o + ma_asc_o.to_value(u.cycle) * self.p_orb_o

            self.k_i = (
                2 * np.pi * self.asini_i / self.p_orb_i / np.sqrt(1 - self.ecc_i**2)
            )
            self.k_o = (
                2 * np.pi * self.asini_o / self.p_orb_o / np.sqrt(1 - self.ecc_o**2)
            )
        else:
            raise ValueError(f"Unrecognized ref_orbital {ref_orbital}")

        # parallax prior - Deller
        if ref_parallax == "Deller":
            self.parallax_prior_mu = 0.88 * u.mas
            self.parallax_prior_sig = 0.06 * u.mas
        else:
            raise ValueError(f"Unrecognized ref_parallax {ref_parallax}")

        # inclination prior - Archibald CW, Archibald CCW, Voisin
        if ref_inclination == "Archibald CW":
            self.i_p_prior_mu = (180 - 39.262) * u.deg
            self.i_p_prior_sig = 0.004 * u.deg
        elif ref_inclination == "Archibald CCW":
            self.i_p_prior_mu = 39.262 * u.deg
            self.i_p_prior_sig = 0.004 * u.deg
        elif ref_inclination == "Voisin":
            self.i_p_prior_mu = (180 - 39.251) * u.deg
            self.i_p_prior_sig = 0.014 * u.deg
        else:
            raise ValueError(f"Unrecognized ref_inclination {ref_inclination}")

        # longitude of ascending node prior - Voisin
        if ref_omega_p == "Voisin":
            self.omega_p_prior_mu = (-44.34 * u.deg) % (360 * u.deg)
            self.omega_p_prior_sig = 0.14 * u.deg
        else:
            raise ValueError(f"Unrecognized ref_omega_p {ref_omega_p}")


class Observatory:
    """Fixed parameters related to observatory (i.e., its location on Earth)."""

    def __init__(self, telescope="GBT"):
        self.telescope = telescope

        # Set location on Earth
        if telescope == "GBT":
            self.earth_loc = EarthLocation.of_site("Green Bank Telescope")
        elif telescope == "WSRT":
            self.earth_loc = EarthLocation("6°36′12″E", "52°54′53″N")
