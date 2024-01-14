import numpy as np

from astropy import units as u
from astropy import constants as const

from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, EarthLocation

from scipy.optimize import curve_fit, minimize

from kepler import kepler as solve_kepler

import emcee

from IPython.display import display, Math

from .params import guess_pars_phen, guess_pars_phys, pars_phen2comm
from .utils import get_v_0_e, th2ma, gaussian, get_16_50_84
from .utils import UNIT_DVEFF, PARS_MDL_LABELS, PARS_UNIT_STRS, PARS_LABELS_UNITS


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


class Data:
    """Scaled effective velocity data."""

    def __init__(self, datafilename):
        self.datafilename = datafilename

        data = np.load(datafilename)

        self.t_obs = Time(data["t_mjd"], format="mjd", scale="utc")
        self.dveff_obs = data["dveff_obs"] * np.sqrt(1000) * UNIT_DVEFF
        self.dveff_err_original = data["dveff_err"] * np.sqrt(1000) * UNIT_DVEFF
        self.dveff_err = self.dveff_err_original

        self.tlim = [np.min(self.t_obs.mjd) - 20, np.max(self.t_obs.mjd) + 40]

        self._equad = 0.0 * UNIT_DVEFF

    @property
    def equad(self):
        return self._equad

    @equad.setter
    def equad(self, new_equad):
        self._equad = new_equad
        self.dveff_err = np.sqrt(self.dveff_err_original**2 + new_equad**2)


class ModelBase:
    """Base class for models of scaled effective velocity as function of time."""

    def __init__(self, target, observatory):
        self.target = target
        self.observatory = observatory

    def get_earth_terms(self, t):
        psr_frame = SkyOffsetFrame(origin=self.target.psr_coord)
        v_earth_xyz = (
            self.observatory.earth_loc.get_gcrs(t)
            .transform_to(psr_frame)
            .velocity.d_xyz
        )
        scaled_v_earth_xyz = v_earth_xyz / get_v_0_e()
        return scaled_v_earth_xyz

    def model_dveff_abs(
        self, pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
    ):
        dveff_signed = self.model_dveff_signed(
            pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
        )

        return np.abs(dveff_signed).to(UNIT_DVEFF)

    def get_dveff_signed_from_t(self, pars, t):
        (sin_term_i, cos_term_i, sin_term_o, cos_term_o) = self.get_pulsar_terms(t)

        scaled_v_earth_xyz = self.get_earth_terms(t)

        dveff_signed = self.model_dveff_signed(
            pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
        )

        return dveff_signed


class ModelPhen(ModelBase):
    """Phenomenological model for scaled effective velocity as function of time."""

    ndim = 5

    def get_pulsar_terms(self, t):
        sin_omg_i = np.sin(self.target.arg_per_i)
        cos_omg_i = np.cos(self.target.arg_per_i)
        sin_omg_o = np.sin(self.target.arg_per_o)
        cos_omg_o = np.cos(self.target.arg_per_o)

        ma_i = ((t - self.target.t_per_i) / self.target.p_orb_i).decompose() * u.cycle
        ma_o = ((t - self.target.t_per_o) / self.target.p_orb_o).decompose() * u.cycle

        _, cos_th_i, sin_th_i = solve_kepler(ma_i.to_value(u.rad), self.target.ecc_i)
        _, cos_th_o, sin_th_o = solve_kepler(ma_o.to_value(u.rad), self.target.ecc_o)

        sin_ph_i = sin_omg_i * cos_th_i + cos_omg_i * sin_th_i
        cos_ph_i = cos_omg_i * cos_th_i - sin_omg_i * sin_th_i
        sin_ph_o = sin_omg_o * cos_th_o + cos_omg_o * sin_th_o
        cos_ph_o = cos_omg_o * cos_th_o - sin_omg_o * sin_th_o

        return sin_ph_i, cos_ph_i, sin_ph_o, cos_ph_o

    def model_dveff_signed(
        self, pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
    ):
        amp_e_ra_cosdec = pars["amp_e_ra_cosdec"]
        amp_e_dec = pars["amp_e_dec"]
        amp_ps = pars["amp_ps"]
        amp_pc = pars["amp_pc"]
        dveff_c = pars["dveff_c"]

        dveff_e = (
            amp_e_ra_cosdec * scaled_v_earth_xyz[1] + amp_e_dec * scaled_v_earth_xyz[2]
        )

        k_ratio = self.target.k_o / self.target.k_i
        sin_term_combi = sin_term_i + k_ratio * sin_term_o
        cos_term_combi = cos_term_i + k_ratio * cos_term_o
        dveff_p = amp_ps * sin_term_combi - amp_pc * cos_term_combi

        dveff = dveff_p - dveff_e + dveff_c

        return (dveff).to(UNIT_DVEFF)

    def pars_fit2mdl(self, pars_fit):
        """Convert list of parameters for fitting to dict of Astropy Quantities."""

        amp_e_ra_cosdec, amp_e_dec, amp_ps, amp_pc, dveff_c = pars_fit

        pars_mdl = {
            "amp_e_ra_cosdec": amp_e_ra_cosdec * UNIT_DVEFF,
            "amp_e_dec": amp_e_dec * UNIT_DVEFF,
            "amp_ps": amp_ps * UNIT_DVEFF,
            "amp_pc": amp_pc * UNIT_DVEFF,
            "dveff_c": dveff_c * UNIT_DVEFF,
        }

        return pars_mdl

    def pars_mdl2fit(self, pars_mdl):
        """Convert dict of Astropy Quantities to list of parameters for fitting."""

        pars_fit = np.array(
            [
                pars_mdl["amp_e_ra_cosdec"].to_value(UNIT_DVEFF),
                pars_mdl["amp_e_dec"].to_value(UNIT_DVEFF),
                pars_mdl["amp_ps"].to_value(UNIT_DVEFF),
                pars_mdl["amp_pc"].to_value(UNIT_DVEFF),
                pars_mdl["dveff_c"].to_value(UNIT_DVEFF),
            ]
        )

        return pars_fit

    def get_dveff_signed_components_from_t(self, pars, t):
        sin_term_i, cos_term_i, sin_term_o, cos_term_o = self.get_pulsar_terms(t)
        scaled_v_earth_xyz = self.get_earth_terms(t)

        # full model
        dveff_full = self.model_dveff_signed(
            pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
        )

        # extract pulsar parameters
        k_ratio = self.target.k_o / self.target.k_i
        ecc_i = self.target.ecc_i
        ecc_o = self.target.ecc_o
        arg_per_i = self.target.arg_per_i
        arg_per_o = self.target.arg_per_o

        # get pulsar signal amplitude and phase
        pars_comm = pars_phen2comm(pars)
        amp_p = pars_comm["amp_p"]
        chi_p = pars_comm["chi_p"]

        # only inner pulsar orbit
        pars_inner = pars.copy()
        pars_inner["amp_e_ra_cosdec"] = 0 * UNIT_DVEFF
        pars_inner["amp_e_dec"] = 0 * UNIT_DVEFF
        pars_inner["dveff_c"] = amp_p * ecc_i * np.sin(arg_per_i - chi_p)
        dveff_inner = self.model_dveff_signed(
            pars_inner, sin_term_i, cos_term_i, 0, 0, scaled_v_earth_xyz
        )

        # only outer pulsar orbit
        pars_outer = pars.copy()
        pars_outer["amp_e_ra_cosdec"] = 0 * UNIT_DVEFF
        pars_outer["amp_e_dec"] = 0 * UNIT_DVEFF
        pars_outer["dveff_c"] = k_ratio * amp_p * ecc_o * np.sin(arg_per_o - chi_p)
        dveff_outer = self.model_dveff_signed(
            pars_outer, 0, 0, sin_term_o, cos_term_o, scaled_v_earth_xyz
        )

        # only Earth
        pars_earth = pars.copy()
        pars_earth["amp_ps"] = 0 * UNIT_DVEFF
        pars_earth["amp_pc"] = 0 * UNIT_DVEFF
        pars_earth["dveff_c"] = 0 * UNIT_DVEFF
        dveff_earth = self.model_dveff_signed(
            pars_earth, 0, 0, 0, 0, scaled_v_earth_xyz
        )

        return dveff_full, dveff_inner, dveff_outer, dveff_earth


class ModelPhys(ModelBase):
    """Physical model for scaled effective velocity as function of time."""

    ndim = 6

    def get_pulsar_terms(self, t):
        sin_omg_i = np.sin(self.target.arg_per_i)
        cos_omg_i = np.cos(self.target.arg_per_i)
        sin_omg_o = np.sin(self.target.arg_per_o)
        cos_omg_o = np.cos(self.target.arg_per_o)

        ma_i = ((t - self.target.t_per_i) / self.target.p_orb_i).decompose() * u.cycle
        ma_o = ((t - self.target.t_per_o) / self.target.p_orb_o).decompose() * u.cycle

        _, cos_th_i, sin_th_i = solve_kepler(ma_i.to_value(u.rad), self.target.ecc_i)
        _, cos_th_o, sin_th_o = solve_kepler(ma_o.to_value(u.rad), self.target.ecc_o)

        sin_ph_i = sin_omg_i * cos_th_i + cos_omg_i * sin_th_i
        cos_ph_i = cos_omg_i * cos_th_i - sin_omg_i * sin_th_i
        sin_ph_o = sin_omg_o * cos_th_o + cos_omg_o * sin_th_o
        cos_ph_o = cos_omg_o * cos_th_o - sin_omg_o * sin_th_o

        sin_term_i = sin_ph_i + self.target.ecc_i * sin_omg_i
        cos_term_i = cos_ph_i + self.target.ecc_i * cos_omg_i
        sin_term_o = sin_ph_o + self.target.ecc_o * sin_omg_o
        cos_term_o = cos_ph_o + self.target.ecc_o * cos_omg_o

        return sin_term_i, cos_term_i, sin_term_o, cos_term_o

    def model_dveff_signed(
        self, pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
    ):
        cosi_p = pars["cosi_p"]
        omega_p = pars["omega_p"]
        d_p = pars["d_p"]
        s = pars["s"]
        xi = pars["xi"]
        v_lens = pars["v_lens"]

        sini_p = np.sqrt(1 - cosi_p**2)
        delta_omega_p = xi - omega_p
        d_eff = (1 - s) / s * d_p

        sinxi, cosxi = np.sin(xi), np.cos(xi)

        mu = self.target.mu_alpha_star * sinxi + self.target.mu_delta * cosxi
        v_p_sys = (d_p * mu).to(u.km / u.s, equivalencies=u.dimensionless_angles())

        v_p_i_0 = self.target.k_i / sini_p
        v_p_o_0 = self.target.k_o / sini_p

        v_i_orb = -v_p_i_0 * (
            np.cos(delta_omega_p) * sin_term_i
            - np.sin(delta_omega_p) * cos_term_i * cosi_p
        )
        v_o_orb = -v_p_o_0 * (
            np.cos(delta_omega_p) * sin_term_o
            - np.sin(delta_omega_p) * cos_term_o * cosi_p
        )

        v_p = v_p_sys + v_i_orb + v_o_orb

        v_earth_xyz = scaled_v_earth_xyz * get_v_0_e()
        v_earth = v_earth_xyz[1] * sinxi + v_earth_xyz[2] * cosxi

        v_eff = 1 / s * v_lens - ((1 - s) / s) * v_p - v_earth

        return (v_eff / np.sqrt(d_eff)).to(UNIT_DVEFF)

    def get_log_prior(self, pars_fit, priors):
        cosi_p, omega_p, d_p, s, xi, v_lens = pars_fit

        if (cosi_p < -1) or (cosi_p > 1) or (d_p <= 0) or (s <= 0) or (s >= 1):
            log_prior = -np.inf
        else:
            prior = 1.0
            if "parallax" in priors:
                prior *= self.get_prior_d_p(d_p)
            if "inclination" in priors:
                prior *= self.get_prior_i_p(cosi_p)
            if "omega" in priors:
                prior *= self.get_prior_omega_p(omega_p)
            log_prior = np.log(prior)

        return log_prior

    def get_prior_d_p(self, d_p):
        parallax = (d_p * u.kpc).to(u.mas, equivalencies=u.parallax())
        prior = 1.0
        # Constant space density
        prior *= d_p**2
        # Parallax constraint
        prior *= gaussian(
            parallax.to_value(u.mas),
            self.target.parallax_prior_mu.to_value(u.mas),
            self.target.parallax_prior_sig.to_value(u.mas),
        )
        return prior

    def get_prior_i_p(self, cosi_p):
        i_p = np.arccos(cosi_p)
        return gaussian(
            i_p,
            self.target.i_p_prior_mu.to_value(u.rad),
            self.target.i_p_prior_sig.to_value(u.rad),
        )

    def get_prior_omega_p(self, omega_p):
        return gaussian(
            omega_p,
            self.target.omega_p_prior_mu.to_value(u.rad),
            self.target.omega_p_prior_sig.to_value(u.rad),
        )

    def pars_fit2mdl(self, pars_fit):
        """Convert list of parameters for fitting to dict of Astropy Quantities."""

        cosi_p, omega_p, d_p, s, xi, v_lens = pars_fit

        pars_mdl = {
            "cosi_p": cosi_p * u.dimensionless_unscaled,
            "omega_p": (omega_p * u.rad).to(u.deg),
            "d_p": d_p * u.kpc,
            "s": s * u.dimensionless_unscaled,
            "xi": (xi * u.rad).to(u.deg),
            "v_lens": v_lens * u.km / u.s,
        }

        return pars_mdl

    def pars_mdl2fit(self, pars_mdl):
        """Convert dict of Astropy Quantities to list of parameters for fitting."""

        pars_fit = np.array(
            [
                pars_mdl["cosi_p"].to_value(u.dimensionless_unscaled),
                pars_mdl["omega_p"].to_value(u.rad),
                pars_mdl["d_p"].to_value(u.kpc),
                pars_mdl["s"].to_value(u.dimensionless_unscaled),
                pars_mdl["xi"].to_value(u.rad),
                pars_mdl["v_lens"].to_value(u.km / u.s),
            ]
        )

        return pars_fit

    def get_dveff_signed_components_from_t(self, pars, t):
        (sin_term_i, cos_term_i, sin_term_o, cos_term_o) = self.get_pulsar_terms(t)

        scaled_v_earth_xyz = self.get_earth_terms(t)

        # HACK: The hack below of setting independent variables to zero doesn't
        # quite work in separating components: it doesn't remove the constant
        # offset, which also includes a small component from the pulsar orbits
        # due to the eccentricities

        scaled_v_earth_xyz_zero = np.zeros((3, len(t)))

        dveff_full = self.model_dveff_signed(
            pars, sin_term_i, cos_term_i, sin_term_o, cos_term_o, scaled_v_earth_xyz
        )
        dveff_inner = self.model_dveff_signed(
            pars, sin_term_i, cos_term_i, 0, 0, scaled_v_earth_xyz_zero
        )
        dveff_outer = self.model_dveff_signed(
            pars, 0, 0, sin_term_o, cos_term_o, scaled_v_earth_xyz_zero
        )
        dveff_earth = self.model_dveff_signed(pars, 0, 0, 0, 0, scaled_v_earth_xyz)

        return dveff_full, dveff_inner, dveff_outer, dveff_earth


class FitBase:
    """Base class for fits of model with a given dataset."""

    def __init__(self, model, data, priors={}):
        self.model = model
        self.data = data
        self.priors = set(priors)
        self.ndim = model.ndim

        # Precompute pulsar's time-dependent terms
        (
            self.sin_term_i_obs,
            self.cos_term_i_obs,
            self.sin_term_o_obs,
            self.cos_term_o_obs,
        ) = self.model.get_pulsar_terms(self.data.t_obs)

        # Precompute Earth's time-dependent terms
        self.scaled_v_earth_xyz_obs = self.model.get_earth_terms(self.data.t_obs)

        # Free parameters and number of degrees of freedom
        self.nfixed = len(priors)
        if "parallax" in priors:
            self.nfixed -= 1
        self.nfree = self.ndim - self.nfixed
        self.ndof = len(self.data.t_obs) - self.nfree

    def get_chi2(self, pars_mdl):
        dveff_mdl = self.model.model_dveff_abs(
            pars_mdl,
            self.sin_term_i_obs,
            self.cos_term_i_obs,
            self.sin_term_o_obs,
            self.cos_term_o_obs,
            self.scaled_v_earth_xyz_obs,
        )
        chi2 = np.sum(((self.data.dveff_obs - dveff_mdl) / self.data.dveff_err) ** 2)
        return chi2

    def print_result(self, pars_mdl):
        """Quick view of parameter values and reduced chi^2."""

        for par_name in pars_mdl:
            print(f"{par_name:8s} {pars_mdl[par_name]:8.2f}")

        if "cosi_p" in pars_mdl:
            i_p = np.arccos(pars_mdl["cosi_p"]) / np.pi * 180 * u.deg
            print(f"\ni_p      {i_p:8.2f}")

        print(f"\nchi2red  {self.get_chi2(pars_mdl)/self.ndof:8.2f}")

        print(f"\nequad    {self.data.equad:8.2f}")

    def find_equad(self, pars_mdl_init=None, init_equad=None, tol=1e-4, maxiter=100):
        """Crude iterative method to find and set EQUAD for which chi^2 = 1"""

        if pars_mdl_init is None:
            pars_mdl_init = self.guess_pars(self.model.target)

        if init_equad is None:
            init_equad = self.data.dveff_err_original.mean()

        chi2diff = np.inf
        niter = 1
        pars_mdl = pars_mdl_init
        new_equad = init_equad
        print(" chi2red   equad")
        while (chi2diff > tol) and niter <= maxiter:
            cur_equad = new_equad

            self.data.equad = cur_equad
            pars_mdl = self.optimize(pars_mdl_init=pars_mdl)
            chi2red = self.get_chi2(pars_mdl) / self.ndof
            print(f"{chi2red:9.6f} {cur_equad:9.6f}")

            new_equad = cur_equad * chi2red
            chi2diff = np.abs(chi2red - 1)
            niter += 1


class FitPhen(FitBase):
    """Fit of phenomenological model with a given dataset."""

    def guess_pars(self, target):
        return guess_pars_phen(target)

    def optimize(self, pars_mdl_init=None):
        if pars_mdl_init is None:
            pars_mdl_init = self.guess_pars(self.model.target)
        pars_fit_init = self.model.pars_mdl2fit(pars_mdl_init)

        indep_vars = np.array(
            [
                self.sin_term_i_obs.to_value(u.dimensionless_unscaled),
                self.cos_term_i_obs.to_value(u.dimensionless_unscaled),
                self.sin_term_o_obs.to_value(u.dimensionless_unscaled),
                self.cos_term_o_obs.to_value(u.dimensionless_unscaled),
                self.scaled_v_earth_xyz_obs[0, :].value,
                self.scaled_v_earth_xyz_obs[1, :].value,
                self.scaled_v_earth_xyz_obs[2, :].value,
            ]
        )

        def model_dveff_fit(indep_vars, *pars_fit):
            sin_term_i = indep_vars[0, :]
            cos_term_i = indep_vars[1, :]
            sin_term_o = indep_vars[2, :]
            cos_term_o = indep_vars[3, :]
            scaled_v_earth_x = indep_vars[4, :]
            scaled_v_earth_y = indep_vars[5, :]
            scaled_v_earth_z = indep_vars[6, :]

            scaled_v_earth_xyz = np.array(
                [
                    scaled_v_earth_x,
                    scaled_v_earth_y,
                    scaled_v_earth_z,
                ]
            )

            pars_mdl = self.model.pars_fit2mdl(pars_fit)

            dveff_abs = self.model.model_dveff_abs(
                pars_mdl,
                sin_term_i,
                cos_term_i,
                sin_term_o,
                cos_term_o,
                scaled_v_earth_xyz,
            )

            return dveff_abs.to_value(UNIT_DVEFF)

        popt, pcov = curve_fit(
            f=model_dveff_fit,
            xdata=indep_vars,
            ydata=self.data.dveff_obs.to_value(UNIT_DVEFF),
            p0=pars_fit_init,
            sigma=self.data.dveff_err.to_value(UNIT_DVEFF),
            absolute_sigma=True,
        )

        pars_mdl = self.model.pars_fit2mdl(popt)

        self.best_fit = pars_mdl

        return pars_mdl


class FitPhys(FitBase):
    """Fit of physical model with a given dataset."""

    def guess_pars(self, target):
        return guess_pars_phys(target)

    def get_log_likelihood(self, pars_fit):
        pars_mdl = self.model.pars_fit2mdl(pars_fit)
        chi2 = self.get_chi2(pars_mdl)
        log_likelihood = -0.5 * chi2
        return log_likelihood

    def get_log_prob(self, pars_fit):
        log_prior = self.model.get_log_prior(pars_fit, self.priors)
        if log_prior == -np.inf:
            log_prob = -np.inf
        else:
            log_likelihood = self.get_log_likelihood(pars_fit)
            log_prob = log_prior + log_likelihood
        return log_prob

    def optimize(
        self, pars_mdl_init=None, method="Nelder-Mead", options={"maxiter": 100000}
    ):
        if pars_mdl_init is None:
            pars_mdl_init = self.guess_pars(self.model.target)
        pars_fit_init = self.model.pars_mdl2fit(pars_mdl_init)

        def get_neg_log_prob(*args):
            return -self.get_log_prob(*args)

        res = minimize(
            get_neg_log_prob,
            pars_fit_init,
            method=method,
            options=options,
        )

        pars_mdl = self.model.pars_fit2mdl(res.x)

        self.best_fit = pars_mdl

        return pars_mdl


class MCMC:
    """Object to do Markov-chain Monte Carlo parameter estimation of a fit."""

    def __init__(self, fit):
        self.fit = fit
        self.ndim = fit.ndim

    def prep_mcmc(
        self, nwalker=64, pars_mdl_init=None, spread_init=1e-8, backend_filename=None
    ):
        """Prepare MCMC sampler."""

        self.nwalker = nwalker

        # Setup backend
        if backend_filename is None:
            prior_str = "".join([p[0] for p in self.fit.priors])
            time_str = Time.now().strftime("%Y_%m_%d__%H_%M_%S")
            backend_filename = f"results/{prior_str}/gbt_ecc_2yr_{time_str}.h5"
        print(f"Creating backend: {backend_filename}")
        self.backend_filename = backend_filename
        self.backend = emcee.backends.HDFBackend(backend_filename)
        self.backend.reset(nwalker, self.ndim)

        # Setup sampler
        self.sampler = emcee.EnsembleSampler(
            nwalker, self.ndim, self.fit.get_log_prob, backend=self.backend
        )

        # Initial positions of walkers
        if pars_mdl_init is None:
            pars_mdl_init = self.fit.optimize()
        pars_fit_init = self.fit.model.pars_mdl2fit(pars_mdl_init)
        rng = np.random.default_rng(seed=1234)
        self.init_pos = pars_fit_init + spread_init * rng.normal(
            size=(nwalker, self.ndim)
        )

    def run_mcmc(
        self, max_n=30000, ntau_goal=100, tol_tau=0.01, conv_check_interval=1000
    ):
        self.max_n = max_n
        self.ntau_goal = ntau_goal
        self.tol_tau = tol_tau
        self.conv_check_interval = conv_check_interval

        # We'll track how the average autocorrelation time estimate changes
        self.index = 0
        max_n_check = int(np.ceil(max_n / conv_check_interval))
        self.autocorr = np.empty((max_n_check, self.ndim))

        # This will be useful to testing convergence
        old_tau = np.inf

        # Now we'll sample for up to max_n steps
        for sample in self.sampler.sample(
            self.init_pos, iterations=max_n, progress=True
        ):
            # Only check convergence every n steps
            if self.sampler.iteration % conv_check_interval:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = self.sampler.get_autocorr_time(tol=0)
            self.autocorr[self.index, :] = tau
            self.index += 1

            # Check convergence
            converged = np.all(tau * ntau_goal < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < tol_tau)
            if converged:
                break
            old_tau = tau

        if converged:
            print("MCMC done and converged!")
        else:
            print("MCMC done, but not converged...")

    def trim_chain(self, backend_filename=None):
        """Remove burn-in and thin chain."""

        if backend_filename is None:
            backend_filename = self.backend_filename

        reader = emcee.backends.HDFBackend(backend_filename)

        tau = reader.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))

        self.flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

        print(f"burn-in: {burnin}")
        print(f"thin: {thin}")
        print(f"flat chain shape: {self.flat_samples.shape}")

    def scale_samples(self):
        """Add units to the samples, and convert cosi_p to i_p."""

        scalings = [1, 180 / np.pi, 1, 180 / np.pi, 1, 1]

        self.scaled_samples = self.flat_samples * scalings
        self.scaled_samples[:, 0] = np.arccos(self.scaled_samples[:, 0]) * 180 / np.pi

    def print_result(self):
        for i in range(self.ndim):
            mcmc = np.percentile(self.scaled_samples[:, i], get_16_50_84())
            q = np.diff(mcmc)
            txt = (
                f"{PARS_MDL_LABELS[i][1:-1]} ="
                f"{mcmc[1]:.3f}_{{ -{q[0]:.3f} }}^{{ +{q[1]:.3f} }}"
                f"\\; \\mathrm{{ {PARS_UNIT_STRS[i][2:-1]} }}"
            )
            display(Math(txt))
