import numpy as np

from astropy import units as u

from astropy.coordinates import SkyOffsetFrame

from kepler import kepler as solve_kepler

from .freeparams import pardict_phen, pardict_phys, pars_phen2comm
from .utils import v_0_e, gaussian, UNIT_DVEFF


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
        scaled_v_earth_xyz = v_earth_xyz / v_0_e
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

    def pars_fit2mdl(self, pars_fit):
        """Convert array of parameters for fitting to dict of Astropy Quantities."""

        pars_mdl = {}
        for key, parfit in zip(self.pardict, pars_fit):
            pars_mdl[key] = parfit * self.pardict[key].unit_ap

        return pars_mdl

    def pars_mdl2fit(self, pars_mdl):
        """Convert dict of Astropy Quantities to array of parameters for fitting."""

        pars_fit = np.empty(self.ndim)
        for i, key in enumerate(self.pardict):
            pars_fit[i] = pars_mdl[key].to_value(self.pardict[key].unit_ap)

        return pars_fit


class ModelPhen(ModelBase):
    """Phenomenological model for scaled effective velocity as function of time."""

    pardict = pardict_phen
    ndim = len(pardict)

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

    pardict = pardict_phys
    ndim = len(pardict)

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

        v_0_i = self.target.k_i / sini_p
        v_0_o = self.target.k_o / sini_p

        v_p_orb_i = -v_0_i * (
            np.cos(delta_omega_p) * sin_term_i
            - np.sin(delta_omega_p) * cos_term_i * cosi_p
        )
        v_p_orb_o = -v_0_o * (
            np.cos(delta_omega_p) * sin_term_o
            - np.sin(delta_omega_p) * cos_term_o * cosi_p
        )

        v_p = v_p_sys + v_p_orb_i + v_p_orb_o

        v_earth_xyz = scaled_v_earth_xyz * v_0_e
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
