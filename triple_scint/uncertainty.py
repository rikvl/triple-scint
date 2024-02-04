import numpy as np

from astropy import units as u
from astropy import uncertainty as unc

from IPython.display import display, Math

from .freeparams import pardict_phen, pardict_phys, pardict_pres
from .freeparams import pars_phen2phys_d_p, pars_phen2phys_cosi_p, pars_phys2pres
from .utils import gaussian, UNIT_DVEFF, tex_uncertainties


class SampleBase:
    """Monte Carlo sample."""

    def display_samp_quantiles(self):
        txt_all = ""
        for par, samp in self.samp_dict.items():
            result = tex_uncertainties(samp.distribution.value, self.weights)
            txt = (
                f"{self.pardict[par].symbol} &= {result} "
                f"\\; {self.pardict[par].unit_str} \\\\[0.5em]"
            )
            txt_all += txt

        txt_all = "\\begin{align}" + txt_all + "\\end{align}"
        display(Math(txt_all))


class SamplePhen(SampleBase):
    """Monte Carlo sample in phenomenological parameters."""

    npar = 5
    pardict = pardict_phen

    def __init__(self, fit, nmc=40000):
        self.nmc = nmc
        self.fit = fit
        self.target = fit.model.target
        self.weights = None

        np.random.seed(654321)

        # generate samples of the correlated phenomenological parameters
        samp_init = np.random.multivariate_normal(
            fit.curve_fit_popt, fit.curve_fit_pcov, size=nmc
        )

        # separate phenomenological parameters
        amp_e_ra_cosdec = unc.Distribution(samp_init[:, 0] * UNIT_DVEFF)
        amp_e_dec = unc.Distribution(samp_init[:, 1] * UNIT_DVEFF)
        amp_ps = unc.Distribution(samp_init[:, 2] * UNIT_DVEFF)
        amp_pc = unc.Distribution(samp_init[:, 3] * UNIT_DVEFF)
        dveff_c = unc.Distribution(samp_init[:, 4] * UNIT_DVEFF)

        self.samp_dict = {
            "amp_e_ra_cosdec": amp_e_ra_cosdec.to(UNIT_DVEFF),
            "amp_e_dec": amp_e_dec.to(UNIT_DVEFF),
            "amp_ps": amp_ps.to(UNIT_DVEFF),
            "amp_pc": amp_pc.to(UNIT_DVEFF),
            "dveff_c": dveff_c.to(UNIT_DVEFF),
        }


class SamplePhys(SampleBase):
    """Monte Carlo sample in physical parameters."""

    npar = 6
    pardict = pardict_phys

    def from_phen_common(self, phen):
        """Create sample in physical parameters from SamplePhen."""

        self.nmc = phen.nmc
        self.fit = phen.fit
        self.target = phen.target

        # initialize weights at unity
        self.weights = np.ones(self.nmc)

    def from_phen_and_i_p(self, phen):
        """Create sample in physical parameters from SamplePhen and inclination.

        Propagate uncertainties from phenomenological model parameters to
        physical parameters using a known pulsar orbital inclination.
        """

        self.from_phen_common(phen)

        i_p_mu = phen.target.i_p_prior_mu
        i_p_sig = phen.target.i_p_prior_sig

        # generate samples according to prior
        i_p_iter = np.random.normal(size=self.nmc) * i_p_sig + i_p_mu
        i_p_iter %= 180 * u.deg

        # replace any negative samples
        while np.any(i_p_iter <= 0):
            idx_neg = np.where(i_p_iter <= 0)
            i_p_replace = np.random.normal(size=len(idx_neg[0])) * i_p_sig + i_p_mu
            i_p_replace %= 180 * u.deg
            i_p_iter[idx_neg] = i_p_replace

        samp_i_p = unc.Distribution(i_p_iter)

        # convert to cosine of inclination
        samp_cosi_p = np.cos(samp_i_p)

        # convert phenomenological parameters to physical parameters
        self.samp_dict = pars_phen2phys_cosi_p(phen.samp_dict, self.target, samp_cosi_p)

    def from_phen_and_parallax(self, phen, cos_sign=None):
        """Create sample in physical parameters from SamplePhen and parallax.

        Propagate uncertainties from phenomenological model parameters to
        physical parameters using a known pulsar parallax.
        """

        self.from_phen_common(phen)

        # pick orientation of orbit
        cos_sign = cos_sign or np.sign(np.cos(phen.target.i_p_prior_mu))
        self.cos_sign = cos_sign

        parallax_mu = phen.target.parallax_prior_mu
        parallax_sig = phen.target.parallax_prior_sig

        # generate samples according to prior
        parallax_iter = np.random.normal(size=self.nmc) * parallax_sig + parallax_mu

        # replace any negative samples
        while np.any(parallax_iter <= 0):
            idx_neg = np.where(parallax_iter <= 0)
            parallax_replace = (
                np.random.normal(size=len(idx_neg[0])) * parallax_sig + parallax_mu
            )
            parallax_iter[idx_neg] = parallax_replace

        samp_parallax = unc.Distribution(parallax_iter)

        # convert to parallax of distance
        samp_d_p = samp_parallax.to(u.kpc, equivalencies=u.parallax())

        # convert phenomenological parameters to physical parameters
        self.samp_dict = pars_phen2phys_d_p(
            phen.samp_dict, self.target, samp_d_p, self.cos_sign
        )

    def weight_by_i_p(self):
        cosi_p = self.samp_dict["cosi_p"].distribution
        i_p = np.arccos(cosi_p)
        weights_i_p = gaussian(
            i_p.to_value(u.deg),
            self.target.i_p_prior_mu.to_value(u.deg),
            self.target.i_p_prior_sig.to_value(u.deg),
        )

        self.weights *= weights_i_p

    def weight_by_omega_p(self):
        omega_p = self.samp_dict["omega_p"].distribution
        weights_omega_p = gaussian(
            omega_p.to_value(u.deg),
            self.target.omega_p_prior_mu.to_value(u.deg),
            self.target.omega_p_prior_sig.to_value(u.deg),
        )

        self.weights *= weights_omega_p

    def weight_by_parallax(self):
        d_p = self.samp_dict["d_p"].distribution
        parallax = d_p.to(u.mas, equivalencies=u.parallax())
        weights_parallax = gaussian(
            parallax.to_value(u.mas),
            self.target.parallax_prior_mu.to_value(u.mas),
            self.target.parallax_prior_sig.to_value(u.mas),
        )

        self.weights *= weights_parallax

    def weight_for_constant_space_density(self):
        d_p = self.samp_dict["d_p"].distribution
        weights_d_p = d_p.to_value(u.kpc) ** 2

        self.weights *= weights_d_p


class SamplePres(SampleBase):
    """Monte Carlo sample in results-presenting parameters."""

    npar = 6
    pardict = pardict_pres

    def from_phys(self, phys):
        """Create sample in results-presenting parameters from SamplePhys."""

        self.nmc = phys.nmc
        self.fit = phys.fit
        self.target = phys.target
        self.weights = phys.weights

        self.samp_dict = pars_phys2pres(phys.samp_dict)
