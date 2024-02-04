from collections import namedtuple

import numpy as np

from astropy import units as u

from .utils import v_0_e, UNIT_DVEFF

unit_str_dveff = r"\mathrm{ km/s / \sqrt{kpc} }"

Param = namedtuple("ParString", ["symbol", "unit_str", "unit_ap"])

# phenomenological parameters
par_amp_e_ra_cosdec = Param(r"A_{\oplus,\alpha\ast}", unit_str_dveff, UNIT_DVEFF)
par_amp_e_dec = Param(r"A_{\oplus,\delta}", unit_str_dveff, UNIT_DVEFF)
par_amp_ps = Param(r"A_{p,s}", unit_str_dveff, UNIT_DVEFF)
par_amp_pc = Param(r"A_{p,s}", unit_str_dveff, UNIT_DVEFF)
par_dveff_c = Param(r"C", unit_str_dveff, UNIT_DVEFF)
pardict_phen = {
    "amp_e_ra_cosdec": par_amp_e_ra_cosdec,
    "amp_e_dec": par_amp_e_dec,
    "amp_ps": par_amp_ps,
    "amp_pc": par_amp_pc,
    "dveff_c": par_dveff_c,
}

# physical parameters
par_cosi_p = Param(r"\cos( i_\mathrm{p} )", r"", u.dimensionless_unscaled)
par_omega_p = Param(r"\Omega_\mathrm{p}", r"\mathrm{deg}", u.deg)
par_d_p = Param(r"d_\mathrm{p}", r"\mathrm{kpc}", u.kpc)
par_s = Param(r"s", r"", u.dimensionless_unscaled)
par_xi = Param(r"\xi", r"\mathrm{deg}", u.deg)
par_v_lens = Param(r"v_\mathrm{lens,\parallel}", r"\mathrm{km/s}", u.km / u.s)
pardict_phys = {
    "cosi_p": par_cosi_p,
    "omega_p": par_omega_p,
    "d_p": par_d_p,
    "s": par_s,
    "xi": par_xi,
    "v_lens": par_v_lens,
}

# intermediate / common parameters
par_d_eff = Param(r"d_\mathrm{eff}", r"\mathrm{kpc}", u.kpc)
par_amp_p = Param(r"A_{p}", unit_str_dveff, UNIT_DVEFF)
par_chi_p = Param(r"\chi_\mathrm{p}", r"\mathrm{deg}", u.deg)
pardict_comm = {
    "d_eff": par_d_eff,
    "xi": par_xi,
    "amp_p": par_amp_p,
    "chi_p": par_chi_p,
    "dveff_c": par_dveff_c,
}

# results-presenting parameters
par_i_p = Param(r"i_\mathrm{p}", r"\mathrm{deg}", u.deg)
par_d_s = Param(r"d_\mathrm{s}", r"\mathrm{kpc}", u.kpc)
pardict_pres = {
    "i_p": par_i_p,
    "omega_p": par_omega_p,
    "d_p": par_d_p,
    # "d_s": par_d_s,
    "s": par_s,
    "xi": par_xi,
    "v_lens": par_v_lens,
}


def guess_pars_phen(target):
    pars_phys = guess_pars_phys(target)
    pars_phen = pars_phys2phen(pars_phys, target)
    return pars_phen


def guess_pars_phys(target):
    pars_phys = {
        "cosi_p": np.cos(target.i_p_prior_mu),
        "omega_p": target.omega_p_prior_mu,
        "d_p": target.parallax_prior_mu.to(u.kpc, equivalencies=u.parallax()),
        "s": 0.8 * u.dimensionless_unscaled,
        "xi": 62.0 * u.deg,
        "v_lens": 4.0 * u.km / u.s,
    }

    return pars_phys


def pars_phys2phen(pars_phys, target):
    """Convert physical parameters to phenomenological model parameters."""

    k_p = target.k_i
    k_ratio = target.k_o / target.k_i
    psr_coord = target.psr_coord
    ecc_i = target.ecc_i
    ecc_o = target.ecc_o
    arg_per_i = target.arg_per_i
    arg_per_o = target.arg_per_o

    cosi_p = pars_phys["cosi_p"]
    omega_p = pars_phys["omega_p"]
    d_p = pars_phys["d_p"]
    s = pars_phys["s"]
    xi = pars_phys["xi"]
    v_lens = pars_phys["v_lens"]

    # effective distance
    d_eff = (1 - s) / s * d_p

    delta_omega_p = xi - omega_p
    sindelta_omega_p = np.sin(delta_omega_p)
    cosdelta_omega_p = np.cos(delta_omega_p)
    b2_p = cosdelta_omega_p**2 + sindelta_omega_p**2 * cosi_p**2

    # amplitude and phase of pulsar signal
    sini_p = np.sqrt(1 - cosi_p**2)
    amp_p = np.sqrt(d_eff) / d_p * k_p / sini_p * np.sqrt(b2_p)
    chi_p = np.arctan2(sindelta_omega_p * cosi_p, cosdelta_omega_p) % (360 * u.deg)

    # constant in scintillometric signal
    mu_p_sys = psr_coord.pm_ra_cosdec * np.sin(xi) + psr_coord.pm_dec * np.cos(xi)
    v_p_sys = (d_p * mu_p_sys).to(u.km / u.s, equivalencies=u.dimensionless_angles())
    dveff_c = (
        1 / s * v_lens / np.sqrt(d_eff)
        - (1 - s) / s * v_p_sys / np.sqrt(d_eff)
        + amp_p * ecc_i * np.sin(arg_per_i - chi_p)
        + amp_p * ecc_o * np.sin(arg_per_o - chi_p) * k_ratio
    )

    # factors for earth signal
    amp_e = v_0_e / np.sqrt(d_eff)
    amp_e_ra_cosdec = amp_e * np.sin(xi)
    amp_e_dec = amp_e * np.cos(xi)

    # amplitudes for pulsar signal
    amp_ps = amp_p * np.cos(chi_p)
    amp_pc = amp_p * np.sin(chi_p)

    pars_phen = {
        "amp_e_ra_cosdec": amp_e_ra_cosdec.to(UNIT_DVEFF),
        "amp_e_dec": amp_e_dec.to(UNIT_DVEFF),
        "amp_ps": amp_ps.to(UNIT_DVEFF),
        "amp_pc": amp_pc.to(UNIT_DVEFF),
        "dveff_c": dveff_c.to(UNIT_DVEFF),
    }

    return pars_phen


def pars_phen2comm(pars_phen):
    """Convert phenomenological model parameters to intermediate parameters."""

    amp_e_ra_cosdec = pars_phen["amp_e_ra_cosdec"]
    amp_e_dec = pars_phen["amp_e_dec"]
    amp_ps = pars_phen["amp_ps"]
    amp_pc = pars_phen["amp_pc"]
    dveff_c = pars_phen["dveff_c"]

    # effective distance
    amp_e = np.sqrt(amp_e_ra_cosdec**2 + amp_e_dec**2)
    d_eff = (v_0_e / amp_e) ** 2

    # screen angle
    xi = np.arctan2(amp_e_ra_cosdec, amp_e_dec) % (360 * u.deg)

    # amplitude and phase of pulsar signal
    amp_p = np.sqrt(amp_ps**2 + amp_pc**2)
    chi_p = np.arctan2(amp_pc, amp_ps) % (360 * u.deg)

    pars_comm = {
        "d_eff": d_eff.to(u.kpc),
        "xi": xi.to(u.deg),
        "amp_p": amp_p.to(UNIT_DVEFF),
        "chi_p": chi_p.to(u.deg),
        "dveff_c": dveff_c.to(UNIT_DVEFF),
    }

    return pars_comm


def pars_cosip2dp(pars_comm, cosi_p, target):
    """Convert cos(i_p) to d_p using intermediate parameters."""

    k_p = target.k_i

    d_eff = pars_comm["d_eff"]
    amp_p = pars_comm["amp_p"]
    chi_p = pars_comm["chi_p"]

    sini_p = np.sqrt(1 - cosi_p**2)
    b_p = np.sqrt((1 - sini_p**2) / (1 - sini_p**2 * np.cos(chi_p) ** 2))
    d_p = np.sqrt(d_eff) / amp_p * k_p / sini_p * b_p

    return d_p


def pars_comm2vlens(pars_comm, s, target):
    """Extract lens velocity from intermediate parameters."""

    k_ratio = target.k_o / target.k_i
    psr_coord = target.psr_coord
    ecc_i = target.ecc_i
    ecc_o = target.ecc_o
    arg_per_i = target.arg_per_i
    arg_per_o = target.arg_per_o

    d_eff = pars_comm["d_eff"]
    xi = pars_comm["xi"]
    amp_p = pars_comm["amp_p"]
    chi_p = pars_comm["chi_p"]
    dveff_c = pars_comm["dveff_c"]

    mu_p_sys = psr_coord.pm_ra_cosdec * np.sin(xi) + psr_coord.pm_dec * np.cos(xi)
    v_eff_p_sys = (d_eff * mu_p_sys).to(
        u.km / u.s, equivalencies=u.dimensionless_angles()
    )
    dveff_ecc_term = (
        amp_p * ecc_i * np.sin(arg_per_i - chi_p)
        + amp_p * ecc_o * np.sin(arg_per_o - chi_p) * k_ratio
    )
    v_lens = s * (v_eff_p_sys + np.sqrt(d_eff) * (dveff_c - dveff_ecc_term))

    return v_lens.to(u.km / u.s)


def pars_phen2phys_d_p(pars_phen, target, d_p, cos_sign):
    """Convert phenomenological model parameters to physical parameters
    when pulsar distance is known."""

    k_p = target.k_i

    pars_comm = pars_phen2comm(pars_phen)

    d_eff = pars_comm["d_eff"]
    xi = pars_comm["xi"]
    amp_p = pars_comm["amp_p"]
    chi_p = pars_comm["chi_p"]

    # pulsar orbital inclination
    z2 = d_eff * (k_p / (amp_p * d_p)) ** 2
    cos2chi_p = np.cos(chi_p) ** 2
    discrim = (1 + z2) ** 2 - 4 * cos2chi_p * z2
    sin2i_p = 2 * z2 / (1 + z2 + np.sqrt(discrim))
    cosi_p = cos_sign * np.sqrt(1 - sin2i_p)

    # pulsar longitude of ascending node
    delta_omega_p = np.arctan2(np.sin(chi_p) / cosi_p, np.cos(chi_p))
    omega_p = (xi - delta_omega_p) % (360 * u.deg)

    # fractional pulsar-screen distance
    s = d_p / (d_p + d_eff)

    # screen velocity
    v_lens = pars_comm2vlens(pars_comm, s, target)

    pars_phys = gather_pars_phys(cosi_p, omega_p, d_p, s, xi, v_lens)

    return pars_phys


def pars_phen2phys_cosi_p(pars_phen, target, cosi_p):
    """Convert phenomenological model parameters to physical parameters
    when cosine of pulsar's orbital inclination is known."""

    pars_comm = pars_phen2comm(pars_phen)

    d_eff = pars_comm["d_eff"]
    xi = pars_comm["xi"]
    chi_p = pars_comm["chi_p"]

    # pulsar longitude of ascending node
    delta_omega_p = np.arctan2(np.sin(chi_p) / cosi_p, np.cos(chi_p))
    omega_p = (xi - delta_omega_p) % (360 * u.deg)

    # pulsar distance
    d_p = pars_cosip2dp(pars_comm, cosi_p, target)

    # fractional pulsar-screen distance
    s = d_p / (d_p + d_eff)

    # screen velocity
    v_lens = pars_comm2vlens(pars_comm, s, target)

    pars_phys = gather_pars_phys(cosi_p, omega_p, d_p, s, xi, v_lens)

    return pars_phys


def pars_phen2phys_omega_p(pars_phen, target, omega_p):
    """Convert phenomenological model parameters to physical parameters
    when cosine of pulsar's orbital inclination is known."""

    pars_comm = pars_phen2comm(pars_phen)

    d_eff = pars_comm["d_eff"]
    xi = pars_comm["xi"]
    chi_p = pars_comm["chi_p"]

    # cosine of pulsar's orbital inclination
    delta_omega_p = xi - omega_p
    cosi_p = np.tan(chi_p) / np.tan(delta_omega_p)

    # pulsar distance
    d_p = pars_cosip2dp(pars_comm, cosi_p, target)

    # fractional pulsar-screen distance
    s = d_p / (d_p + d_eff)

    # screen velocity
    v_lens = pars_comm2vlens(pars_comm, s, target)

    pars_phys = gather_pars_phys(cosi_p, omega_p, d_p, s, xi, v_lens)

    return pars_phys


def gather_pars_phys(cosi_p, omega_p, d_p, s, xi, v_lens):
    """Gather physical parameters into dict with standard units."""

    pars_phys = {
        "cosi_p": cosi_p.to(u.dimensionless_unscaled),
        "omega_p": omega_p.to(u.deg),
        "d_p": d_p.to(u.kpc),
        "s": s.to(u.dimensionless_unscaled),
        "xi": xi.to(u.deg),
        "v_lens": v_lens.to(u.km / u.s),
    }

    return pars_phys


def pars_phys2pres(pars_phys):
    """Convert physical parameters to results-presenting parameters,"""

    cosi_p = pars_phys["cosi_p"]
    omega_p = pars_phys["omega_p"]
    d_p = pars_phys["d_p"]
    s = pars_phys["s"]
    xi = pars_phys["xi"]
    v_lens = pars_phys["v_lens"]

    i_p = np.arccos(cosi_p)
    # d_s = (1 - s) * d_p

    pars_pres = {
        "i_p": i_p.to(u.deg),
        "omega_p": omega_p.to(u.deg),
        "d_p": d_p.to(u.kpc),
        # "d_s": d_s.to(u.kpc),
        "s": s.to(u.dimensionless_unscaled),
        "xi": xi.to(u.deg),
        "v_lens": v_lens.to(u.km / u.s),
    }

    return pars_pres
