import numpy as np

from astropy import units as u

from .utils import get_v_0_e, UNIT_DVEFF


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
    sindelomg_p = np.sin(delta_omega_p)
    cosdelomg_p = np.cos(delta_omega_p)
    b2_p = cosdelomg_p**2 + sindelomg_p**2 * cosi_p**2

    # amplitude and phase of pulsar signal
    sini_p = np.sqrt(1 - cosi_p**2)
    amp_p = np.sqrt(d_eff) / d_p * k_p / sini_p * np.sqrt(b2_p)
    chi_p = np.arctan2(sindelomg_p * cosi_p, cosdelomg_p) % (360 * u.deg)

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
    amp_e = get_v_0_e() / np.sqrt(d_eff)
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
    d_eff = (get_v_0_e() / amp_e) ** 2

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

    pars_phys = {
        "cosi_p": cosi_p.to(u.dimensionless_unscaled),
        "omega_p": omega_p.to(u.deg),
        "d_p": d_p.to(u.kpc),
        "s": s.to(u.dimensionless_unscaled),
        "xi": xi.to(u.deg),
        "v_lens": v_lens.to(u.km / u.s),
    }

    return pars_phys


def pars_phen2phys_cosi_p(pars_phen, target, cosi_p):
    """Convert phenomenological model parameters to physical parameters
    when cosine of pulsar's orbital inclination is known."""

    k_p = target.k_i

    pars_comm = pars_phen2comm(pars_phen)

    d_eff = pars_comm["d_eff"]
    xi = pars_comm["xi"]
    amp_p = pars_comm["amp_p"]
    chi_p = pars_comm["chi_p"]

    # pulsar longitude of ascending node
    delta_omega_p = np.arctan2(np.sin(chi_p) / cosi_p, np.cos(chi_p))
    omega_p = (xi - delta_omega_p) % (360 * u.deg)

    # pulsar distance
    sini_p = np.sqrt(1 - cosi_p**2)
    b_p = np.sqrt((1 - sini_p**2) / (1 - sini_p**2 * np.cos(chi_p) ** 2))
    d_p = np.sqrt(d_eff) / amp_p * k_p / sini_p * b_p

    # fractional pulsar-screen distance
    s = d_p / (d_p + d_eff)

    # screen velocity
    v_lens = pars_comm2vlens(pars_comm, s, target)

    pars_phys = {
        "cosi_p": cosi_p.to(u.dimensionless_unscaled),
        "omega_p": omega_p.to(u.deg),
        "d_p": d_p.to(u.kpc),
        "s": s.to(u.dimensionless_unscaled),
        "xi": xi.to(u.deg),
        "v_lens": v_lens.to(u.km / u.s),
    }

    return pars_phys
