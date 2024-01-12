import numpy as np

from scipy.special import erf

from astropy import units as u

UNIT_DVEFF = u.km / u.s / u.kpc**0.5
UNIT_DTSQRTTAU = u.us**0.5 / u.yr

PARS_FIT_LABELS = [
    r"$\cos(i_\mathrm{p})$",
    r"$\Omega_\mathrm{p}$",
    r"$d_\mathrm{p}$",
    r"$s$",
    r"$\xi$",
    r"$v_\mathrm{lens,\parallel}$",
]

PARS_MDL_LABELS = PARS_FIT_LABELS.copy()
PARS_MDL_LABELS[0] = r"$i_\mathrm{p}$"

PARS_UNIT_STRS = [" (deg)", " (deg)", " (kpc)", "", " (deg)", " (km/s)"]
PARS_LABELS_UNITS = [
    label + unit_str for label, unit_str in zip(PARS_MDL_LABELS, PARS_UNIT_STRS)
]


def get_v_0_e():
    """Get measure of orbital speed for Earth."""

    p_orb_e = 365.256363004 * u.day
    a_e = 1.0000010178 * u.au
    ecc_e = 0.0167086

    v_0_e = 2 * np.pi * a_e / (p_orb_e * np.sqrt(1 - ecc_e**2))

    return v_0_e


def th2ma(th, ecc):
    """Get mean anomaly from true anomaly."""

    common = 1 + ecc * np.cos(th)

    sin_ea = np.sqrt(1 - ecc**2) * np.sin(th) / common
    cos_ea = (ecc + np.cos(th)) / common

    ma = np.arctan2(sin_ea, cos_ea) - ecc * sin_ea * u.rad

    return ma


def gaussian(x, mu, sig):
    return 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def get_16_50_84(sigma_level=1):
    quantiles = 0.5 + np.array([-1, 0, 1]) * erf(sigma_level / np.sqrt(2)) / 2
    percentiles = quantiles * 100
    return percentiles
