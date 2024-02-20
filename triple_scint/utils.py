import numpy as np

from scipy.special import erf

from astropy import units as u

SIDEREAL_YEAR = 365.256363004 * u.day

UNIT_DVEFF = u.km / u.s / u.kpc**0.5
UNIT_DTSQRTTAU = u.us**0.5 / u.yr

# Earth's orbital shape parameters
p_orb_e = SIDEREAL_YEAR
a_e = 1.0000010178 * u.au
ecc_e = 0.0167086

# Earth's midrange orbital speed
v_0_e = (2 * np.pi * a_e / p_orb_e / np.sqrt(1 - ecc_e**2)).to(u.km / u.s)


def th2ma(th, ecc):
    """Get mean anomaly from true anomaly."""

    common = 1 + ecc * np.cos(th)

    sin_ea = np.sqrt(1 - ecc**2) * np.sin(th) / common
    cos_ea = (ecc + np.cos(th)) / common

    ma = np.arctan2(sin_ea, cos_ea) - ecc * sin_ea * u.rad

    return ma


def gaussian(x, mu, sig):
    return 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


# from https://stackoverflow.com/a/29677616
def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def get_sigma_quantiles(sigma_level=1):
    quantiles = 0.5 + np.array([-1, 0, 1]) * erf(sigma_level / np.sqrt(2)) / 2
    return quantiles


def get_format(floats):
    characteristic = np.floor(np.log10(floats)).astype(int)
    decimals = np.maximum(-characteristic + 1, 0).min()
    fmt = f"{{0:.{decimals}f}}".format
    return fmt


def tex_uncertainties(values, weights=None):
    q_16, q_50, q_84 = weighted_quantile(values, get_sigma_quantiles(), weights)
    q_m, q_p = q_50 - q_16, q_84 - q_50
    fmt = get_format([q_m, q_p])
    if fmt(q_m) == fmt(q_p):
        txt = f"{fmt(q_50)} \\pm {fmt(q_m)}"
    else:
        txt = f"{fmt(q_50)}_{{-{fmt(q_m)}}}^{{+{fmt(q_p)}}}"
    return txt
