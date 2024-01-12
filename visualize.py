import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.colors import CenteredNorm
# Note: matplotlib.colors.CenteredNorm requires matplotlib version >= 3.4.0

from astropy import units as u
from astropy import constants as const

from astropy.time import Time

from astropy.visualization import quantity_support, time_support

from .utils import PARS_FIT_LABELS, UNIT_DVEFF, UNIT_DTSQRTTAU


obs_style = {
    "linestyle": "none",
    "color": "black",
    "marker": "o",
    "markerfacecolor": "none",
}

mdl_style = {"linestyle": "-", "linewidth": 2, "color": "C0"}

fail_style = {
    "linestyle": "none",
    "color": "grey",
    "marker": "o",
}

dveff_unit = r"$\mathrm{ \left( \dfrac{ km/s }{ \sqrt{ kpc } } \right) }$"

dveff_lbl = (
    # r"scaled effective velocity "
    r"$\dfrac{ | v_\mathrm{eff,\!\!\parallel} | }{ \sqrt{ d_\mathrm{eff} } }$ "
    f"{dveff_unit}"
)

dveff_signed_lbl = (
    # r"scaled effective velocity "
    r"$\dfrac{ v_\mathrm{eff,\!\!\parallel} }{ \sqrt{ d_\mathrm{eff} } }$ "
    f"{dveff_unit}"
)

dveff_res_lbl = (
    r"residuals "
    f"{dveff_unit}"
)

dtsqrttau_unit = r"$\mathrm{ \left( \sqrt{ms} / yr \right) }$"

dtsqrttau_lbl = (
    # r"effective drift rate "
    r"$\partial_t \sqrt{ \tau }$ "
    f"{dtsqrttau_unit}"
)

dtsqrttau_res_lbl = (
    r"residual "
    r"$\partial_t \sqrt{ \tau }$ "
    f"{dtsqrttau_unit}"
)

title_kwargs = {"loc": "left", "x": 0.01, "y": 1.0, "pad": -14}


def plot_data(data):
    plt.figure(figsize=(12, 10))

    plt.subplots_adjust(wspace=0.1)

    ax0 = plt.subplot(211)

    plt.errorbar(data.t_obs.mjd, data.dveff_obs, yerr=data.dveff_err, **obs_style)

    plt.xlabel("MJD")
    plt.ylabel(dveff_lbl)

    ylim = (0, ax0.get_ylim()[1])
    plt.ylim(ylim)

    xlim_list = [
        (59299.4, 59300.4),
        (59403.1, 59404.1),
        (59497.9, 59498.9),
        (59574.6, 59575.6),
        (59647.4, 59648.4),
    ]

    for isubplot, xlims in zip(range(6, 11), xlim_list):
        ax = plt.subplot(2, 5, isubplot)

        plt.ylim(ylim)

        plt.errorbar(data.t_obs.mjd, data.dveff_obs, yerr=data.dveff_err, **obs_style)

        if isubplot == 6:
            plt.ylabel(dveff_lbl)
        else:
            plt.ylabel("")
            ax.set_yticklabels([])

        plt.xlim(xlims)

        plt.draw()

        mjd_offset = ax.get_xaxis().get_major_formatter().offset

        ax.xaxis.offsetText.set_visible(False)
        plt.xlabel(rf"$\mathrm{{MJD}} - {mjd_offset:.0f}$")

    plt.show()


def visualize_model_zoom(model, data, pars):
    quantity_support()
    time_support(format="iso")

    plt.figure(figsize=(12, 12))

    # plt.subplots_adjust(wspace=0.1, hspace=0.2)

    mdl_alpha = 0.4
    obs_alpha = 1.0
    col_zoom_lbl = "black"

    # compute model at observation times, then find residuals
    dveff_mdl = np.abs(model.get_dveff_signed_from_t(pars, data.t_obs))
    dveff_res = data.dveff_obs - dveff_mdl

    # compute model at failed observation times
    t_fail_mjd = [
        59297.03092491187,
        59426.40080007567,
        59434.73706297613,
        59464.657160493385,
        59797.675009191335,
        59943.94425367851,
    ]
    t_fail = Time(t_fail_mjd, format="mjd")
    dveff_mdl_fail = np.abs(model.get_dveff_signed_from_t(pars, t_fail))

    # --- full time series ---

    # compute model
    # tlim_long = [np.min(data.t_obs.mjd) - 20, np.max(data.t_obs.mjd) + 40]
    tlim_long = [59280, 60000]
    t_mjd_many = np.arange(tlim_long[0], tlim_long[-1], 0.1)
    t_many = Time(t_mjd_many, format="mjd")
    dveff_mdl_many = model.get_dveff_signed_from_t(pars, t_many)

    # insert zeros and take norm
    t_many, dveff_mdl_many = insert_zeros(t_many, dveff_mdl_many)
    dveff_mdl_many = np.abs(dveff_mdl_many)

    # model and data
    ax1 = plt.subplot(411)
    plt.plot(t_many, dveff_mdl_many, **mdl_style, alpha=mdl_alpha)
    plt.errorbar(
        data.t_obs.mjd,
        data.dveff_obs,
        yerr=data.dveff_err,
        **obs_style,
        alpha=obs_alpha,
    )
    plt.plot(t_fail.mjd, dveff_mdl_fail, **fail_style)

    tick_t_iso = [
        "2021-04-01",
        "2021-07-01",
        "2021-10-01",
        "2022-01-01",
        "2022-04-01",
        "2022-07-01",
        "2022-10-01",
        "2023-01-01",
    ]
    tick_t = Time(tick_t_iso, format="iso")
    ax1.xaxis.tick_top()
    ax1.set_xticks(tick_t)

    plt.xlim(tlim_long)
    plt.title("(a)   full model", **title_kwargs)
    plt.xlabel("")
    plt.ylabel(dveff_lbl)

    # save ylims
    ylim1 = (0, ax1.get_ylim()[1] * 1.1)
    plt.ylim(ylim1)

    secax = ax1.secondary_yaxis("right", functions=(dveff2dtsqrttau, dtsqrttau2dveff))
    secax.set_ylabel(dtsqrttau_lbl)

    # residuals
    ax2 = plt.subplot(412)
    plt.errorbar(
        data.t_obs.mjd, dveff_res, yerr=data.dveff_err, **obs_style, alpha=obs_alpha
    )
    plt.axhline(**mdl_style)
    plt.xlim(tlim_long)
    plt.title("(b)   residuals", **title_kwargs)
    plt.xlabel("MJD")
    plt.ylabel(dveff_res_lbl)

    # # save ylims
    ylim2 = ax2.get_ylim()
    plt.ylim(ylim2[0], ylim2[1] * 1.1)

    secax = ax2.secondary_yaxis("right", functions=(dveff2dtsqrttau, dtsqrttau2dveff))
    secax.set_ylabel(dtsqrttau_res_lbl)

    # --- zooms ---

    tlim_zoom_list = [
        59299.84 + np.array([-3.5, 0.5]),
        59403.57 + np.array([-0.5, 0.5]),
        59498.39 + np.array([-2.5, 0.5]),
        59575.09 + np.array([-0.5, 0.5]),
        59647.95 + np.array([-0.5, 0.5]),
        59732.65 + np.array([-0.5, 0.5]),
        59827.41 + np.array([-0.5, 0.5]),
        59919.22 + np.array([-0.5, 0.5]),
    ]

    for izoom, tlim_zoom in zip(np.arange(len(tlim_zoom_list)), tlim_zoom_list):
        letter = chr(ord("`") + izoom + 3)

        # add zoom label to main plot
        tmid_zoom = np.mean(tlim_zoom)
        ax1.text(
            tmid_zoom,
            # (0.3 + 0.9 * ((izoom == 4) or (izoom == 6))) * np.sqrt(1000),
            -5,
            f"{letter}",
            fontsize=15,
            color=col_zoom_lbl,
            va="top",
            ha="center",
        )

        # compute model for zoomed region
        t_mjd_zoom = np.arange(tlim_zoom[0], tlim_zoom[-1], 0.005)
        t_zoom = Time(t_mjd_zoom, format="mjd")
        dveff_mdl_zoom = model.get_dveff_signed_from_t(pars, t_zoom)

        # insert zeros and take norm
        t_zoom, dveff_mdl_many = insert_zeros(t_zoom, dveff_mdl_zoom)
        dveff_mdl_zoom = np.abs(dveff_mdl_zoom)

        if izoom == 0:
            ax3 = plt.subplot(4, 5, (11, 12))
        elif izoom == 2:
            ax3 = plt.subplot(4, 5, (14, 15))
        else:
            ax3 = plt.subplot(4, 5, izoom + 12 + (izoom > 2))

        plt.ylim(ylim1)

        if (izoom == 0) or (izoom == 3):
            plt.ylabel(dveff_lbl)
        else:
            plt.ylabel("")
            ax3.set_yticklabels([])

        secax = ax3.secondary_yaxis(
            "right", functions=(dveff2dtsqrttau, dtsqrttau2dveff)
        )

        if (izoom == 2) or (izoom == 7):
            secax.set_ylabel(dtsqrttau_lbl)
        else:
            secax.set_yticklabels([])

        plt.plot(t_zoom.mjd, dveff_mdl_zoom, **mdl_style)
        plt.errorbar(data.t_obs.mjd, data.dveff_obs, yerr=data.dveff_err, **obs_style)
        plt.plot(t_fail.mjd, dveff_mdl_fail, **fail_style)
        plt.xlim(tlim_zoom)

        caldate = Time(tlim_zoom_list[izoom][-1] - 0.5, format="mjd").iso[:10]
        plt.title(f" ({letter})   {caldate}", color=col_zoom_lbl, **title_kwargs)

        plt.draw()

        mjd_offset = ax3.get_xaxis().get_major_formatter().offset

        ax3.xaxis.offsetText.set_visible(False)
        plt.xlabel(rf"$\mathrm{{MJD}} - {mjd_offset:.0f}$")

    plt.tight_layout()
    plt.show()


def insert_zeros(x, y):
    """Insert explicit zeros at zero crossings using linear interpolation."""

    x_is_time = isinstance(x, Time)
    if x_is_time:
        x_format = x.format
        x_scale = x.scale
        x = x.mjd

    y_is_quantity = isinstance(y, u.Quantity)
    if y_is_quantity:
        y_unit = y.unit
        y = y.value

    # Find indices just before zero crossings
    i0 = np.where(np.diff(np.sign(y)))[0]

    # Find x values of zero crossings through linear interpolation
    x0 = x[i0] + np.diff(x)[i0] * np.abs(y[i0] / np.diff(y)[i0])

    # Insert zeros in y array, and the corresponding x values in the x array
    y_new = np.insert(y, i0 + 1, np.zeros_like(x0))
    x_new = np.insert(np.array(x, dtype=np.float64), i0 + 1, x0)

    if x_is_time:
        x_new = Time(x_new, format=x_format, scale=x_scale)

    if y_is_quantity:
        y_new *= y_unit

    return x_new, y_new


def dveff2dtsqrttau(dveff):
    return (dveff * UNIT_DVEFF / np.sqrt(2 * const.c)).to_value(UNIT_DTSQRTTAU)


def dtsqrttau2dveff(dtsqrttau):
    return (dtsqrttau * UNIT_DTSQRTTAU * np.sqrt(2 * const.c)).to_value(UNIT_DVEFF)


def plot_convergence(mcmc):
    """Plot autocorrelation length vs. step."""

    n = np.arange(1, mcmc.index + 1) * mcmc.conv_check_interval

    plt.plot(n, n / mcmc.ntau_goal, "--k")

    for i, y in enumerate(mcmc.autocorr.T):
        y = y[: mcmc.index]
        plt.plot(n, y, label=PARS_FIT_LABELS[i])
    plt.plot(n, np.mean(mcmc.autocorr[: mcmc.index, :], axis=1), ":k", label="mean")

    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))

    plt.xlabel("number of steps")
    plt.ylabel(r"$\hat{\tau}$")
    plt.legend()

    plt.show()


def plot_chains(mcmc):
    """Plot MCMC chains."""

    samples = mcmc.sampler.get_chain()

    fig, axes = plt.subplots(mcmc.ndim, figsize=(12, 14), sharex=True)

    for i in range(mcmc.ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(PARS_FIT_LABELS[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
