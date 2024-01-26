import numpy as np

from astropy import units as u

from astropy.time import Time

from .utils import UNIT_DVEFF


t_dud_mjd = [
    59297.03092491187,
    59426.40080007567,
    59434.73706297613,
    59464.657160493385,
    59797.675009191335,
    59943.94425367851,
    60055.72824764992,
]


class Data:
    """Scaled effective velocity data."""

    def __init__(
        self,
        datafilename,
        t_dud_mjd=t_dud_mjd,
        dveff_unit=None,
        t_lim_mjd=None,
        t_margin_days=20,
    ):
        unit_dveff_file = dveff_unit or u.km / u.s / u.pc**0.5
        corr_unit_dveff = (unit_dveff_file / UNIT_DVEFF).to(u.dimensionless_unscaled)
        self.datafilename = datafilename

        data = np.load(datafilename)

        self.t_obs = Time(data["t_mjd"], format="mjd", scale="utc")
        self.dveff_obs = data["dveff_obs"] * corr_unit_dveff * UNIT_DVEFF
        self.dveff_err_original = data["dveff_err"] * corr_unit_dveff * UNIT_DVEFF
        self.dveff_err = self.dveff_err_original

        self.t_dud = Time(t_dud_mjd, format="mjd")

        # Apply manual time limit or calculate time limits from data
        if t_lim_mjd:
            self.apply_t_lim(t_lim_mjd)
        else:
            t_min_mjd = np.min((self.t_obs.mjd.min(), self.t_dud.mjd.min()))
            t_max_mjd = np.max((self.t_obs.mjd.max(), self.t_dud.mjd.max()))
            t_lim_mjd = [t_min_mjd - t_margin_days, t_max_mjd + t_margin_days]
        self.t_lim_mjd = t_lim_mjd

        self._efac = 1
        self._equad = 0 * UNIT_DVEFF

    def apply_t_lim(self, t_lim_mjd):
        """Remove data points outside time limits."""

        mask_obs = (self.t_obs.mjd > t_lim_mjd[0]) & (self.t_obs.mjd < t_lim_mjd[1])
        mask_dud = (self.t_dud.mjd > t_lim_mjd[0]) & (self.t_dud.mjd < t_lim_mjd[1])

        self.t_obs = self.t_obs[mask_obs]
        self.dveff_obs = self.dveff_obs[mask_obs]
        self.dveff_err_original = self.dveff_err_original[mask_obs]
        self.dveff_err = self.dveff_err_original

        self.t_dud = self.t_dud[mask_dud]

    @property
    def efac(self):
        return self._efac

    @efac.setter
    def efac(self, new_efac):
        self._efac = new_efac
        self.dveff_err = np.sqrt(
            (self._efac * self.dveff_err_original) ** 2 + self._equad**2
        )

    @property
    def equad(self):
        return self._equad

    @equad.setter
    def equad(self, new_equad):
        self._equad = new_equad
        self.dveff_err = np.sqrt(
            (self._efac * self.dveff_err_original) ** 2 + self._equad**2
        )
