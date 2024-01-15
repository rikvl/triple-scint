import numpy as np

from astropy import units as u

from astropy.time import Time

from .utils import UNIT_DVEFF


class Data:
    """Scaled effective velocity data."""

    def __init__(self, datafilename, dveff_unit=None):
        unit_dveff_file = dveff_unit or u.km / u.s / u.pc**0.5
        corr_unit_dveff = (unit_dveff_file / UNIT_DVEFF).to(u.dimensionless_unscaled)
        self.datafilename = datafilename

        data = np.load(datafilename)

        self.t_obs = Time(data["t_mjd"], format="mjd", scale="utc")
        self.dveff_obs = data["dveff_obs"] * corr_unit_dveff * UNIT_DVEFF
        self.dveff_err_original = data["dveff_err"] * corr_unit_dveff * UNIT_DVEFF
        self.dveff_err = self.dveff_err_original

        self.tlim = [np.min(self.t_obs.mjd) - 20, np.max(self.t_obs.mjd) + 40]

        self._efac = 1
        self._equad = 0 * UNIT_DVEFF

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
