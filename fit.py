import numpy as np

from astropy import units as u

from astropy.time import Time

from scipy.optimize import curve_fit, minimize
from scipy.stats import kstest

import emcee

from IPython.display import display, Math

from .freeparams import guess_pars_phen, guess_pars_phys
from .utils import tex_uncertainties, UNIT_DVEFF, PARS_MDL_LABELS, PARS_UNIT_STRS


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

    def get_residuals(self, pars_mdl):
        dveff_mdl = self.model.model_dveff_abs(
            pars_mdl,
            self.sin_term_i_obs,
            self.cos_term_i_obs,
            self.sin_term_o_obs,
            self.cos_term_o_obs,
            self.scaled_v_earth_xyz_obs,
        )

        dveff_res = self.data.dveff_obs - dveff_mdl

        return dveff_res

    def get_chi2(self, pars_mdl):
        dveff_res = self.get_residuals(pars_mdl)
        chi2 = np.sum((dveff_res / self.data.dveff_err) ** 2)
        return chi2

    def print_result(self, pars_mdl):
        """Quick view of parameter values and reduced chi^2."""

        for par_name in pars_mdl:
            print(f"{par_name:8s} {pars_mdl[par_name]:8.2f}")

        if "cosi_p" in pars_mdl:
            i_p = np.arccos(pars_mdl["cosi_p"]) / np.pi * 180 * u.deg
            print(f"\ni_p      {i_p:8.2f}")

        print(f"\nchi2red  {self.get_chi2(pars_mdl)/self.ndof:8.2f}")

        print(f"\nefac     {self.data.efac:8.2f}")
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

    def corr_errors(self, ef_grid=None, eq_grid=None):
        """Try grid of EFAC and EQUAD."""

        # create grid of EFAC and EQUAD values
        ef_grid = ef_grid if ef_grid is not None else np.linspace(0.5, 2.5, 21)
        eq_grid = (
            eq_grid if eq_grid is not None else (np.linspace(0, 3, 31) * UNIT_DVEFF)
        )

        nef = len(ef_grid)
        neq = len(eq_grid)

        ks_grid = np.zeros((nef, neq))

        print("efac, percent done")

        for ief, ef in enumerate(ef_grid):
            self.data.efac = ef

            print(f"{ef:4.1f}, {ief / nef * 100:4.1f}")

            for ieq, eq in enumerate(eq_grid):
                self.data.equad = eq

                # fit model with new EFAC and EQUAD values
                pars_mdl = self.optimize(pars_mdl_init=None)

                # test if scaled residuals follow normal distribution using KS test
                dveff_res = self.get_residuals(pars_mdl)
                scaled_errors = dveff_res / self.data.dveff_err
                test_dist = scaled_errors.to_value(u.dimensionless_unscaled)
                ksres = kstest(test_dist, "norm")

                ks_grid[ief, ieq] = ksres.statistic

        print("     100.0\n")

        # find minimum KS statistic
        ind_min_ks = np.unravel_index(np.argmin(ks_grid, axis=None), ks_grid.shape)
        ef_opt = ef_grid[ind_min_ks[0]]
        eq_opt = eq_grid[ind_min_ks[1]]

        self.data.efac = ef_opt
        self.data.equad = eq_opt

        pars_mdl = self.optimize(pars_mdl_init=None)

        return pars_mdl


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

        # find optimal phenomenological parameters
        popt, pcov = curve_fit(
            f=model_dveff_fit,
            xdata=indep_vars,
            ydata=self.data.dveff_obs.to_value(UNIT_DVEFF),
            p0=pars_fit_init,
            sigma=self.data.dveff_err.to_value(UNIT_DVEFF),
            absolute_sigma=True,
        )

        # find sign for which convention 0 <= xi < 180 deg holds, which is equal
        # to the sign of sin( xi ), which is equal to the sign of amp_e_ra_cosdec,
        # which is the index 0 entry in the vector of parameters
        self.sol_sign = np.sign(popt[0])

        # set optimum solution to have correct sign
        popt *= self.sol_sign

        # store curve_fit solution and covariance matrix
        self.curve_fit_popt = popt
        self.curve_fit_pcov = pcov

        # convert to dict of astropy quantities
        pars_mdl = self.model.pars_fit2mdl(popt)

        # store optimum solution
        self.pars_opt = pars_mdl

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

        self.pars_opt = pars_mdl

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
            result = tex_uncertainties(self.scaled_samples[:, i])
            txt = (
                f"{PARS_MDL_LABELS[i][1:-1]} = {result} "
                f"\\; \\mathrm{{ {PARS_UNIT_STRS[i][2:-1]} }}"
            )
            display(Math(txt))
