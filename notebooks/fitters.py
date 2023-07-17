"""Provides fitting functions or classes for performing fits."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np
import scipy.optimize
import scipy.stats

import plotters


class LinearFitter:
    """Class for performing fit by a linear function.

    For now requires statistical errors for all fitted points.
    """

    class Results:
        """Structure to store fit's results."""

        c0_val: float
        c0_err: float
        c1_val: float
        c1_err: float

        def __init__(self, popt, pcov) -> None:
            self.c0_val = popt[0]
            self.c0_err = np.sqrt(np.diag(pcov))[0]
            self.c1_val = popt[1]
            self.c1_err = np.sqrt(np.diag(pcov))[1]

    @dataclass
    class LegacyResults:
        """Structure to store fit results of the legacy fitting function."""

        a: float  # shift
        a_error: float
        b: float  # slope
        b_error: float
        chi: float
        chi_dof: float
        dof: int
        p_score: float

    @staticmethod
    def perform_fit(x: Sequence, y: Sequence,
                    sigma: Sequence) -> LegacyResults:
        """Returns results of the fit as LegacyResults struct.

        Sequences x, y, sigma are assumed to have the same length.

        Args:
            x: Sequence of x-parameter values.
            y: Sequence of fitted function's values.
            sigma: Sequence of errors of fitted function's values.

        Returns:
            LegacyResults structure containing fit's results.
        """
        data_len = len(x)
        if len(y) != len(x) or len(sigma) != len(x):
            raise ValueError(f"Arguments have to be arrays of the same size "
                             f"(received len(x)={len(x)}, len(y)={len(y)}, "
                             f"len(sigma)={len(sigma)}).")
        x = np.array(x)
        y = np.array(y)
        sigma = np.array(sigma)

        one_sigma_sqr = np.sum(np.ones(data_len) / sigma**2)
        x_sqr_sigma_sqr = np.sum(x**2 / sigma**2)
        x_sigma_sqr = np.sum(x / sigma**2)
        xy_sigma_sqr = np.sum(x * y / sigma**2)
        y_sigma_sqr = np.sum(y / sigma**2)

        delta = one_sigma_sqr * x_sqr_sigma_sqr - x_sigma_sqr**2

        a = (x_sqr_sigma_sqr * y_sigma_sqr
             - x_sigma_sqr * xy_sigma_sqr) / delta
        b = (one_sigma_sqr * xy_sigma_sqr
             - x_sigma_sqr * y_sigma_sqr) / delta

        a_error = np.sqrt(x_sqr_sigma_sqr / delta)
        b_error = np.sqrt(one_sigma_sqr / delta)

        chi = float(np.sum((y - a - b * x)**2 / sigma**2))
        chi_dof = chi / (data_len - 2)

        return LinearFitter.LegacyResults(
            a, a_error, b, b_error, chi, chi_dof, data_len - 2,
            1.0 - scipy.stats.chi2.cdf(chi, data_len - 2))

    @classmethod
    def fit_pointrep(cls, rep: plotters.PointRep) -> LegacyResults:
        """Version for fitting a PointRep.

        Args:
            rep: plotters.PointRep with values to fit.

        Returns:
            LegacyResults structure containing fit's results.
        """
        return cls.perform_fit(rep.x_val, rep.y_val, rep.y_err)

    @staticmethod
    def perform_fit_new(x: Sequence, y: Sequence, sigma: Sequence) -> Results:
        """Updated version of perform_fit.

        Sequences x, y, sigma are assumed to have the same length.

        Args:
            x: Sequence of x-parameter values.
            y: Sequence of fitted function's values.
            sigma: Sequence of errors of fitted function's values.

        Returns:
            Results structure containing fit's results.
        """
        if len(y) != len(x) or len(sigma) != len(x):
            raise ValueError(f"Arguments have to be arrays of the same size "
                             f"(received len(x)={len(x)}, len(y)={len(y)}, "
                             f"len(sigma)={len(sigma)}).")
        return LinearFitter.Results(*scipy.optimize.curve_fit(
            (lambda z, c0, c1: c0 + c1 * z),
            np.array(x), np.array(y), sigma=np.array(sigma),
            absolute_sigma=True))
    
    @staticmethod
    def perform_fit_new_combined(x: Sequence, y: Sequence, sigmas: Sequence) -> Results:
        """Returns combined (statistical and systematic) errors. 

        Sequences x, y, sigma are assumed to have the same length.

        Args:
            x: Sequence of x-parameter values.
            y: Sequence of fitted function's values.
            sigma: Sequence of errors of fitted function's values.

        Returns:
            Results structure containing fit's results.
        """
        if len(y) != len(x) or len(sigmas) != len(x):
            raise ValueError(f"Arguments have to be arrays of the same size "
                             f"(received len(x)={len(x)}, len(y)={len(y)}, "
                             f"len(sigma)={len(sigmas)}).")
        x = np.array(x)
        y = np.array(y)
        sigmas = np.array(sigmas)
        dof = len(x) - 2
        min_sigma = sigmas[0]
        for sigma in sigmas:
            if sigma < min_sigma:
                min_sigma = sigma
        counter = 0
        while counter < 10000:
            fit_results = LinearFitter.Results(*scipy.optimize.curve_fit(
            (lambda z, c0, c1: c0 + c1 * z),
            x, y, sigma=sigmas,
            absolute_sigma=True))
            chi2 = np.sum((y - fit_results.c0_val - fit_results.c1_val*x)**2 / sigmas**2)
            if chi2 > dof:
                sigmas += min_sigma * 0.02
                min_sigma *= 1.02
                if counter > 9000:
                    raise RuntimeError("Max number of iterations exceeded.")
            else:
                break
        return fit_results


class BilinearFitter:
    """Class for performing fit by a bilinear function.

    The assumed function is (a + b1*x1 + b2*x2).

    For now requires statistical errors for all fitted points.

    Two sets of linearly dependent data are assumed. (With different slopes.)
    """

    class Results:
        """Structure to store fit results."""

        c_val: float
        c_err: float
        slope1_val: float
        slope1_err: float
        slope2_val: float
        slope2_err: float

        def __init__(self, popt, pcov) -> None:
            self.c_val = popt[0]
            self.c_err = np.sqrt(np.diag(pcov))[0]
            self.slope1_val = popt[1]
            self.slope1_err = np.sqrt(np.diag(pcov))[1]
            self.slope2_val = popt[2]
            self.slope2_err = np.sqrt(np.diag(pcov))[2]

    @staticmethod
    def perform_fit(x1: Sequence, y1: Sequence, sigma1: Sequence,
                    x2: Sequence, y2: Sequence, sigma2: Sequence) -> Results:
        """Performs combined (bi)linear fit with two sets of data.

        The fitted function is (a + b1*x1 + b2*x2).

        Sequences (x1, y1, sigma1) are assumed to have the same length.
        The same is assumed for (x2, y2, sigma2)

        Args:
            x1: Sequence of x-parameter values for the 1st set.
            y1: Sequence of fitted function's values for the 1st set.
            sigma1: Sequence of errors of fitted function's values
              for the 1st set.
            x2: Sequence of x-parameter values for the 2nd set.
            y2: Sequence of fitted function's values for the 2nd set.
            sigma2: Sequence of errors of fitted function's values
              for the 2nd set.

        Returns:
            Results structure containing fit's results.
        """
        if len(y1) != len(x1) or len(sigma1) != len(x1):
            raise ValueError(f"Arguments (x1, y1, sigma1) "
                             f"have to be arrays of the same size "
                             f"(received len(x1)={len(x1)}, len(y1)={len(y1)}, "
                             f"len(sigma1)={len(sigma1)}).")
        if len(y2) != len(x2) or len(sigma2) != len(x2):
            raise ValueError(f"Arguments (x2, y2, sigma2) "
                             f"have to be arrays of the same size "
                             f"(received len(x2)={len(x2)}, len(y2)={len(y2)}, "
                             f"len(sigma2)={len(sigma2)}).")

        def bilinear_function(some_tuple: Tuple, c, k1, k2):
            x_1, x_2 = some_tuple
            return c + k1*x_1 + k2*x_2

        _x_val = (np.array([i for i in x1] + [0.0 for _ in x2]),
                  np.array([0.0 for _ in x1] + [i for i in x2]))
        y_data = np.array([i for i in y1] + [i for i in y2])
        y_err = np.array([i for i in sigma1] + [i for i in sigma2])
        return BilinearFitter.Results(*scipy.optimize.curve_fit(
            bilinear_function, _x_val, y_data, sigma=y_err,
            absolute_sigma=True))


class QuadraticFitter:
    """Class for performing fit by a quadratic function.

    For now requires statistical errors for all fitted points.
    """

    class Results:
        """Structure to store fit's results."""

        def __init__(self, popt, pcov) -> None:
            self.c0_val = popt[0]
            self.c0_err = np.sqrt(np.diag(pcov))[0]
            self.c1_val = popt[1]
            self.c1_err = np.sqrt(np.diag(pcov))[1]
            self.c2_val = popt[2]
            self.c2_err = np.sqrt(np.diag(pcov))[2]
            self.c1c2_corr = pcov[1][2] / self.c1_err / self.c2_err

    @staticmethod
    def perform_fit(x: Sequence, y: Sequence, sigma: Sequence) -> Results:
        """Returns results of a fit.

        Sequences x, y, sigma are assumed to have the same length.

        Args:
            x: Sequence of x-parameter values.
            y: Sequence of fitted function's values.
            sigma: Sequence of errors of fitted function's values.

        Returns:
            Results structure containing fit's results.
        """
        if len(y) != len(x) or len(sigma) != len(x):
            raise ValueError(f"Arguments have to be arrays of the same size "
                             f"(received len(x)={len(x)}, len(y)={len(y)}, "
                             f"len(sigma)={len(sigma)}).")
        return QuadraticFitter.Results(*scipy.optimize.curve_fit(
            (lambda z, c0, c1, c2: c0 + c1*z + c2*z*z),
            np.array(x), np.array(y), sigma=np.array(sigma),
            absolute_sigma=True))

    def fit_pointrep(self, rep: plotters.PointRep) -> Results:
        """Version for fitting a PointRep.

        Args:
            rep: plotters.PointRep with values to fit.

        Returns:
            Results structure containing fit's results.
        """
        return self.perform_fit(rep.x_val, rep.y_val, rep.y_err)
