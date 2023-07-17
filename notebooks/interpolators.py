"""Provides classes for function interpolation.

For now contains only spline interpolator for one dimensional function.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Optional

import numpy as np
import scipy.interpolate


class Interpolator(ABC):
    """Abstract interface for 1-dim function interpolators."""

    @abstractmethod
    def perform_interpolation(self, x_val: Sequence, y_val: Sequence,
                              y_err: Optional[Sequence] = None) -> Tuple:
        """Perform interpolation of given data.

        Args:
            x_val: Sequence of x-parameter values.
            y_val: Sequence of interpolated function's values.
            y_err: Sequence of errors of interpolated function's values.

        Returns:
            Results inner representation of interpolation results.
            Can be later used in get_interpolation(...) function to calculate
            interpolation's values at points of interest.
        """

    @abstractmethod
    def get_interpolation(self,
                          x_values: Sequence,
                          tck: Optional[Tuple] = None,
                          der: int = 0) -> np.ndarray:
        """Calculate interpolation values at given points.


        Args:
            x_values: Variable's values to calculate interpolation's values at.
            tck: Tuple representing the interpolation (returned by
              perform_interpolation(...) function).
            der: If der > 0, calculate derivative of degree der instead.

        Returns:
            np.ndarray with values of interest at x_values points.
        """


class SplineInterpolator(Interpolator):
    """Class for interpolation via scipy.interpolate.splrep."""

    _tck: Tuple

    def perform_interpolation(self, x_val: Sequence, y_val: Sequence,
                              y_err: Optional[Sequence] = None) -> Tuple:
        """See base class.

        Ignores y_err in current implementation.
        """
        self._tck = scipy.interpolate.splrep(x_val, y_val)
        return self._tck

    def get_interpolation(self,
                          x_values: Sequence,
                          tck: Optional[Tuple] = None,
                          der: int = 0) -> np.ndarray:
        """See base class.

        Derivative's degree has to lie within [0, 3].
        """
        if der < 0 or der > 3:
            raise ValueError(f"Cannot calculate derivative of such degree "
                             f"(received der={der}).")
        if tck is None:
            tck = self._tck
        return np.array(scipy.interpolate.splev(x_values, tck, der))
