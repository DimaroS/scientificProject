"""Module for scale setting parameters' conversion.

For now contains conversions for Wilson action gluodynamics.
"""

from abc import ABC, abstractmethod
import math as m
from typing import Callable, Sequence

import numpy as np

import domains


def _bin_root_search(f: Callable[[float], float], root_val: float,
                     left: float, right: float, accuracy: float) -> float:
    """Finds root of the equation f(x) == root_val via binary search.

    Args:
        f: Function to find the root of the equation: f(x) = root_val.
        root_val: Value at the root point.
          (The solved equation is f(x) = root_val.)
        left: Left border of the binary search interval.
        right: Right border of the binary search interval.
        accuracy: Precision to what the function shoul evaluate the root:
          abs(root_exact - root_found) <= accuracy.

    Returns:
        Found root value.
    """
    if accuracy <= 0.0:
        raise ValueError(f"accuracy has to be positive "
                         f"(received accuracy={accuracy}).")
    left_val = f(left)
    right_val = f(right)
    if (root_val > left_val and root_val > right_val
            or root_val < left_val and root_val < right_val):
        raise ValueError(f"Values at both ends are not separated by root_val "
                         f"(received root_val={root_val}, left_val={left_val},"
                         f" right_val={right_val}, left={left},"
                         f" right={right}).")
    if left > right:
        raise ValueError(f"left point cannot be greater then right "
                         f"(received left={left}, right={right}).")
    if (right - left) < accuracy:
        return left

    middle_point = (left + right) / 2
    middle_val = f(middle_point)
    if middle_val > root_val:
        if left_val > root_val:
            return _bin_root_search(f, root_val,
                                    middle_point, right, accuracy)
        else:
            return _bin_root_search(f, root_val,
                                    left, middle_point, accuracy)
    else:
        if left_val > root_val:
            return _bin_root_search(f, root_val,
                                    left, middle_point, accuracy)
        else:
            return _bin_root_search(f, root_val,
                                    middle_point, right, accuracy)


class GluoScaleSetter(ABC):
    """General scale setting for gluodynamics."""

    @abstractmethod
    def get_spacing_in_fm(self, beta: Sequence[float]) -> np.ndarray:
        """Returns spacing in fermi for given beta.

        Args:
            beta: Coupling parameter to calculate the spacing at.

        Returns:
            Returns spacing in femtometres.
        """

    @abstractmethod
    def get_spacing_in_tension_units(self,
                                     beta: Sequence[float]) -> np.ndarray:
        """Returns dimensionless spacing (in terms of string tension).

        Args:
            beta: Coupling parameter to calculate the spacing at.

        Returns:
            Returns dimensionless spacing (in terms of string tension).
        """

    @abstractmethod
    def get_temperature_in_mev(self, beta: Sequence[float],
                               nt: int) -> np.ndarray:
        """Returns temperature for the lattice with given nt and beta.

        Args:
            beta: Coupling parameter to calculate the spacing at.
            nt: Lattice size in temporal direction.

        Returns:
            Returns corresponding temperature measured in MeV.
        """

    @abstractmethod
    def get_temperature_in_tension_units(self, beta: Sequence[float],
                                         nt: int) -> np.ndarray:
        """Returns dimensionless temperature (in terms of string tension).

        Args:
            beta: Coupling parameter to calculate the spacing at.
            nt: Lattice size in temporal direction.

        Returns:
            Returns dimensionless temperature (measured in terms of
            string tension).
        """

    @abstractmethod
    def beta_from_temperature_in_mev(self, temp: Sequence[float], nt: int,
                                     accuracy: float = 0.0001) -> np.ndarray:
        """Returns beta that gives given temperature on given lattice.

        Args:
            temp: Given temperature in MeV.
            nt: Lattice size in temporal direction.
            accuracy: Requested precision for the coupling parameter beta.

        Returns:
            Returns coupling parameter's value corresponding to given
            temperature and lattice temporal extent.
        """

    @abstractmethod
    def get_beta_domain(self) -> domains.GeometricRange:
        """Return domain in terms of beta.

        Returns:
            domains.GeometricRange representing in terms of coupling parameter
            beta the range of applicability of present scale setter.
        """


class WilsonScaleSetter(GluoScaleSetter):
    """Scale setter for Wilson action gluodynamics.

    Applicable bor betas within [5.7, 7.4].
    """

    _w0_in_fm: float = 0.1670
    _r0_in_fm: float = 0.49
    _tension_in_mev: float = 440.0
    _mev_in_tension_units: float = 1 / 440
    _fm_in_MeV: float = 1 / 197.327

    def get_beta_domain(self) -> domains.GeometricRange:
        """See base class."""
        return domains.GeometricRange((5.7, 7.4))

    @staticmethod
    def _gluing_function(v1: float, v2: float, fraction: float) -> float:
        """Used to glue two scale setting functions."""
        if fraction < 0.5:
            factor = 2 * fraction**2
        else:
            factor = 1 - 2 * (1 - fraction)**2
        return (1 - factor) * v1 + factor * v2

    def _spacing_in_fm_0108008(self, beta: float) -> float:
        """Applicable for beta within [5.7, 6.9].

        From arXiv[0108008].
        """
        _beta = beta - 6.0
        return self._r0_in_fm * m.exp(-1.6804
                                      - 1.7331 * _beta
                                      + 0.7849 * _beta ** 2
                                      - 0.4428 * _beta ** 3)

    def _spacing_in_fm_9711003(self, beta: float) -> float:
        """Applicable for beta within [5.6, 6.5].

        From arXiv[9711003].
        """
        def f(_beta: float) -> float:
            """Universal two-oop SU(3) scaling function."""
            b0 = 11 / (4 * m.pi)**2
            b1 = 102 / (4 * m.pi)**4
            return ((6 * b0 / _beta) ** (-b1 / (2 * b0**2))
                    * m.exp(-_beta / (6 * 2 * b0)))
        frac = f(beta) / f(6.0)
        return self._r0_in_fm * f(beta) * (1 + 0.2106 * frac**2
                                           + 0.05492 * frac**4) / 0.01596

    def _spacing_in_fm_1610_07810(self, beta: float) -> float:
        """Applicable for beta within [6.3, 7.4].

        From arXiv[1610.07810].
        """
        return self._w0_in_fm / m.exp(4 * m.pi**2 * beta / 33 - 9.1268
                                      + 41.806 / beta - 158.26 / beta**2)

    def get_spacing_in_fm(self, beta: Sequence[float]) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.7, 7.4].
        """
        return np.array([self._get_spacing_in_fm(_beta) for _beta in beta])

    def _get_spacing_in_fm(self, beta: float) -> float:
        if 5.7 <= beta <= 6.3:
            return self._spacing_in_fm_0108008(beta)
        elif 6.3 < beta <= 6.9:
            return self._gluing_function(self._spacing_in_fm_0108008(beta),
                                         self._spacing_in_fm_1610_07810(beta),
                                         (beta - 6.3) / (6.9 - 6.3))
        elif 6.9 < beta <= 7.4:
            return self._spacing_in_fm_1610_07810(beta)
        else:
            raise ValueError(f"beta has to be within [5.7, 7.4] interval "
                             f"(received beta={beta}).")

    def get_spacing_in_tension_units(self,
                                     beta: Sequence[float]) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.7, 7.4].
        """
        return np.array([self._get_spacing_in_tension_units(_beta)
                         for _beta in beta])

    def _get_spacing_in_tension_units(self, beta: float) -> float:
        return (self._get_spacing_in_fm(beta) * self._fm_in_MeV
                / self._tension_in_mev)

    def get_temperature_in_mev(self, beta: Sequence[float],
                               nt: int) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.7, 7.4].
        """
        return np.array([self._get_temperature_in_mev(_beta, nt)
                         for _beta in beta])

    def _get_temperature_in_mev(self, beta: float, nt: int) -> float:
        return 1 / nt / self._get_spacing_in_fm(beta) / self._fm_in_MeV

    def get_temperature_in_tension_units(self, beta: Sequence[float],
                                         nt: int) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.7, 7.4].
        """
        return np.array([self._get_temperature_in_tension_units(_beta, nt)
                         for _beta in beta])

    def _get_temperature_in_tension_units(self, beta: float, nt: int) -> float:
        return 1 / nt / self._get_spacing_in_tension_units(beta)

    def beta_from_temperature_in_mev(self, temp: Sequence[float], nt: int,
                                     accuracy: float = 0.0001) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.7, 7.4].
        """
        return np.array(
            [self._beta_from_temperature_in_mev(_temp, nt, accuracy)
             for _temp in temp])

    def _beta_from_temperature_in_mev(self, temp: float, nt: int,
                                      accuracy: float = 0.0001) -> float:
        return _bin_root_search(
            (lambda beta: self._get_temperature_in_mev(beta, nt)),
            temp, 5.7, 7.4, accuracy)


class ExtendedWilsonScaleSetter(WilsonScaleSetter):
    """Extended range scale setter for Wilson action gluodynamics.

    Applicable for betas within [5.6, 7.4].
    """

    def get_beta_domain(self) -> domains.GeometricRange:
        """See base class."""
        return domains.GeometricRange((5.6, 7.4))

    def get_spacing_in_fm(self, beta: Sequence[float]) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.6, 7.4].
        """
        return np.array([self._get_spacing_in_fm(_beta) for _beta in beta])

    def _get_spacing_in_fm(self, beta: float) -> float:
        """See base class.

        Applicable for beta within [5.6, 7.4].
        """
        if 5.6 <= beta <= 5.7:
            return self._spacing_in_fm_9711003(beta)
        elif 5.7 < beta <= 5.8:
            return self._gluing_function(self._spacing_in_fm_9711003(beta),
                                         self._spacing_in_fm_0108008(beta),
                                         (beta - 5.7) / (5.8 - 5.7))
        elif 5.8 < beta <= 6.3:
            return self._spacing_in_fm_0108008(beta)
        elif 6.3 < beta <= 6.9:
            return self._gluing_function(self._spacing_in_fm_0108008(beta),
                                         self._spacing_in_fm_1610_07810(beta),
                                         (beta - 6.3) / (6.9 - 6.3))
        elif 6.9 < beta <= 7.4:
            return self._spacing_in_fm_1610_07810(beta)
        else:
            raise ValueError(f"beta has to be within [5.6, 7.4] interval "
                             f"(received beta={beta}).")

    def beta_from_temperature_in_mev(self, temp: Sequence[float], nt: int,
                                     accuracy: float = 0.0001) -> np.ndarray:
        """See base class.

        Applicable for beta within [5.6, 7.4].
        """
        return np.array(
            [self._beta_from_temperature_in_mev(_temp, nt, accuracy)
             for _temp in temp])

    def _beta_from_temperature_in_mev(self, temp: float, nt: int,
                                      accuracy: float = 0.0001) -> float:
        """See base class.

        Applicable for beta within [5.6, 7.4].
        """
        return _bin_root_search(
            (lambda beta: self._get_temperature_in_mev(beta, nt)),
            temp, 5.6, 7.4, accuracy)


class SymanzikScaleSetter(GluoScaleSetter):
    """Scale setter for Symanzik action gluodynamics.

    Applicable for betas within [3.85, 5.0].
    """

    _w0_in_fm: float = 0.1670
    _r0_in_fm: float = 0.49
    _tension_in_mev: float = 440.0
    _mev_in_tension_units: float = 1 / 440
    _fm_in_MeV: float = 1 / 197.327

    def get_beta_domain(self) -> domains.GeometricRange:
        """See base class."""
        return domains.GeometricRange((3.85, 5.0))

    def _spacing_in_fm_9707023(self, beta: float) -> float:
        """Applicable for beta within [3.85, 5.0].

        This fit is based on [arXiv:hep-lat/9707023].
        """
        def f(_beta: float) -> float:
            """Universal two-oop SU(3) scaling function."""
            b0 = 11 / (4 * m.pi)**2
            b1 = 102 / (4 * m.pi)**4
            return ((6 * b0 / _beta) ** (-b1 / (2 * b0**2))
                    * m.exp(-_beta / (6 * 2 * b0)))
        frac = f(beta) / f(4.0)
        return self._mev_in_tension_units / self._fm_in_MeV * f(beta) * (
                1 + 0.21005651 * frac**2 + 0.33740354 * frac**4
                - 0.13137041 * frac**6) / 0.06829413

    def get_spacing_in_fm(self, beta: Sequence[float]) -> np.ndarray:
        """See base class.

        Applicable for beta within [3.85, 5.0].
        """
        return np.array([self._get_spacing_in_fm(_beta) for _beta in beta])

    def _get_spacing_in_fm(self, beta: float) -> float:
        if 3.85 <= beta <= 5.0:
            return self._spacing_in_fm_9707023(beta)
        else:
            raise ValueError(f"beta has to be within [3.85, 5.0] interval "
                             f"(received beta={beta}).")

    def get_spacing_in_tension_units(self,
                                     beta: Sequence[float]) -> np.ndarray:
        """See base class.

        Applicable for beta within [3.85, 5.0].
        """
        return np.array([self._get_spacing_in_tension_units(_beta)
                         for _beta in beta])

    def _get_spacing_in_tension_units(self, beta: float) -> float:
        return (self._get_spacing_in_fm(beta) * self._fm_in_MeV
                / self._tension_in_mev)

    def get_temperature_in_mev(self, beta: Sequence[float],
                               nt: int) -> np.ndarray:
        """See base class.

        Applicable for beta within [3.85, 5.0].
        """
        return np.array([self._get_temperature_in_mev(_beta, nt)
                         for _beta in beta])

    def _get_temperature_in_mev(self, beta: float, nt: int) -> float:
        return 1 / nt / self._get_spacing_in_fm(beta) / self._fm_in_MeV

    def get_temperature_in_tension_units(self, beta: Sequence[float],
                                         nt: int) -> np.ndarray:
        """See base class.

        Applicable for beta within [3.85, 5.0].
        """
        return np.array([self._get_temperature_in_tension_units(_beta, nt)
                         for _beta in beta])

    def _get_temperature_in_tension_units(self, beta: float, nt: int) -> float:
        return 1 / nt / self._get_spacing_in_tension_units(beta)

    def beta_from_temperature_in_mev(self, temp: Sequence[float], nt: int,
                                     accuracy: float = 0.0001) -> np.ndarray:
        """See base class.

        Applicable for beta within [3.85, 5.0].
        """
        return np.array(
            [self._beta_from_temperature_in_mev(_temp, nt, accuracy)
             for _temp in temp])

    def _beta_from_temperature_in_mev(self, temp: float, nt: int,
                                      accuracy: float = 0.0001) -> float:
        return _bin_root_search(
            (lambda beta: self._get_temperature_in_mev(beta, nt)),
            temp, 3.85, 5.0, accuracy)




# #GCC     	=g++  -std=c++1y
# #GCC     	=/Users/dimaros/opt/anaconda3/envs/compilers_env/bin/clang-11
#
# #LDFLAGS 	+= -I/Users/dimaros/opt/anaconda3/envs/compilers_env/include
# LDFLAGS 	+= -I/Users/dimaros/opt/anaconda3/envs/openmp/include

# rrcmpi (test1)
# 6.799999999999999822e+00 2.044243761553360184e+00 7.910297720849120547e-06

# rrcmpi (test2)
# 6.799999999999999822e+00 2.044256089379907326e+00 8.180415162988010787e-06