"""This module contains EosDataHolder class for single EOS integration line.

This class stores the main data, but calls other corresponding modules to load,
integrate and interpolate it.
"""

from dataclasses import dataclass
from typing import Tuple, Mapping, Sequence, Optional, List

import numpy as np

import domains
import integrators
import interpolators
import plotters
import sychev_csv_reader


@dataclass
class LatticeSizes:
    """Struct used to store dim=4 lattice's sizes in lattice units."""

    nt: int
    nx: int
    ny: int
    nz: int


class EosDataHolder:
    """Implements higher level operation on single EOS line.

    This includes loading from a file, integrating EOS, making interpolations.
    Also provides interface to access pointwise data and interpolations.

    Calls other modules for particular implementation of loading from file,
    integration and interpolation.
    """

    velocity: float
    _beta_integration_node: Optional[float]
    _beta_integrate_from: Optional[float]
    _lattice_size_T: LatticeSizes
    _lattice_size_O: LatticeSizes
    aspect_ratio: float
    _raw_data_T: List[List[float]]
    _raw_data_O: List[List[float]]
    _betas: np.ndarray
    _deltaS_Nt4_val: np.ndarray
    _deltaS_Nt4_err: np.ndarray
    _betas_full: np.ndarray
    _deltaS_Nt4_val_full: np.ndarray
    _deltaS_Nt4_err_full: np.ndarray
    _deltaS_Nt4_err_fraction: np.ndarray
    _int_val: np.ndarray
    _int_err: np.ndarray
    _int_weights: List[np.ndarray]
    _tck_deltaS_val: Tuple
    _tck_deltaS_err: Tuple
    _tck_int_val: Tuple
    _tck_int_err: Tuple

    def __init__(
        self,
        description: Mapping,
        integrator: Optional[integrators.Integrator] = None,
        interpolator: Optional[interpolators.Interpolator] = None
    ) -> None:
        """Init from parsed json file.

        TrapezoidalIntegrator and SplineInterpolator are used as defaults
        respectively.

        Args:
            description: Mapping representing json's data representation
              for the current EOS integration line.
              (Use "to_input_Nt8.json" as an example.)
            integrator: integrators.Integrator to be used as integrator
              to calculate the free energy f/T^4.
            interpolator: interpolators.Interpolator to be used
              where necessary.
        """
        self.velocity = description["velocity"]
        if "beta_integration_node" in description:
            self._beta_integration_node = description["beta_integration_node"]
        else:
            self._beta_integration_node = None
        if "beta_integrate_from" in description:
            self._beta_integrate_from = description["beta_integrate_from"]
        else:
            self._beta_integrate_from = None
        self._lattice_size_T = LatticeSizes(
            description["Nt_T"],
            description["lattice_spacial_size"]["Nx"],
            description["lattice_spacial_size"]["Ny"],
            description["lattice_spacial_size"]["Nz"])
        self._lattice_size_O = LatticeSizes(
            description["Nt_O"],
            description["lattice_spacial_size"]["Nx"],
            description["lattice_spacial_size"]["Ny"],
            description["lattice_spacial_size"]["Nz"])
        self.aspect_ratio = self._lattice_size_T.nz / self._lattice_size_T.nt
        self._raw_data_T = sychev_csv_reader.load_float_csv(
            description["Nt_T_filename"])
        self._raw_data_O = sychev_csv_reader.load_float_csv(
            description["Nt_O_filename"])
        self._calculate_diff()
        if integrator is None:
            integrator = integrators.TrapezoidalIntegrator()
        self._integrate(integrator)
        if interpolator is None:
            interpolator = interpolators.SplineInterpolator()
        self._interpolate(interpolator)

    def _calculate_diff(self) -> None:
        """Calculates physically sensible quantity based on raw data.

        Represents part of the constructor.
        """
        _betas = []
        if len(self._raw_data_T) != len(self._raw_data_O):
            raise ValueError(f"FixNt and Nt=Ns data-filed should contain same "
                             f"number of entries (received "
                             f"len(FixNt)={len(self._raw_data_T)}, "
                             f"len(Nt=Ns)={len(self._raw_data_O)}). "
                             f"v = {self.velocity}")
        for i in range(len(self._raw_data_T)):
            if float(self._raw_data_O[i][0]) == float(self._raw_data_T[i][0]):
                _betas.append(float(self._raw_data_T[i][0]))
            else:
                raise Warning(f"Skipping mismatching betas "
                              f"(beta_T = {self._raw_data_T[i][0]}; "
                              f"beta_O = {self._raw_data_O[i][0]}).")
        self._betas = np.array(_betas)
        t_val = np.array([float(self._raw_data_T[i][1])
                          for i in range(len(self._raw_data_T))])
        t_err = np.array([float(self._raw_data_T[i][2])
                          for i in range(len(self._raw_data_T))])
        o_val = np.array([float(self._raw_data_O[i][1])
                          for i in range(len(self._raw_data_O))])
        o_err = np.array([float(self._raw_data_O[i][2])
                          for i in range(len(self._raw_data_O))])
        self._deltaS_Nt4_val = (o_val - t_val) * self._lattice_size_T.nt ** 4
        self._deltaS_Nt4_err = ((o_err**2 + t_err**2) ** 0.5
                                * self._lattice_size_T.nt ** 4)
        self._deltaS_Nt4_err_fraction = t_err / o_err

    def _integrate(self, integrator: integrators.Integrator) -> None:
        """Uses provided integrator to extract EOS data.

        Represents part of the constructor.
        """
        parameters = None
        if self._beta_integration_node is not None:
            parameters = integrator.IntegrationParameters(
                self._beta_integration_node, 0.0001)
        idx_from = 0
        if self._beta_integrate_from is not None:
            for idx, beta in enumerate(self._betas):
                if self._betas[idx] > (self._beta_integrate_from - 0.0001):
                    idx_from = idx
                    break
        self._betas_full = self._betas
        self._deltaS_Nt4_val_full = self._deltaS_Nt4_val
        self._deltaS_Nt4_err_full = self._deltaS_Nt4_err
        self._betas = self._betas[idx_from:]
        self._deltaS_Nt4_val = self._deltaS_Nt4_val[idx_from:]
        self._deltaS_Nt4_err = self._deltaS_Nt4_err[idx_from:]
        self._int_val, self._int_err, self._int_weights = integrator.integrate(
            self._betas, self._deltaS_Nt4_val, self._deltaS_Nt4_err,
            parameters)

    def _interpolate(self, interpolator: interpolators.Interpolator) -> None:
        """Uses provided interpolator to parametrize discrete EOS data.

        Represents part of the constructor.
        """
        self.interpolator = interpolator
        self._tck_deltaS_val = interpolator.perform_interpolation(
            self._betas, self._deltaS_Nt4_val, self._deltaS_Nt4_err)
        self._tck_deltaS_err = interpolator.perform_interpolation(
            self._betas, self._deltaS_Nt4_err)
        self._tck_int_val = interpolator.perform_interpolation(
            self._betas, self._int_val, self._int_err)
        self._tck_int_err = interpolator.perform_interpolation(
            self._betas, self._int_err)

    def get_data_domain(self) -> domains.GeometricRange:
        """Returns domain (in terms of beta) of current EOS line.

        Returns:
            domains.GeometricRange representing that data.
        """
        return domains.GeometricRange((self._betas[0], self._betas[-1]))

    def get_delta_s_nt4_interpolation(
            self) -> Tuple[interpolators.Interpolator, Tuple, Tuple]:
        """Returns raw delta action interpolation data.

        Returns:
            Tuple (interpolator, delta_s_val, delta_s_err) where:
              interpolator: interpolators.Interpolator used to perform
                interpolation.
              delta_s_val: interpolator's representation of the delta_s value's
                dependence on beta.
              delta_s_err: interpolator's representation of the delta_s error's
                dependence on beta.
        """
        return self.interpolator, self._tck_deltaS_val, self._tck_deltaS_err

    def get_delta_s_nt4_splrep(self) -> plotters.SplineRep:
        """Returns SplineRep representation of delta action data.

        Returns:
            plotters.SplineRep representing delta_s values and its error.
        """
        return plotters.SplineRep.from_points(self._betas,
                                              self._deltaS_Nt4_val,
                                              self._deltaS_Nt4_err)

    def get_delta_s_nt4_interpolation_at_values(
            self, betas: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates interpolation for delta action data at given points.

        Args:
            betas: Sequence of beta values to calculate delta_s at.

        Returns:
            Tuple (delta_s_val, delta_s_err) of np.ndarrays with values of
            delta_s at given betas and their estimated errors.
        """
        result_val = (
            self.interpolator.get_interpolation(betas, self._tck_deltaS_val))
        result_err = (
            self.interpolator.get_interpolation(betas, self._tck_deltaS_err))
        return result_val, result_err

    def get_delta_s_nt4_data(
            self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns raw delta action data.

        Returns:
            Tuple (betas, delta_s_val, delta_s_err) of np.ndarrays containing:
              betas: Values of coupling parameter of raw calculations.
              delta_s_val: Measured delta_s values.
              delta_s_err: Estimated errors of measured delta_s values.
        """
        x_val = self._betas_full.copy()
        y_val = self._deltaS_Nt4_val_full.copy()
        y_err = self._deltaS_Nt4_err_full.copy()
        return x_val, y_val, y_err

    def get_delta_s_nt4_pointrep(self) -> plotters.PointRep:
        """Returns raw delta action data as PointRep.

        Returns:
            plotters.PointRep representing delta_s values
            from direct measurements.
        """
        return plotters.PointRep(self._betas, self._deltaS_Nt4_val,
                                 self._deltaS_Nt4_err)

    def get_energy_interpolation_at_values(
            self, betas: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the free energy's interpolation calculated at given points.

        Args:
            betas: Sequence of beta values to calculate
              the free energy's f/T^4 interpolation at.

        Returns:
            Tuple (val, err) of np.ndarrays with values of
            the free energy's f/T^4 interpolation at given betas
            and their estimated errors.
        """
        result_val = (
            self.interpolator.get_interpolation(betas, self._tck_int_val))
        result_err = (
            self.interpolator.get_interpolation(betas, self._tck_int_err))
        return result_val, result_err

    def get_energy_interpolation(
            self) -> Tuple[interpolators.Interpolator, Tuple, Tuple]:
        """Returns raw free energy f/T^4 interpolation.

        Returns:
            Tuple (interpolator, delta_s_val, delta_s_err) where:
              interpolator: interpolators.Interpolator used to perform
                interpolation.
              energy_val: interpolator's representation of f/T^4 value's
                dependence on beta.
              energy_err: interpolator's representation of f/T^4 error's
                dependence on beta.
        """
        return self.interpolator, self._tck_int_val, self._tck_int_err

    def get_energy_splrep(self) -> plotters.SplineRep:
        """Returns SplineRep free energy interpolation.

        Returns:
            plotters.SplineRep representing the free energy f/T^4 values
            and its error band.
        """
        return plotters.SplineRep.from_points(self._betas,
                                              self._int_val, self._int_err)

    def get_energy_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns raw free energy f/T^4 pointwise data.

        Returns:
            Tuple (betas, energy_val, energy_err) of np.ndarrays containing:
              betas: Values of coupling parameter of raw calculations.
              energy_val: Evaluated free energy's f/T^4 values.
              energy_err: Their estimated statistical errors.
        """
        x_val = self._betas.copy()
        y_val = self._int_val.copy()
        y_err = self._int_err.copy()
        return x_val, y_val, y_err
