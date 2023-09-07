"""Module with highest level abstractions.

These are DataHoldersPool, MomentOfInertiaCalculator and Calculator.
They provide operations with between several EOS-holder classes.
For example, calculation moment of inertia or studying aspect ratio dependence.
"""

from __future__ import annotations
import json
from typing import Set, Tuple, Optional, Dict

import numpy as np

import data_holder_module
import domains
import fitters
import integrators
import interpolators
import plotters
import scale_setters


class DataHoldersPool:
    """Represents a collection of data_holder_module.EosDataHolder classes."""

    data_holders: Set[data_holder_module.EosDataHolder]

    def __init__(self) -> None:
        """Creates empty pool."""
        self.data_holders = set()

    def add(self, holder: data_holder_module.EosDataHolder) -> None:
        """Adds a holder to the pool.

        Args:
            holder: EosDataHolder to add to the collection.
        """
        self.data_holders.add(holder)

    def common_range(self) -> domains.GeometricRange:
        """Calculates a common domain of EOS holders of the pool.

        Returns:
            domains.GeometricRange representing common domain of holders
            in the pool (in terms of beta). Returns empty domain
            in case if the pool is empty.
        """
        if not self.data_holders:
            return domains.GeometricRange()
        holder_domains = [holder.get_data_domain()
                          for holder in self.data_holders]
        result = holder_domains[0]
        for domain in holder_domains:
            result = domain.intersection_with(domain)
        return result


class MomentOfInertiaCalculator:
    """Contains data DataHoldersPool and calculates its moment of inertia."""

    holders_pool: DataHoldersPool
    domain: domains.GeometricRange
    betas: np.ndarray
    _energies: Dict[float, np.ndarray]
    _energy_err: Dict[float, np.ndarray]
    result_linear_c0_val: np.ndarray
    result_linear_c0_err: np.ndarray
    result_linear_c2_val: np.ndarray
    result_linear_c2_err: np.ndarray
    result_quadratic_c0_val: np.ndarray
    result_quadratic_c0_err: np.ndarray
    result_quadratic_c2_val: np.ndarray
    result_quadratic_c2_err: np.ndarray
    result_quadratic_c4_val: np.ndarray
    result_quadratic_c4_err: np.ndarray

    def __init__(self, holders_pool: DataHoldersPool) -> None:
        """Initialize self based on given EosDataHolders' collection.

        Args:
            holders_pool: EosDataHolders' collection with common lattice size
              used to estimate moment of inertia of the system.
        """
        self.holders_pool = holders_pool
        self.domain = self.holders_pool.common_range()
        if self.domain.empty():
            raise ValueError("Cannot calculate EOS v-dependence "
                             "for empty common domain.")
        self.betas = np.arange(self.domain.left, self.domain.right, 0.002)
        energies_val = dict()
        energies_err = dict()
        for holder in self.holders_pool.data_holders:
            value, error = holder.get_energy_interpolation_at_values(
                self.betas)
            energies_val[holder.velocity] = value
            energies_err[holder.velocity] = error
        self._energies = energies_val
        self._energy_err = energies_err

        velocities = [holder.velocity
                      for holder in self.holders_pool.data_holders]
        velocities.sort()
        velocities = np.array(velocities)
        result_linear_c0_val = []
        result_linear_c0_err = []
        result_linear_c2_val = []
        result_linear_c2_err = []
        result_quadratic_c0_val = []
        result_quadratic_c0_err = []
        result_quadratic_c2_val = []
        result_quadratic_c2_err = []
        result_quadratic_c4_val = []
        result_quadratic_c4_err = []
        for i in range(len(self.betas)):
            energies = np.array([energies_val[velocity][i]
                                 for velocity in velocities])
            energy_errors = np.array([energies_err[velocity][i]
                                      for velocity in velocities])

            linear_fit_results = fitters.LinearFitter.perform_fit_new(
                velocities**2, energies, energy_errors)
#             linear_fit_results = fitters.LinearFitter.perform_fit_new_combined(
#                 velocities**2, energies, energy_errors)
            result_linear_c0_val.append(linear_fit_results.c0_val)
            result_linear_c0_err.append(linear_fit_results.c0_err)
            result_linear_c2_val.append(linear_fit_results.c1_val)
            result_linear_c2_err.append(linear_fit_results.c1_err)

            quadratic_fit_results = fitters.QuadraticFitter.perform_fit(
                velocities**2, energies, energy_errors)
            result_quadratic_c0_val.append(quadratic_fit_results.c0_val)
            result_quadratic_c0_err.append(quadratic_fit_results.c0_err)
            result_quadratic_c2_val.append(quadratic_fit_results.c1_val)
            result_quadratic_c2_err.append(quadratic_fit_results.c1_err)
            result_quadratic_c4_val.append(quadratic_fit_results.c2_val)
            result_quadratic_c4_err.append(quadratic_fit_results.c2_err)

        self.result_linear_c0_val = np.array(result_linear_c0_val)
        self.result_linear_c0_err = np.array(result_linear_c0_err)
        self.result_linear_c2_val = np.array(result_linear_c2_val)
        self.result_linear_c2_err = np.array(result_linear_c2_err)
        self.result_quadratic_c0_val = np.array(result_quadratic_c0_val)
        self.result_quadratic_c0_err = np.array(result_quadratic_c0_err)
        self.result_quadratic_c2_val = np.array(result_quadratic_c2_val)
        self.result_quadratic_c2_err = np.array(result_quadratic_c2_err)
        self.result_quadratic_c4_val = np.array(result_quadratic_c4_val)
        self.result_quadratic_c4_err = np.array(result_quadratic_c4_err)

    def get_energies(
        self
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray]]:
        """Returns inner raw representation of the energy data.

        The data is produced by interpolation of EosDataHolders' energy data
        at several beta values in the EosDataHolders' common domain.

        Returns:
            Returns tuple (betas, energy_val, energy_err) where:
              betas: np.ndarray with common beta values.
              energy_val: Dict[float, np.ndarray] with velocities as keys
                and np.ndarrays values of energy estimation at given betas.
              energy_err: Dict[float, np.ndarray] with velocities as keys
                and np.ndarrays error estimates of energy values
                at given betas.
        """
        return self.betas, self._energies, self._energy_err

    def get_quadratic_fit_splrep(
        self
    ) -> Tuple[plotters.SplineRep, plotters.SplineRep, plotters.SplineRep]:
        """Returns SplineRep's of fitted c0, c2 and c4 coefficients.

        Returns:
            Tuple (c0, c2, c4) of plotters.SplineRep representing
            c0, c2 and c4 coefficients' estimates for
            (c0 + c2 * v**2 + c4 * v**4) fitting function for
            normalized free energy f/T^4.
        """
        c0 = plotters.SplineRep.from_points(self.betas[1:],
                                            self.result_quadratic_c0_val[1:],
                                            self.result_quadratic_c0_err[1:])
        c2 = plotters.SplineRep.from_points(self.betas[1:],
                                            self.result_quadratic_c2_val[1:],
                                            self.result_quadratic_c2_err[1:])
        c4 = plotters.SplineRep.from_points(self.betas[1:],
                                            self.result_quadratic_c4_val[1:],
                                            self.result_quadratic_c4_err[1:])
        return c0, c2, c4

    def get_linear_fit_splrep(
            self) -> Tuple[plotters.SplineRep, plotters.SplineRep]:
        """Returns SplineRep's of fitted c0 and c2 coefficients.

        Returns:
            Tuple (c0, c2) of plotters.SplineRep representing
            c0 and c2 coefficients' estimates for
            (c0 + c2 * v**2) fitting function for
            normalized free energy f/T^4.
        """
        c0 = plotters.SplineRep.from_points(self.betas[1:],
                                            self.result_linear_c0_val[1:],
                                            self.result_linear_c0_err[1:])
        c2 = plotters.SplineRep.from_points(self.betas[1:],
                                            self.result_linear_c2_val[1:],
                                            self.result_linear_c2_err[1:])
        return c0, c2


class Calculator:
    """Class for higher logic operations like loading and grouping.

    Groups EosDataHolder classes into groups for moment of inertia and
    aspect ratio dependence analysis.
    """

    nt: int
    beta_critical: float
    scale_setter: scale_setters.GluoScaleSetter
    tc_in_mev: float
    eos_holders_pool: DataHoldersPool
    moi_calc: Optional[MomentOfInertiaCalculator]

    def __init__(self) -> None:
        """Creates an empty Calculator."""
        self.nt = 0
        self.eos_holders_pool = DataHoldersPool()
        self.moi_calc = None

    @classmethod
    def load_from_json(
        cls,
        data_list_json_filename: str,
        integrator: Optional[integrators.Integrator] = None,
        interpolator: Optional[interpolators.Interpolator] = None
    ) -> Calculator:
        """Makes initialization based on json's file instructions.

        (Use "to_input_Nt8.json" as an example.)

        TrapezoidalIntegrator and SplineInterpolator are used as defaults.

        Args:
            data_list_json_filename: Path to json-file's instructions.
            integrator: integrators.Integrator to be used for
              the free energy f/T^4 calculation.
            interpolator: interpolators.Interpolator to be used to perform
              interpolation of the data to common temperatures.

        Returns:
            Returns the initialized Calculator.
        """
        instance = cls()
        instance._load_data(data_list_json_filename, integrator, interpolator)
        return instance

    def _load_data(
        self,
        data_list_json_filename: str,
        integrator: Optional[integrators.Integrator] = None,
        interpolator: Optional[interpolators.Interpolator] = None
    ) -> None:
        """Loads data based on JSON file."""
        with open(data_list_json_filename) as f:
            data_to_load = json.load(f)
            for item in data_to_load["eos_lines"]:
                data_holder = data_holder_module.EosDataHolder(
                    item, integrator, interpolator)
                self.eos_holders_pool.add(data_holder)
                self.nt = data_holder._lattice_size_T.nt
        self.moi_calc = MomentOfInertiaCalculator(self.eos_holders_pool)
        self.beta_critical = data_to_load["beta_critical"]
        if data_to_load["default_scale_setter"] == "symanzik":
            self.scale_setter = scale_setters.SymanzikScaleSetter()
        elif data_to_load["default_scale_setter"] == "wilson":
            self.scale_setter = scale_setters.WilsonScaleSetter()
        elif data_to_load["default_scale_setter"] == "wilson_extended":
            self.scale_setter = scale_setters.ExtendedWilsonScaleSetter()
        else:
            raise RuntimeError("Unknown GluoScaleSetter identifier.")
        self.tc_in_mev = self.scale_setter.get_temperature_in_mev(
            [self.beta_critical], self.nt)[0]
        