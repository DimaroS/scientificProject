"""This module contains classes for constant and linear extrapolations.

Designed to perform extrapolation on plotters.SplineReps.
"""

from typing import Sequence

import numpy as np

import fitters
import plotters


class ConstantExtrapolator:
    """Extrapolation based on fitting by a constant function."""

    @staticmethod
    def from_splinereps(
            x_val: Sequence[float],
            y_valerr: Sequence[plotters.SplineRep]) -> plotters.SplineRep:
        """Return results of extrapolation as plotters.SplineRep.

        Extrapolation is performed in common domain of given SplineReps.

        Current implementation requires definite errors for SplineReps.

        Points are assumed to be statistically independent.
        """
        x_val = [i for i in x_val]
        y_valerr = [i for i in y_valerr]
        if len(x_val) != len(y_valerr):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length. (Given: len(x_val)={len(x_val)};"
                f"len(y_valerr)={len(y_valerr)})")
        if len(x_val) == 0:
            raise ValueError("Cannot perform extrapolation on zero points.")
        comm_domain = y_valerr[0].domain
        for splrep in y_valerr:
            comm_domain = comm_domain.intersection_with(splrep.domain)

        points = np.linspace(comm_domain.left, comm_domain.right, 500)
        y_val = []
        y_err = []
        for splrep in y_valerr:
            _val, _err = splrep.calculate_points(points)
            y_val.append(_val)
            y_err.append(_err)
            if _err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for SplineReps.")
        result_val = []
        result_err = []
        for p_idx in range(len(points)):
            _y_val = np.array([_val[p_idx] for _val in y_val])
            _y_err = np.array([_err[p_idx] for _err in y_err])
            inv_err = _y_err**(-2)
            total_inv_err = np.sum(inv_err)

            weighted_sum = np.sum(_y_val * inv_err)
            tot_value = weighted_sum / total_inv_err
            tot_err = total_inv_err**(-0.5)
            result_val.append(tot_value)
            result_err.append(tot_err)
        return plotters.SplineRep.from_points(points, result_val, result_err)


class LinearExtrapolator:
    """Extrapolation based on fitting by a linear function."""

    @staticmethod
    def from_splinereps(
            x_val: Sequence[float],
            y_valerr: Sequence[plotters.SplineRep]) -> plotters.SplineRep:
        """Return extrapolation's results as plotters.SplineRep.

        Extrapolation is performed in common domain of given SplineReps.

        Current implementation requires definite errors for SplineReps.

        Points are assumed to be statistically independent.
        """
        x_val = [i for i in x_val]
        y_valerr = [i for i in y_valerr]
        if len(x_val) != len(y_valerr):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length. (Given: len(x_val)={len(x_val)};"
                f"len(y_valerr)={len(y_valerr)})")
        if len(x_val) == 0:
            raise ValueError("Cannot perform extrapolation on zero points.")
        comm_domain = y_valerr[0].domain
        for splrep in y_valerr:
            comm_domain = comm_domain.intersection_with(splrep.domain)
        points = np.linspace(comm_domain.left, comm_domain.right, 500)

        y_val = []
        y_err = []
        for splrep in y_valerr:
            _val, _err = splrep.calculate_points(points)
            y_val.append(_val)
            y_err.append(_err)
            if _err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for SplineReps.")
        result_val = []
        result_err = []
        fitter = fitters.LinearFitter()
        for p_idx in range(len(points)):
            _x_val = np.array(x_val)
            _y_val = np.array([_val[p_idx] for _val in y_val])
            _y_err = np.array([_err[p_idx] for _err in y_err])
            res = fitter.perform_fit_new(_x_val, _y_val, _y_err)
            result_val.append(res.c0_val)
            result_err.append(res.c0_err)
        return plotters.SplineRep.from_points(points, result_val, result_err)
    
    @staticmethod
    def from_splinereps_combined(
            x_val: Sequence[float],
            y_valerr: Sequence[plotters.SplineRep]) -> plotters.SplineRep:
        """Return extrapolation's results as plotters.SplineRep.

        Extrapolation is performed in common domain of given SplineReps.

        Current implementation requires definite errors for SplineReps.

        Points are assumed to be statistically independent.
        """
        x_val = [i for i in x_val]
        y_valerr = [i for i in y_valerr]
        if len(x_val) != len(y_valerr):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length. (Given: len(x_val)={len(x_val)};"
                f"len(y_valerr)={len(y_valerr)})")
        if len(x_val) == 0:
            raise ValueError("Cannot perform extrapolation on zero points.")
        comm_domain = y_valerr[0].domain
        for splrep in y_valerr:
            comm_domain = comm_domain.intersection_with(splrep.domain)
        points = np.linspace(comm_domain.left, comm_domain.right, 500)

        y_val = []
        y_err = []
        for splrep in y_valerr:
            _val, _err = splrep.calculate_points(points)
            y_val.append(_val)
            y_err.append(_err)
            if _err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for SplineReps.")
        result_val = []
        result_err = []
        fitter = fitters.LinearFitter()
        for p_idx in range(len(points)):
            _x_val = np.array(x_val)
            _y_val = np.array([_val[p_idx] for _val in y_val])
            _y_err = np.array([_err[p_idx] for _err in y_err])
            res = fitter.perform_fit_new_combined(_x_val, _y_val, _y_err)
            result_val.append(res.c0_val)
            result_err.append(res.c0_err)
        return plotters.SplineRep.from_points(points, result_val, result_err)
    
    @staticmethod
    def from_pointreps(
            x_val: Sequence[float],
            y_valerr: Sequence[plotters.PointRep]) -> plotters.PointRep:
        """Return extrapolation's results as plotters.PointRep.

        Current implementation requires definite errors for PointReps.

        Points are assumed to be statistically independent.
        """
        x_val = [i for i in x_val]
        y_valerr = [i for i in y_valerr]
        if len(x_val) != len(y_valerr):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length. (Given: len(x_val)={len(x_val)};"
                f"len(y_valerr)={len(y_valerr)})")
        if len(x_val) == 0:
            raise ValueError("Cannot perform extrapolation on zero points.")
        points = y_valerr[0].x_val.copy()

        y_val = []
        y_err = []
        for prep in y_valerr:
            y_val.append(prep.y_val)
            y_err.append(prep.y_err)
            if prep.y_err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for PointReps.")
        result_val = []
        result_err = []
        fitter = fitters.LinearFitter()
        for p_idx in range(len(points)):
            _x_val = np.array(x_val)
            _y_val = np.array([_val[p_idx] for _val in y_val])
            _y_err = np.array([_err[p_idx] for _err in y_err])
            res = fitter.perform_fit_new(_x_val, _y_val, _y_err)
            result_val.append(res.c0_val)
            result_err.append(res.c0_err)
        return plotters.PointRep(points, result_val, result_err)
    
    @staticmethod
    def from_pointreps_combined(
            x_val: Sequence[float],
            y_valerr: Sequence[plotters.PointRep]) -> plotters.PointRep:
        """Return extrapolation's results as plotters.PointRep.

        Current implementation requires definite errors for PointReps.

        Points are assumed to be statistically independent.
        """
        x_val = [i for i in x_val]
        y_valerr = [i for i in y_valerr]
        if len(x_val) != len(y_valerr):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length. (Given: len(x_val)={len(x_val)};"
                f"len(y_valerr)={len(y_valerr)})")
        if len(x_val) == 0:
            raise ValueError("Cannot perform extrapolation on zero points.")
        points = y_valerr[0].x_val.copy()

        y_val = []
        y_err = []
        for prep in y_valerr:
            y_val.append(prep.y_val)
            y_err.append(prep.y_err)
            if prep.y_err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for PointReps.")
        result_val = []
        result_err = []
        fitter = fitters.LinearFitter()
        for p_idx in range(len(points)):
            _x_val = np.array(x_val)
            _y_val = np.array([_val[p_idx] for _val in y_val])
            _y_err = np.array([_err[p_idx] for _err in y_err])
            res = fitter.perform_fit_new_combined(_x_val, _y_val, _y_err)
            result_val.append(res.c0_val)
            result_err.append(res.c0_err)
        return plotters.PointRep(points, result_val, result_err)


class BilinearExtrapolator:
    """Extrapolation based on fitting by a bilinear function.

    Meaning the function is assumed to be (a + b1*x1 + b2*x2).

    Two sets of linearly dependent data are assumed. (With different slopes.)
    """

    @staticmethod
    def from_splinereps(
            x_val_1: Sequence[float],
            y_valerr_1: Sequence[plotters.SplineRep],
            x_val_2: Sequence[float],
            y_valerr_2: Sequence[plotters.SplineRep]) -> plotters.SplineRep:
        """Return extrapolation's results as plotters.SplineRep.

        Extrapolation is performed in common domain of given SplineReps.

        Current implementation requires definite errors for SplineReps.

        Points are assumed to be statistically independent.
        """
        x_val_1 = [i for i in x_val_1]
        x_val_2 = [i for i in x_val_2]
        y_valerr_1 = [i for i in y_valerr_1]
        y_valerr_2 = [i for i in y_valerr_2]
        if len(x_val_1) != len(y_valerr_1):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length in the same pair. "
                f"(Given: len(x_val_1)={len(x_val_1)};"
                f"len(y_valerr_1)={len(y_valerr_1)})")
        if len(x_val_2) != len(y_valerr_2):
            raise ValueError(
                f"Extrapolation function requires arguments of "
                f"the same length in the same pair. "
                f"(Given: len(x_val_2)={len(x_val_2)};"
                f"len(y_valerr_2)={len(y_valerr_2)})")
        if len(x_val_1) == 0 or len(x_val_2) == 0:
            raise ValueError("Each subset has to contain "
                             "at least one SplineRep.")
        comm_domain = y_valerr_1[0].domain
        for splrep in y_valerr_1:
            comm_domain = comm_domain.intersection_with(splrep.domain)
        for splrep in y_valerr_2:
            comm_domain = comm_domain.intersection_with(splrep.domain)
        points = np.linspace(comm_domain.left, comm_domain.right, 500)

        y_val_1 = []
        y_err_1 = []
        for splrep in y_valerr_1:
            _val, _err = splrep.calculate_points(points)
            y_val_1.append(_val)
            y_err_1.append(_err)
            if _err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for SplineReps.")
        y_val_2 = []
        y_err_2 = []
        for splrep in y_valerr_2:
            _val, _err = splrep.calculate_points(points)
            y_val_2.append(_val)
            y_err_2.append(_err)
            if _err is None:
                raise NotImplementedError("Current implementation requires "
                                          "definite errors for SplineReps.")
        result_val = []
        result_err = []
        fitter = fitters.BilinearFitter()
        for p_idx in range(len(points)):
            _x_val_1 = np.array(x_val_1)
            _y_val_1 = np.array([_array[p_idx] for _array in y_val_1])
            _y_err_1 = np.array([_array[p_idx] for _array in y_err_1])
            _x_val_2 = np.array(x_val_2)
            _y_val_2 = np.array([_array[p_idx] for _array in y_val_2])
            _y_err_2 = np.array([_array[p_idx] for _array in y_err_2])
            res = fitter.perform_fit(_x_val_1, _y_val_1, _y_err_1,
                                     _x_val_2, _y_val_2, _y_err_2)
            result_val.append(res.c_val)
            result_err.append(res.c_err)
        return plotters.SplineRep.from_points(points, result_val, result_err)
