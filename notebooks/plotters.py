"""Provides classes and functions for higher level operations with graphs.

Contains PointRep and SplineRep classes for describing sequences of points or
continuous lines correspondingly. Also contains functions for graphing
PointRep and SplineRep via pyplot module.
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence

import numpy as np

import domains
import interpolators


def make_figure(_plt):
    """Creates new pyplot figure with FullHD settings.

    Args:
        _plt: pyplot module.

    Returns:
        Resulting figure.
    """
    my_dpi = 223
    _fig = _plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    _fig.patch.set_facecolor("white")
    return _fig


def make_figure_and_111axis(_plt) -> Tuple:
    """Creates new pyplot one axis figure with FullHD settings.

    Args:
        _plt: pyplot module.

    Returns:
        Resulting Tuple (figure, axis).
    """

    fig = make_figure(_plt)
    ax = fig.add_subplot(111)
    return fig, ax


class PointRep:
    """Class representing discrete series of data with or without errors."""

    x_val: np.ndarray
    y_val: np.ndarray
    y_err: Optional[np.ndarray] = None
    domain: domains.GeometricRange
    initialized: bool = False

    def __init__(self, x_val: Sequence, y_val: Sequence,
                 y_err: Optional[Sequence] = None) -> None:
        """Creates a PointRep from given data. x_val is assumed to be sorted.

        Stores a copy of given data.

        Assumed len(x_val) == len(y_val)
        and len(x_val) == len(y_err) if present.

        Args:
            x_val: Sequence of values of the independent variable.
            y_val: Sequence of values of the dependent variable.
            y_err: Sequence errors of values of the dependent variable.
        """
        if len(x_val) != len(y_val):
            raise ValueError(f"Array with x-values should have the same "
                             f"length as array with y-values (received "
                             f"len(x_val)={len(x_val)}, "
                             f"len(y_val)={len(y_val)}).")
        if y_err is not None:
            if len(x_val) != len(y_err):
                raise ValueError(f"Array with y-errors, if present, should "
                                 f"have the same length as array with "
                                 f"y-values (received "
                                 f"len(x_val)={len(x_val)}, "
                                 f"len(y_err)={len(y_err)}).")
        self.x_val = np.array(x_val)
        self.y_val = np.array(y_val)
        self.y_err = None if y_err is None else np.array(y_err)
        self.domain = domains.GeometricRange((x_val[0], x_val[-1]))
        self.initialized = True

    def subtract_pointrep(self, other: PointRep,
                          add_errors: bool = True) -> PointRep:
        """Calculates difference between PointReps.

        If one of the PointReps doesn't have errors than resulting PointRep
        also doesn't have them.

        Both PointReps are assumed to contain the same points.
        (Equality is not directly checked because of rounding errors
        for floating point data.)

        Args:
            other: PointRep to subtract.
            add_errors: Flag describing if it is needed to add statistical
              errors. (PointReps are assumed to be statically independent
              in that case.)

        Returns:
            Resulting PointRep.
        """
        if not self.initialized or not other.initialized:
            raise ValueError("At least one of the PointReps is not "
                             "initialized.")
        if len(self.x_val) != len(other.x_val):
            raise ValueError(f"PointReps have to contain the same "
                             f"number of points in order to subtract them "
                             f"(received len(self.x_val)={len(self.x_val)}, "
                             f"len(other.x_val)={len(other.x_val)}).")
        x_val = [x for x in self.x_val]
        y_val = [self.y_val[i] - other.y_val[i]
                 for i in range(len(self.x_val))]
        if self.y_err is None or other.y_err is None:
            y_err = None
        else:
            if add_errors:
                y_err = [(self.y_err[i]**2 + other.y_err[i]**2) ** 0.5
                         for i in range(len(self.x_val))]
            else:
                y_err = [self.y_err[i] for i in range(len(self.x_val))]
        return PointRep(x_val, y_val, y_err)

    def multiply_by_constant(self, factor: float) -> PointRep:
        """Returns copy of current SplineRep scaled by constant factor.

        Args:
            factor: Scalar factor to be used for multiplication.

        Returns:
            Resulting SplineRep.
        """
        if not self.initialized:
            raise ValueError("PointRep is not initialized.")
        x_val = [x for x in self.x_val]
        y_val = [self.y_val[i] * factor
                 for i in range(len(self.x_val))]
        if self.y_err is None:
            y_err = None
        else:
            y_err = [self.y_err[i] * abs(factor)
                     for i in range(len(self.x_val))]
        return PointRep(x_val, y_val, y_err)

    def subpointrep_in_domain(
            self, domain: domains.GeometricRange) -> PointRep:
        """Selects points within given domain.

        Args:
            domain: domains.GeometricRange to be used to filter x_values.

        Returns:
            PointRep representing filtered values.
        """
        x_val = [self.x_val[i] for i in range(len(self.x_val))
                 if domain.contains_point(self.x_val[i])]
        y_val = [self.y_val[i] for i in range(len(self.x_val))
                 if domain.contains_point(self.x_val[i])]
        if self.y_err is None:
            y_err = None
        else:
            y_err = [self.y_err[i] for i in range(len(self.x_val))
                     if domain.contains_point(self.x_val[i])]
        return PointRep(x_val, y_val, y_err)


class SplineRep:
    """Class for continuous representation of discrete data via spline."""

    initialized: bool = False
    domain: domains.GeometricRange
    interpolator: interpolators.Interpolator
    tck_val: Tuple
    tck_err: Optional[Tuple] = None
    _points_to_interpolate: int = 500

    def __init__(self) -> None:
        """Uninitialized SplineRep."""
        self.initialized = False

    @classmethod
    def from_points(cls, x_val: Sequence, y_val: Sequence,
                    y_err: Optional[Sequence] = None) -> SplineRep:
        """Initialize from given raw points.

        Args:
            x_val: Sequence of values of the independent variable.
            y_val: Sequence of values of the dependent variable.
            y_err: Sequence errors of values of the dependent variable.

        Returns:
            Resulting SplineRep.
        """
        return cls.from_point_rep(PointRep(x_val, y_val, y_err))

    @classmethod
    def from_point_rep(
        cls, rep: PointRep,
        interpolator: Optional[interpolators.Interpolator] = None
    ) -> SplineRep:
        """Initialize from given PointRep.

        Args:
            rep: PointRep containing discrete points for interpolation.
            interpolator: interpolators.Interpolator to be used
              for interpolation. (interpolators.SplineInterpolator is used
              as defaut.)

        Returns:
            Resulting SplineRep.
        """
        instance = cls()
        if interpolator is None:
            interpolator = interpolators.SplineInterpolator()
        instance._from_point_rep(rep, interpolator)
        return instance

    def _from_point_rep(self, rep: PointRep,
                        interpolator: interpolators.Interpolator) -> None:
        """Initialize from given PointRep."""
        self.domain = rep.domain
        self.interpolator = interpolator
        self.tck_val = interpolator.perform_interpolation(rep.x_val, rep.y_val,
                                                          rep.y_err)
        if rep.y_err is not None:
            self.tck_err = interpolator.perform_interpolation(rep.x_val,
                                                              rep.y_err)
        else:
            self.tck_err = None
        self.initialized = True

    @classmethod
    def from_interpolation(cls,
                           interpolator: interpolators.Interpolator,
                           domain: domains.GeometricRange,
                           tck_val: Tuple,
                           tck_err: Optional[Tuple] = None) -> SplineRep:
        """Initialize from a given interpolation.

        Args:
            interpolator: interpolators.Interpolator used to perform
              interpolations.
            domain: domains.GeometricRange representing domain of correctness
              of the interpolation.
            tck_val: interpolator's representation of
              dependent function's values.
            tck_err: interpolator's representation of statistical errors of
              dependent function's values.

        Returns:
            Resulting SplineRep.
        """
        instance = cls()
        instance._from_interpolation(interpolator, domain, tck_val, tck_err)
        return instance

    def _from_interpolation(self,
                            interpolator: interpolators.Interpolator,
                            domain: domains.GeometricRange,
                            tck_val: Tuple,
                            tck_err: Optional[Tuple] = None) -> None:
        """Initialize from a given interpolation."""
        self.interpolator = interpolator
        self.domain = domain
        self.tck_val = tck_val
        self.tck_err = tck_err
        self.initialized = True

    def subtract_splrep(self, other: SplineRep,
                        add_errors: Optional[bool] = True) -> SplineRep:
        """Calculates difference with another SplineRep.

        Args:
            other: SplineRep to subtract.
            add_errors: Flag describing if it is needed to add statistical
              errors. (SplineReps are assumed to be statically independent
              in that case.)

        Returns:
            Resulting SplineRep.
        """
        if not self.initialized or not other.initialized:
            raise ValueError("One or more of SplineRep's is not initialized.")
        domain = self.domain.intersection_with(other.domain)
        if domain.empty():
            raise ValueError("SplineRep's domains' intersection is empty.")
        x_val = np.linspace(domain.left, domain.right,
                            num=self._points_to_interpolate)
        self_val = self.interpolator.get_interpolation(x_val, self.tck_val)
        other_val = other.interpolator.get_interpolation(x_val, other.tck_val)
        if self.tck_err is None or other.tck_err is None:
            return self.from_points(x_val, self_val - other_val)
        else:
            self_err = self.interpolator.get_interpolation(x_val, self.tck_err)
            other_err = other.interpolator.get_interpolation(x_val,
                                                             other.tck_err)
            return self.from_points(
                x_val, self_val - other_val,
                (self_err**2 + other_err**2)**0.5 if add_errors else self_err)

    def subtract_pointrep(self, pointrep: PointRep,
                          add_errors: bool = True) -> SplineRep:
        """Calculates difference with a PointRep.

        More precisely, with a SplineRep constructed from the PointRep
        via default interpolator.

        Args:
            pointrep: PointRep to subtract.
            add_errors: Flag describing if it is needed to add statistical
              errors. (SplineRep and PointRep are assumed to be
              statically independent in that case.)

        Returns:
            SplineRep representing the result.
        """
        return self.subtract_splrep(self.from_point_rep(pointrep), add_errors)

    def multiply_by_constant(self, factor: float) -> SplineRep:
        """Returns copy of current SplineRep scaled by constant factor.

        Args:
            factor: Scalar factor to be used for multiplication.

        Returns:
            Resulting SplineRep.
        """
        if not self.initialized:
            raise ValueError("SplineRep is not initialized.")
        x_val = np.linspace(self.domain.left, self.domain.right,
                            num=self._points_to_interpolate)
        self_val = self.interpolator.get_interpolation(x_val, self.tck_val)
        if self.tck_err is not None:
            self_err = self.interpolator.get_interpolation(x_val, self.tck_err)
            return self.from_points(x_val, self_val * factor,
                                    self_err * abs(factor))
        else:
            return self.from_points(x_val, self_val * factor)

    def calculate_points(
            self, x_val: Sequence) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Calculate values at given points.

        Args:
            x_val: Sequence of values of dependent variable to calculate
             SplineRep's values at.

        Returns:
            Tuple (val, err) where:
              val: np.ndarray with calculated values at given points.
              err: np.ndarray with estimated statistical errors of calculated
                values at given points. (None is returned if that information
                is not accessible.)
        """
        if not self.initialized:
            raise ValueError("Usage of uninitialized SplineRep "
                             "is not allowed.")
        for value in x_val:
            if not self.domain.contains_point(value):
                raise ValueError(f"One or more entries lies outside of "
                                 f"spline's domain (value {value} is "
                                 f"not contained in self.domain).")
        result_val = self.interpolator.get_interpolation(x_val, self.tck_val)
        if self.tck_err is None:
            result_err = None
        else:
            result_err = self.interpolator.get_interpolation(x_val,
                                                             self.tck_err)
        return result_val, result_err

    def calculate_der_in_points(self, x_val: Sequence,
                                der: int = 1) -> np.ndarray:
        """Calculate derivative of giver order at given points.

        Args:
            x_val: Sequence of values of dependent variable to calculate
             SplineRep's derivative's values at.
            der: Order of derivative to calculate.

        Returns:
            np.ndarray containing calculated values of interest
            at given points.
        """
        if der < 0 or der > 3:
            raise ValueError(f"Order of derivative der has to lie "
                             f"within [0, 3] (received der={der}).")
        if not self.initialized:
            raise ValueError("Usage of uninitialized SplineRep "
                             "is not allowed.")
        for value in x_val:
            if not self.domain.contains_point(value):
                raise ValueError(f"One or more entries lies outside of "
                                 f"spline's domain (value {value} is "
                                 f"not contained in self.domain).")
        result_val = self.interpolator.get_interpolation(x_val, self.tck_val,
                                                         der)
        return result_val


def draw_point_rep(_plt,
                   ax,
                   point_rep: PointRep,
                   colorcode: str,
                   label: Optional[str] = None,
                   linestyle: str = ":",
                   linewidth: Optional[float] = None,
                   domain: Optional[domains.GeometricRange] = None,
                   alpha: float = 1.0,
                   errorfactor: float = 1.0,
                   markertype: str = "o") -> None:
    """Function for plotting PointRep.

    In case of absent domain draws full range.

    Args:
        _plt: pyplot module.
        ax: axis class (of the pyplot module).
        point_rep: PointRep to draw.
        colorcode: str with coded color to used for drawing.
        label: If present, label to be added to axis' legend.
        linestyle: Describes style of the line to connect points.
        linewidth: Width of the line to draw.
        domain: Represents the domain of independent variable to be drawn.
        alpha: Alpha channel for the graph.
        markertype: Defines type of the point marker on the plot.
    """
    if domain is None:
        domain = domains.GeometricRange((point_rep.x_val[0],
                                         point_rep.x_val[-1]))
    elif domain.empty():
        return  # nothing to plot :(
    x_val = []
    y_val = []
    y_err = []  # unused if point_rep.y_err is None
    for i in range(len(point_rep.x_val)):
        if domain.contains_point(point_rep.x_val[i]):
            x_val.append(point_rep.x_val[i])
            y_val.append(point_rep.y_val[i])
            if point_rep.y_err is not None:
                y_err.append(point_rep.y_err[i] * errorfactor)
#     default_capsize = 4
#     default_markersize = 4
    default_capsize = 3.5
    default_markersize = 5.0
    if point_rep.y_err is not None:
        if label is None:
            _plt.errorbar(x_val, y_val, yerr=y_err, capsize=default_capsize, markersize=default_markersize,
                          color=colorcode, linestyle=linestyle,
                          alpha=alpha, marker=markertype)
        else:
            _plt.errorbar(x_val, y_val, yerr=y_err, label=label, capsize=default_capsize,
                          markersize=default_markersize, color=colorcode,
                          linestyle=linestyle, alpha=alpha, marker=markertype)
    else:
        if label is None:
            _plt.errorbar(x_val, y_val, capsize=default_capsize, markersize=default_markersize,
                          color=colorcode, linestyle=linestyle,
                          alpha=alpha, marker=markertype)
        else:
            _plt.errorbar(x_val, y_val, label=label, capsize=default_capsize,
                          markersize=default_markersize, color=colorcode,
                          linestyle=linestyle, alpha=alpha, marker=markertype)


def draw_spl_rep(_plt,
                 ax,
                 spl_rep: SplineRep,
                 colorcode: str,
                 label: Optional[str] = None,
                 domain: Optional[domains.GeometricRange] = None,
                 linestyle: str = "-",
                 linewidth: Optional[float] = None,
                 no_error: bool = False,
                 errorfactor: float = 1.0,
                 alpha: float = 1.0,
                 label_band: bool = False) -> None:
    """Function for plotting SplineRep.

    In case of absent domain draws full range.

    Args:
        _plt: pyplot module.
        ax: axis class (of the pyplot module).
        spl_rep: SplineRep to draw.
        colorcode: str with coded color to used for drawing.
        label: If present, label to be added to axis' legend.
        linestyle: Describes style of the line to connect points.
        linewidth: Width of the line to draw.
        domain: Represents the domain of independent variable to be drawn.
        no_error: Flag guaranteeing the error bands are not plotted.
        alpha: Alpha channel for the graph.
    """
    if domain is None:
        domain = spl_rep.domain
    else:
        domain = domain.intersection_with(spl_rep.domain)
        if domain.empty():
            return  # nothing to plot :(
    number_of_points = 1000
    x_val = np.linspace(domain.left, domain.right, num=number_of_points)
    y_val, y_err = spl_rep.calculate_points(x_val)
    if label is None or label_band is True:
        if linewidth is None:
            _plt.plot(x_val, y_val, color=colorcode, linestyle=linestyle,
                      alpha=alpha)
        else:
            _plt.plot(x_val, y_val, color=colorcode, linestyle=linestyle,
                      alpha=alpha, linewidth=linewidth)
    else:
        if linewidth is None:
            _plt.plot(x_val, y_val, color=colorcode, label=label,
                      linestyle=linestyle, alpha=alpha)
        else:
            _plt.plot(x_val, y_val, color=colorcode, label=label,
                      linestyle=linestyle, alpha=alpha, linewidth=linewidth)
    if y_err is not None and not no_error:
        if label_band is False:
            ax.fill_between(x_val, y_val - y_err*errorfactor, y_val + y_err*errorfactor,
                            color=colorcode, alpha=0.2*alpha) 
#                             hatch="\\\\"
        else:
            ax.fill_between(x_val, y_val - y_err*errorfactor, y_val + y_err*errorfactor,
                            label=label, color=colorcode, alpha=0.2*alpha)
