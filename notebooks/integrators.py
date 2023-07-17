"""Provides classes for one dimensional integration.

For now contains only TrapezoidalIntegrator.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Optional, List

import numpy as np

import domains


class Monomial:
    """Class for shifted monomial's storage, integration and evaluation."""

    amplitude: float
    weights: np.ndarray
    power: int
    base: float

    def __init__(self, amplitude: float, weights: np.ndarray, power: int,
                 base: float = 0.0) -> None:
        """Represents single variable monomial: amplitude * (x - base)**power.

        Args:
            amplitude: Scalar factor to (x - base)**power monomial.
            weights: Weights used to construct the amplitude.
            power: Power of the monomial.
            base: Shift in the variable in the monomial. (Introduced to help
              minimize numerical errors when calculating
              values of composite polynomials in case of big cancellations.)
        """
        self.amplitude = amplitude
        self.weights = weights
        if power < 0:
            raise ValueError(f"Negative powers are not supported "
                             f"for monomials (received power={power}).")
        self.power = power
        self.base = base

    def evaluate_at(self, point: float) -> Tuple[float, np.ndarray]:
        """Evaluate the monomial's value at given point.

        Args:
            point: Point to evaluate monomial's value at.

        Returns:
            Tuple (v, w) containing v - value of the monomial at given point
            and w - vector of resulting weights in the result
            (initial weights are given via ctor).
        """
        factor = (point - self.base)**self.power
        return self.amplitude * factor, self.weights * factor

    def integrate(self,
                  domain: domains.GeometricRange) -> Tuple[float, np.ndarray]:
        """Integrate the monomial over given interval.

        Args:
            domain: Domain to perform definite integration over.

        Returns:
            Tuple (v, w) containing v - value of the definite integral
            (0.0 for an empty domain) and w - vector of resulting weights
            in the result (initial weights are given via ctor).
        """
        if domain.empty():
            return 0.0, 0.0 * self.weights
        factor = (((domain.right - self.base)**(self.power + 1)
                   - (domain.left - self.base)**(self.power + 1))
                  / (self.power + 1))
        return self.amplitude * factor, self.weights * factor


class PolynamialInterpolant(ABC):
    """Abstract interface for exact polynomial interpolation."""

    @abstractmethod
    def integrate(self,
                  domain: domains.GeometricRange) -> Tuple[float, np.ndarray]:
        """Integrates the interpolant over given range.

        Args:
            domain: Domain to perform definite integration over.

        Returns:
            Tuple (v, w) containing v - value of the definite integral
            (0.0 for an empty domain) and w - vector of resulting weights
            in the result (initial weights are given via ctor).
        """

    @abstractmethod
    def evaluate_at(self, point: float) -> Tuple[float, np.ndarray]:
        """Evaluates the interpolant's value at given point.

        Args:
            point: Point to evaluate interpolant's value at.

        Returns:
            Tuple (v, w) containing v - value of the monomial at given point
            and w - vector of resulting weights in the result
            (initial weights are given via ctor).
        """


class ConstantInterpolant(PolynamialInterpolant):
    """Class for constant interpolation."""

    value: float
    weights: np.ndarray
    domain: domains.GeometricRange

    def __init__(self, value: float, weights: np.ndarray,
                 domain: domains.GeometricRange) -> None:
        """Initializes ConstantInterpolant with given value and range.

        Args:
            value: Value of the interpolation.
            weights: Weights used to construct the amplitude.
            domain: domains.GeometricRange representing domain of correctness
              of the interpolation.
        """
        self.value = value
        self.weights = weights
        self.domain = domain
        if domain.empty():
            raise ValueError("Cannot create an interpolant "
                             "for an empty domain.")

    def integrate(self,
                  domain: domains.GeometricRange) -> Tuple[float, np.ndarray]:
        """See base class."""
        if not self.domain.contains_geometric_range(domain):
            raise ValueError("Requested integration's domain does not "
                             "lie within interpolation domain.")
        factor = domain.right - domain.left
        return self.value * factor, self.weights * factor

    def evaluate_at(self, point: float) -> Tuple[float, np.ndarray]:
        """See base class."""
        if not self.domain.contains_point(point):
            raise ValueError("Given point lies outside interpolant's domain.")
        return self.value, self.weights


class LinearInterpolant(PolynamialInterpolant):
    """Class for exact linear interpolation."""

    domain: domains.GeometricRange
    _left_val: float
    _weights_left: np.ndarray
    _right_val: float
    _weights_right: np.ndarray

    def __init__(self, left_val: float, weights_left: np.ndarray,
                 right_val: float, weights_right: np.ndarray,
                 domain: domains.GeometricRange) -> None:
        """Exact linear interpolation with given end-values and domain.

        Args:
            left_val: Value of the interpolation at left endpoint.
            weights_left: Weights used to construct left_val.
            right_val: Value of the interpolation at right endpoint.
            weights_right: Weights used to construct right_val.
            domain: domains.GeometricRange representing domain of correctness
              of the interpolation.
        """
        if domain.empty():
            raise ValueError("Cannot create an interpolant "
                             "for an empty domain.")
        if domain.left == domain.right:
            raise ValueError("Cannot initialize linear interpolation "
                             "for a zero length domain.")
        self.domain = domain
        self._left_val = left_val
        self._weights_left = weights_left
        self._right_val = right_val
        self._weights_right = weights_right

    @classmethod
    def from_points(cls, endpoints: Tuple[float, float],
                    values: Tuple[float, float]) -> LinearInterpolant:
        """Initialize the interpolator by given values.

        Args:
            endpoints: Tuple (left, right) containing interpolation's
              variable's endpoints. Assumed left < right.
            values: Tuple (left_val, right_val) where:
              left_val: Value of the interpolation at the left endpoint.
              right_val: Value of the interpolation at the right endpoint.

        Returns:
            Resulting interpolant.
        """
        return cls(values[0], np.array([1.0, 0.0]),
                   values[1], np.array([0.0, 1.0]),
                   domains.GeometricRange(endpoints))

    def integrate(self,
                  domain: domains.GeometricRange) -> Tuple[float, np.ndarray]:
        """See base class."""
        if not self.domain.contains_geometric_range(domain):
            raise ValueError("Requested integration's domain does not "
                             "lie within interpolation domain.")
        v_l, w_l = self.evaluate_at(domain.left)
        v_r, w_r = self.evaluate_at(domain.right)
        factor = (domain.right - domain.left) / 2
        return (v_l + v_r) * factor, (w_l + w_r) * factor

    def evaluate_at(self, point: float) -> Tuple[float, np.ndarray]:
        """See base class."""
        if not self.domain.contains_point(point):
            raise ValueError("Given point lies outside interpolant's domain.")
        fraction = ((point - self.domain.left)
                    / (self.domain.right - self.domain.left))
        factor_l = 1 - fraction
        factor_r = fraction
        value = self._left_val * factor_l + self._right_val * factor_r
        weights = (self._weights_left * factor_l
                   + self._weights_right * factor_r)
        return value, weights


class QuadraticInterpolant(PolynamialInterpolant):
    """Class for exact quadratic interpolation."""

    domain: domains.GeometricRange
    _monomials: Tuple[Monomial, Monomial, Monomial]

    def __init__(self, c0_v: float, c0_w: np.ndarray,
                 c1_v: float, c1_w: np.ndarray,
                 c2_v: float, c2_w: np.ndarray,
                 base: float, domain: domains.GeometricRange) -> None:
        """Initializes interpolation by given coefficients, base and domain.

        Represents f(x) = c0 + c1*(x - base) + c2*(x - base)**2 interpolation.

        Args:
            c0_v: Interpolation's coefficient.
            c0_w: Weights used to construct c0_v.
            c1_v: Interpolation's coefficient.
            c0_w: Weights used to construct c1_v.
            c2_v: Interpolation's coefficient.
            c0_w: Weights used to construct c2_v.
            base: Shift in the interpolant's variable.
            domain: Interpolation's domain of correctness.
        """
        if domain.empty():
            raise ValueError("Cannot create an interpolant "
                             "for an empty domain.")
        if domain.left == domain.right:
            raise ValueError("Cannot initialize linear interpolation "
                             "for a zero length domain.")
        self.domain = domain
        self._monomials = (Monomial(c0_v, c0_w, 0, base),
                           Monomial(c1_v, c1_w, 1, base),
                           Monomial(c2_v, c2_w, 2, base))

    @classmethod
    def from_points(
            cls, nodes: Tuple[float, float, float],
            values: Tuple[float, float, float]) -> QuadraticInterpolant:
        """Initialize the interpolator by given values.

        Args:
            nodes: Tuple (left, middle, right) with variable's values for
              interpolation.
            values: Tuple (left_val, middle_val, right_val) with interpolated
              function's values at given points.

        Returns:
            Resulting interpolant.
        """
        if nodes[0] >= nodes[1] or nodes[1] >= nodes[2]:
            raise ValueError(f"Given tuple of arguments is not sorted "
                             f"(received nodes={nodes}).")
        base = nodes[1]
        domain = domains.GeometricRange((nodes[0], nodes[2]))
        w0 = np.array([1.0, 0.0, 0.0])
        w1 = np.array([0.0, 1.0, 0.0])
        w2 = np.array([0.0, 0.0, 1.0])
        f1 = values[2] - values[1]
        f1_w = w2 - w1
        f2 = values[0] - values[1]
        f2_w = w0 - w1
        delta1 = nodes[2] - base
        delta2 = base - nodes[0]
        c0 = values[1]
        c0_w = w1
        c1 = ((f1 * delta2**2 - f2 * delta1**2)
              / (delta1 * delta2 * (delta1 + delta2)))
        c1_w = ((f1_w * delta2**2 - f2_w * delta1**2)
                / (delta1 * delta2 * (delta1 + delta2)))
        c2 = ((f1 * delta2 + f2 * delta1)
              / (delta1 * delta2 * (delta1 + delta2)))
        c2_w = ((f1_w * delta2 + f2_w * delta1)
                / (delta1 * delta2 * (delta1 + delta2)))
        return cls(c0, c0_w, c1, c1_w, c2, c2_w, base, domain)

    def integrate(self,
                  domain: domains.GeometricRange) -> Tuple[float, np.ndarray]:
        """See base class."""
        if not self.domain.contains_geometric_range(domain):
            raise ValueError("Requested integration's domain does not "
                             "lie within interpolation domain.")
        v0, w0 = self._monomials[0].integrate(domain)
        v1, w1 = self._monomials[1].integrate(domain)
        v2, w2 = self._monomials[2].integrate(domain)
        return v0 + v1 + v2, w0 + w1 + w2

    def evaluate_at(self, point: float) -> Tuple[float, np.ndarray]:
        """See base class."""
        if not self.domain.contains_point(point):
            raise ValueError("Given point lies outside interpolant's domain.")
        v0, w0 = self._monomials[0].evaluate_at(point)
        v1, w1 = self._monomials[1].evaluate_at(point)
        v2, w2 = self._monomials[2].evaluate_at(point)
        return v0 + v1 + v2, w0 + w1 + w2


class Integrator(ABC):
    """Abstract interface for Integrator classes."""

    class IntegrationParameters:
        """Class representing parameters for trapezoidal integration.

        Currently, has no parameters.
        """

        def __init__(self, x_value: float, x_value_precision: float) -> None:
            pass

    @abstractmethod
    def integrate(
        self,
        x_val: Sequence,
        y_val: Sequence,
        y_err: Optional[Sequence] = None,
        parameters: Optional[IntegrationParameters] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Integrate given at discrete points function with optional errors.

        Args:
            x_val: Sequence of function's parameters values.
            y_val: Sequence of function's values at given x_val points.
            y_err: Sequence of statistical errors of function's values
              at given x_val points.
            parameters: Integration parameters (including specific
              to integration method).

        Returns:
            Tuple (int_val, int_err) of partial integral values (like partial
            sums) and their estimated statistical errors respectively.
        """


class TrapezoidalIntegrator(Integrator):
    """Class for trapezoidal numerical integration method."""

    class IntegrationParameters:
        """Class representing parameters for trapezoidal integration.

        Currently, has no parameters.
        """

        def __init__(self, x_value: float, x_value_precision: float) -> None:
            pass

    def integrate(
        self,
        x_val: Sequence,
        y_val: Sequence,
        y_err: Optional[Sequence] = None,
        parameters: Optional[IntegrationParameters] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Numerically integrates given data via trapezoidal method.

        Args:
            x_val: Sequence of function's parameters values.
            y_val: Sequence of function's values at given x_val points.
            y_err: Sequence of statistical errors of function's values
              at given x_val points. If equals None zero errors are assumed.
            parameters: Integration parameters (including specific
              to integration method).

        Returns:
            Tuple (int_val, int_err) of partial integral values (like partial
            sums) and their estimated statistical errors respectively.
        """
        x_val = np.array([i for i in x_val])
        y_val = np.array([i for i in y_val])
        if len(x_val) != len(y_val):
            raise ValueError(f"Array with x-values should have the same "
                             f"length as array with y-values "
                             f"(received len(x_val)={len(x_val)}, "
                             f"len(y_val)={len(y_val)}).")
        if y_err is not None:
            y_err = np.array([i for i in y_err])
            if len(x_val) != len(y_err):
                raise ValueError(f"Array with y-arrors, if present, should "
                                 f"have the same length as y-values "
                                 f"(received len(x_val)={len(x_val)}, "
                                 f"len(y_err)={len(y_err)}).")
        else:
            y_err = 0.0 * x_val
        int_val = [0.0]
        int_err = [0.00000000001]
        int_weights = [0.00000000001 + 0.0 * x_val]
        partial_weights = 0.0 * x_val
        for i in range(1, len(x_val)):
            interpolant = LinearInterpolant.from_points(
                (x_val[i - 1], x_val[i]), (y_val[i - 1], y_val[i]))
            step_value, step_weights = interpolant.integrate(
                domains.GeometricRange((x_val[i - 1], x_val[i])))
            int_val.append(int_val[-1] + step_value)
            partial_weights[i - 1] += step_weights[0]
            partial_weights[i] += step_weights[1]
            int_err.append(
                np.sqrt(np.sum((y_err * partial_weights)**2)))
            int_weights.append(partial_weights.copy())
        return np.array(int_val), np.array(int_err), int_weights


class QuadraticIntegrator(Integrator):
    """Class for numerical integration using exact quadratic fits.

    For now quadratic integration is performed
    after specified integration node (if it is given).
    """

    class IntegrationParameters:
        """Class representing parameters for quadratic integration."""

        x_node_value: float
        x_node_value_precision: float

        def __init__(self, x_value: float, x_value_precision: float) -> None:
            self.x_node_value = x_value
            self.x_node_value_precision = x_value_precision

    def integrate(
        self,
        x_val: Sequence,
        y_val: Sequence,
        y_err: Optional[Sequence] = None,
        parameters: Optional[IntegrationParameters] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Numerically integrates given data via quadratic interpolation.

        For now quadratic integration is performed after specified
        integration node (if it is given). If not, performs
        quadratic integration over the whole integration interval
        (in case the number of intervals is odd, integrates first interval via
        trapezoidal method).

        Args:
            x_val: Sequence of function's parameters values.
            y_val: Sequence of function's values at given x_val points.
            y_err: Sequence of statistical errors of function's values
              at given x_val points.
            parameters: Integration parameters (including specific
              to integration method).

        Returns:
            Tuple (int_val, int_err) of partial integral values (like partial
            sums) and their estimated statistical errors respectively.
        """
        x_val = np.array([i for i in x_val])
        y_val = np.array([i for i in y_val])
        if len(x_val) != len(y_val):
            raise ValueError(f"Array with x-values should have the same "
                             f"length as array with y-values (received "
                             f"len(x_val)={len(x_val)}, "
                             f"len(y_val)={len(y_val)}).")
        if y_err is not None:
            y_err = np.array([i for i in y_err])
            if len(x_val) != len(y_err):
                raise ValueError(f"Array with y-arrors, if present, should "
                                 f"have the same length as y-values "
                                 f"(received len(x_val)={len(x_val)}, "
                                 f"len(y_err)={len(y_err)}).")
        else:
            y_err = np.zeros(len(x_val))

        class IntegrationToken:
            """Structure to index numerical integration primary intervals."""

            def __init__(self, left_idx: int, right_index: int) -> None:
                self.left_idx = left_idx
                self.right_index = right_index

        intervals_to_handle: List[IntegrationToken] = list()
        if parameters is None:
            start_idx = 2
            if len(x_val) % 2 == 0:
                intervals_to_handle.append(IntegrationToken(0, 1))
                start_idx = 3
            for i in range(start_idx, len(x_val), 2):
                intervals_to_handle.append(IntegrationToken(i - 2, i))
        else:
            x_len = len(x_val)
            x_node_idx = 0
            while x_node_idx < x_len:
                if (abs(x_val[x_node_idx] - parameters.x_node_value) <
                        parameters.x_node_value_precision):
                    break
                x_node_idx += 1
            if x_node_idx == x_len:
                raise RuntimeError(
                    f"Specified x_node_value is not found. ("
                    f"Given x_node_value = {parameters.x_node_value}; "
                    f"x_node_value_precision = "
                    f"{parameters.x_node_value_precision})")
            for i in range(x_node_idx):
                intervals_to_handle.append(IntegrationToken(i, i + 1))
            if (x_len - x_node_idx) % 2 == 1:
                for i in range(x_node_idx, x_len - 1, 2):
                    intervals_to_handle.append(IntegrationToken(i, i + 2))
            else:
                for i in range(x_node_idx, x_len - 2, 2):
                    intervals_to_handle.append(IntegrationToken(i, i + 2))
                intervals_to_handle.append(IntegrationToken(x_len - 2,
                                                            x_len - 1))

        int_val = [0.0]
        int_err = [0.00000000001]
        int_weights = [0.00000000001 + 0.0*x_val]
        partial_weights = 0.0 * x_val
        for token in intervals_to_handle:
            if token.right_index - token.left_idx == 1:
                i = token.right_index
                interpolant = LinearInterpolant.from_points(
                    (x_val[i - 1], x_val[i]), (y_val[i - 1], y_val[i]))
                step_value, step_weights = interpolant.integrate(
                    domains.GeometricRange((x_val[i - 1], x_val[i])))
                int_val.append(int_val[-1] + step_value)
                partial_weights[i - 1] += step_weights[0]
                partial_weights[i] += step_weights[1]
                int_err.append(
                    np.sqrt(np.sum((y_err * partial_weights) ** 2)))
                int_weights.append(partial_weights.copy())
            elif token.right_index - token.left_idx == 2:
                i = token.right_index
                x_val_tuple = (x_val[i - 2], x_val[i - 1], x_val[i])
                y_val_tuple = (y_val[i - 2], y_val[i - 1], y_val[i])
                interpolant = QuadraticInterpolant.from_points(x_val_tuple,
                                                               y_val_tuple)

                step_value, step_weights = interpolant.integrate(
                    domains.GeometricRange((x_val[i - 2], x_val[i - 1])))
                int_val.append(int_val[-1] + step_value)
                partial_weights[i - 2] += step_weights[0]
                partial_weights[i - 1] += step_weights[1]
                partial_weights[i] += step_weights[2]
                int_err.append(
                    np.sqrt(np.sum((y_err * partial_weights) ** 2)))
                int_weights.append(partial_weights.copy())

                step_value, step_weights = interpolant.integrate(
                    domains.GeometricRange((x_val[i - 1], x_val[i])))
                int_val.append(int_val[-1] + step_value)
                partial_weights[i - 2] += step_weights[0]
                partial_weights[i - 1] += step_weights[1]
                partial_weights[i] += step_weights[2]
                int_err.append(
                    np.sqrt(np.sum((y_err * partial_weights) ** 2)))
                int_weights.append(partial_weights.copy())
            else:
                raise RuntimeError(f"Unknown IntegrationToken: "
                                   f"token.left_idx = {token.left_idx}; "
                                   f"token.right_index = {token.right_index}.")
        return np.array(int_val), np.array(int_err), int_weights
