"""Module contains functions for autocorrelations' analysis.

Currently only analysis of autocorrelations of one parameter is implemented.
"""

import itertools
from typing import Sequence, List, Optional, Union, Callable, Tuple

import numpy as np


def int_log_range(_min: Union[int, float], _max: Union[int, float],
                  factor: float) -> List[int]:
    """Return list of ints that approximates geometric series.

    Args:
        _min: Floor of the resulting sequence (required _min >= 1).
        _max: Ceiling of the resulting sequence (required _max > _min).
        factor: Specifies step of the geometric series (required factor > 1).

    Returns:
        Resulting list of integers forming approximately geometric series.
    """
    if _min < 1.0:
        raise ValueError(f"_min has to be not less then 1.0 "
                         f"(received _min={_min}).")
    if _max < _min:
        raise ValueError(f"_max has to be not less then _min "
                         f"(received _min={_min}, _max={_max}).")
    if factor <= 1.0:
        raise ValueError(f"factor has to be greater then 1.0 "
                         f"(received factor={factor}).")
    result = [int(_min)]
    current = float(_min)
    while current * factor < _max:
        current *= factor
        if int(current) != result[-1]:
            result.append(int(current))
    return result


def np_1dim_array_bin_copytokenizer(data: Sequence,
                                    bin_size: int) -> List[np.ndarray]:
    """Form bins by copying data from initial container.

    Resulted bins contain at least bin_size elements.

    Args:
        data: Sequence of data to tokenize.
        bin_size: Minimal size of bins.

    Returns:
        List of np.ndarrays representing bins of the input data.
    """
    if bin_size < 1:
        raise ValueError(f"bin_size has to be not less then 1 "
                         f"(received bin_size={bin_size}).")
    if len(data) < bin_size:
        raise ValueError(f"data has to have length not less then bin_size "
                         f"(received len(data)={len(data)}, "
                         f"bin_size={bin_size}).")
    nbins = len(data) // bin_size
    bin_sizes = [bin_size for _ in range(nbins)]
    residual_size = len(data) - nbins * bin_size
    idx = 0
    while residual_size > 0:
        bin_sizes[idx] += 1
        residual_size -= 1
        idx = (idx + 1) % nbins
    bin_border_indices = [0] + list(itertools.accumulate(bin_sizes))
    return [np.array(data[bin_border_indices[i]: bin_border_indices[i + 1]])
            for i in range(nbins)]


def np_1dim_array_jackknife_copytokenizer(data: Sequence, bin_size: int):
    """Generates (via yield) samples for jackknife analysis.

    Works with copy of the initial data.
    Skipped bins contain at least bin_size elements.

    Args:
        data: Sequence of data to tokenize.
        bin_size: Minimal size of bins for removal to acquire tesulting
          jackknife tokens.

    Returns:
        Sequence of np.ndarrays representing jackknife samples of the data.
    """
    if bin_size < 1:
        raise ValueError(f"bin_size has to be not less then 1 "
                         f"(received bin_size={bin_size}).")
    if len(data) < 2 * bin_size:
        raise ValueError(f"data has to have length at least 2*bin_size "
                         f"(received len(data)={len(data)}, "
                         f"2*bin_size={2 * bin_size}).")
    nbins = len(data) // bin_size
    bins = np_1dim_array_bin_copytokenizer(data, bin_size)
    for i in range(nbins):
        result = []
        for j in range(nbins):
            if j == i:
                continue
            for item in bins[j]:
                result.append(item)
        yield np.array(result)


def mean(data: Sequence) -> float:
    """Calculates mean value of a sequence.

    Args:
        data: Sequence to estimate mean on.

    Returns:
        Estimated mean value.
    """
    if len(data) == 0:
        raise ValueError("Cannot calculate average of an empty sequence.")
    _sum = 0.0
    counter = 0
    for value in data:
        _sum += value
        counter += 1
    return _sum / counter


def st_div(data: Sequence) -> float:
    """Calculates standard deviation estimator based on data sequence.

    Args:
        data: Sequence of data to perform estimation on.

    Returns:
        Estimated standard deviation of a single measurement.
    """
    if len(data) <= 1:
        raise ValueError(f"Sequence must have at least two values to "
                         f"calculate variance (given len(data)={len(data)}).")
    avg = mean(data)
    _sum = 0
    counter = 0
    for piece in data:
        _sum += (piece - avg)**2
        counter += 1
    return np.sqrt(_sum / (counter - 1))


def bootstrap_meanstd_handler(data: Sequence, bin_size: int,
                              target_function: Callable) -> Tuple:
    """Estimates mean and std of target_function on data by binning.

    Args:
        data: Sequence of data for the analysis.
        bin_size: Minimal size of the bin during analysis.
        target_function: Function taking a datasample and returning value of
          some derived observable.

    Returns:
        Tuple containing estimate of the mean value of the derived observable
        and estimated standard deviation of that mean.
    """
    bins = np_1dim_array_bin_copytokenizer(data, bin_size)
    nbins = len(bins)
    partial_results = [target_function(_bin) for _bin in bins]
    return mean(partial_results), st_div(partial_results) / np.sqrt(nbins)


def jackknife_meanstd_handler(data: Sequence, bin_size: int,
                              target_function: Callable) -> Tuple:
    """Estimates mean and std of target_function on data via jackknife bins.

    Args:
        data: Sequence of data for the analysis.
        bin_size: Minimal size of removed bin during jackknife analysis.
        target_function: Function taking a datasample and returning value of
          some derived observable.

    Returns:
        Tuple containing estimate of the mean value of the derived observable
        and estimated standard deviation of that mean.
    """
    bins = np_1dim_array_jackknife_copytokenizer(data, bin_size)
    partial_results = [target_function(_bin) for _bin in bins]
    nbins = len(partial_results)
    return (mean(partial_results),
            st_div(partial_results) * (nbins - 1) / np.sqrt(nbins))


class AutocorrCalculator:
    """Class for estimating autocorrelations within datasample."""

    def _average(self, data: Sequence) -> float:
        """Calculates mean value of datasample."""
        if len(data) == 0:
            raise ValueError("Cannot calculate average of an empty sequence.")
        result = 0.0
        for value in data:
            result += value
        return result / len(data)

    def _variance(self, data: Sequence) -> float:
        """Estimates variance based on datasample."""
        if len(data) <= 1:
            raise ValueError(f"Sequence must have at least two values to "
                             f"calculate variance "
                             f"(given len(data)={len(data)}).")
        mean = self._average(data)
        result = 0.0
        for value in data:
            result += (value - mean)**2
        return result / (len(data) - 1)

    def _cov_at_distance(self, data: Sequence, diff: int,
                         _mean: Optional[float] = None) -> float:
        """Estimates covariance of points at distance diff."""
        if diff >= len(data):
            raise ValueError(f"To calculate covariance at some distance this "
                             f"distance has to be less then sample length "
                             f"(received diff={diff}, len(data)={len(data)}).")
        if _mean is None:
            _mean = self._average(data)
        result = 0.0
        for i in range(len(data) - diff):
            result += (data[i] - _mean) * (data[i + diff] - _mean)
        return result / (len(data) - diff)

    def get_naive_autocorr_time(self, data: Sequence) -> Optional[float]:
        """Estimates autocorrelations' time. Assumes single autocorr time.

        Estimation is based on autocorrelations stronger then exr(-1).

        Args:
            data: Sequence of data to analyse autocorrelations on.

        Returns:
            Returns estimated integrated autocorrelation value if the analysis
            was successful. Else returns None.
        """
        if len(data) < 10:
            raise ValueError(f"Data sample has to contain at least 10 elements "
                             f"in order to calculate autocorrelations "
                             f"(received len(data)={len(data)}).")
        avg = self._average(data)
        var = self._variance(data)
        diff_to_test = int_log_range(1, len(data) // 3, 1.2)
        i = 0
        while i < len(diff_to_test):
            cov = self._cov_at_distance(data, diff_to_test[i], avg)
            if cov < 0.3 * var:
                break
            i += 1
        if i == len(diff_to_test):
            return None
        t_upper = min(len(data), 2 * diff_to_test[i])
        diff_to_calculate = [0]
        diff_step = max(1, t_upper // 50)
        while diff_to_calculate[-1] < t_upper:
            diff_to_calculate.append(diff_to_calculate[-1] + diff_step)
        corr_at_diff = np.array([self._cov_at_distance(data, diff, avg)
                                 for diff in diff_to_calculate]) / var

        t_corr = 0
        while t_corr < len(corr_at_diff) and corr_at_diff[t_corr] > 0.368:
            t_corr += 1
        if t_corr == len(corr_at_diff):
            raise RuntimeError("Something went wrong.")
        if t_corr <= 1:
            tau_corr = 0.0
        else:
            tau_corr = 0.5
            for i in range(1, t_corr - 1):
                tau_corr += corr_at_diff[i]
            tau_corr += corr_at_diff[t_corr - 1] / 2
            tau_corr += ((corr_at_diff[t_corr - 1]**2 - 0.135)
                         / (corr_at_diff[t_corr - 1] - corr_at_diff[t_corr])
                         / 2.0)
            tau_corr /= 0.632
            tau_corr *= diff_step
        return tau_corr

    def get_integrated_autocorr_time(
        self,
        data: Sequence,
        threshold: Optional[float] = None
    ) -> Optional[float]:
        """Estimates integrated autocorrelations' time.

        Rough result for small samples and/or high thresholds.

        Args:
            data: Sequence of data to analyse autocorrelations on.
            threshold: Integration of correlation function is performed up to
              (corr(delta_t) < threshold) point. If present,
               has to be positive.

        Returns:
            Returns estimated integrated autocorrelation value if the analysis
            was successful. Else returns None.
        """
        if len(data) < 10:
            raise ValueError(f"Data sample has to contain at least 10 elements"
                             f" in order to calculate autocorrelations "
                             f"(received len(data)={len(data)}).")
        if threshold is None:
            threshold = 10 / np.sqrt(len(data))
            if threshold > 0.368:
                threshold = 0.368
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(f"Threshold should lie within (0.0, 1.0), "
                             f" received threshold = {threshold}.")
        avg = self._average(data)
        var = self._variance(data)
        diff_to_test = int_log_range(1, len(data) // 3, 1.2)
        i = 0
        while i < len(diff_to_test):
            cov = self._cov_at_distance(data, diff_to_test[i], avg)
            if cov < threshold * var:
                break
            i += 1
        if i == len(diff_to_test):
            return None

        t_upper = diff_to_test[i]
        # TODO: Place setting of algorithm's constants into constructor.
        diff_factor = 1.05
        diff_to_calculate = [0] + int_log_range(1, 1.07 * (t_upper + 2),
                                                diff_factor)
        corr_at_diff = np.array([self._cov_at_distance(data, diff, avg)
                                 for diff in diff_to_calculate]) / var
        t_corr = 0
        while t_corr < len(corr_at_diff) and corr_at_diff[t_corr] > threshold:
            t_corr += 1
        if t_corr == len(corr_at_diff):
            raise RuntimeError("Something went wrong.")
        # TODO: Use an integrator from corresponding module.
        tau_corr = 0.0
        for i in range(1, t_corr - 1):
            tau_corr += (0.5 * (corr_at_diff[i] + corr_at_diff[i - 1])
                         * (diff_to_calculate[i] - diff_to_calculate[i - 1]))
        return tau_corr

    def get_autocorr_normed(self, data: Sequence,
                            _max: Optional[int] = None,
                            step: int = 1) -> List:
        """Estimates correlation of elements at distance step.

        Args:
            data: Sequence of data to analyse autocorrelations on.
            _max: Autocorrelations value is performed up to _max separation
              in the sequence of data. Has to be positive and do
              not exceed len(data) // 3 (otherwise the estimation would be
              very unreliable).
            step: Step at which separation of the data for
              autocorrelation analysis is considered.

        Returns:
            List containing estimated autocorrelations at distances from
            range(0, _max, step).
        """
        if len(data) < 10:
            raise ValueError(f"Data sample has to contain at least 10 elements"
                             f" in order to calculate autocorrelations "
                             f"(received len(data)={len(data)}).")
        if _max is None:
            _max = len(data) // 3
        if _max < 0:
            raise ValueError(f"_max if present should not be negative "
                             f"(received _max={_max}).")
        if _max > len(data) // 3:
            raise ValueError(f"_max greater then len(data)//3 should "
                             f"not be used due to unreliable results "
                             f"(received len(data)={len(data)}, "
                             f"_max={_max}).")
        if step < 1:
            raise ValueError(f"Step should not be less then 1 "
                             f"(received step={step}).")
        _mean = self._average(data)
        cov_at_diff = np.array(
            [self._cov_at_distance(data, diff, _mean=_mean)
             for diff in range(0, min(len(data) // 3, _max), step)])
        return cov_at_diff / cov_at_diff[0]
