"""Contains useful class GeometricRange for dim=1 continuous domain."""

from __future__ import annotations
from typing import Optional, Tuple


class GeometricRange:
    """Represents a domain of continuous one dimensional function."""

    _empty: bool = True
    left: float
    right: float

    def __init__(self, from_to: Optional[Tuple[float, float]] = None) -> None:
        """Creates a domain (closed) within given boundaries.

        Creates empty interval if called without arguments or if left endpoint
        is greater than right endpoint.

        Args:
            from_to: Tuple (_from, _to) with domain's boundaries.
        """
        if from_to is None:
            self._empty = True
            return
        left = from_to[0]
        right = from_to[1]
        if left > right:
            self._empty = True
            return
        self.left = left
        self.right = right
        self._empty = False

    def intersection_with(self, other: GeometricRange) -> GeometricRange:
        """Returns intersection with other domain.

        Args:
            other: GeometricRange to calculate intersection with.

        Returns:
            Resulting intersection as GeometricRange.
        """
        if self.empty() or other.empty():
            return GeometricRange()
        return GeometricRange((max(self.left, other.left),
                               min(self.right, other.right)))

    def contains_point(self, value: float) -> bool:
        """Returns bool value representing if the domain contains a point.

        Args:
            value: Floating point value to check.

        Returns:
            Bool result representing if value is lying within the domain.
        """
        if self._empty:
            return False
        return self.left <= value <= self.right

    def contains_geometric_range(self, other: GeometricRange) -> bool:
        """Returns True if the domain contains another domain.

        Args:
            other: GeometricRange to check.

        Returns:
            Bool result representing if the other GeometricRange
            is lying within this domain's boundaries.
        """
        if self.empty() or other.empty():
            return False
        return (self.left <= other.left <= self.right
                and self.left <= other.right <= self.right)

    def empty(self) -> bool:
        """Returns bool values representing if the domain is empty or not."""
        return self._empty
