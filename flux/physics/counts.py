"""Exact 1D and 2D marginal range counts for the NPEET screen."""

from contextlib import contextmanager

from numba import njit
import numpy as np
from scipy.special import digamma


@njit(cache=True)
def _bounds(sorted_values, center, radius):
    # Compare differences as the max-norm metric does. Searching for the
    # floating-point endpoints center ± radius can round to a different count.
    low, high = 0, len(sorted_values)
    while low < high:
        middle = (low + high) // 2
        if center - sorted_values[middle] > radius:
            low = middle + 1
        else:
            high = middle
    first = low
    high = len(sorted_values)
    while low < high:
        middle = (low + high) // 2
        if sorted_values[middle] - center > radius:
            high = middle
        else:
            low = middle + 1
    return first, low


@njit(cache=True)
def count_intervals(points, radii):
    """Count all points in each inclusive 1D query interval."""
    sorted_points = np.sort(points)
    counts = np.empty(len(points), dtype=np.int64)
    for row in range(len(points)):
        start, end = _bounds(sorted_points, points[row], radii[row])
        counts[row] = end - start
    return counts


@njit(cache=True)
def _prefix_count(tree, end):
    count = 0
    while end:
        count += tree[end]
        end -= end & -end
    return count


@njit(cache=True)
def count_rectangles(points, radii):
    """Count max-norm neighbors using a sweep and a Fenwick tree of y ranks."""
    n = len(points)
    order_x = np.argsort(points[:, 0])
    order_y = np.argsort(points[:, 1])
    sorted_x = points[order_x, 0]
    sorted_y = points[order_y, 1]
    rank_y = np.empty(n, dtype=np.int64)
    for rank in range(n):
        rank_y[order_y[rank]] = rank

    heads = np.full(n + 1, -1, dtype=np.int64)
    next_event = np.empty(n * 2, dtype=np.int64)
    low_y = np.empty(n, dtype=np.int64)
    high_y = np.empty(n, dtype=np.int64)
    for row in range(n):
        start, end = _bounds(sorted_x, points[row, 0], radii[row])
        low_y[row], high_y[row] = _bounds(sorted_y, points[row, 1], radii[row])
        # Each row adds the y-interval count at its upper x boundary and
        # subtracts the count below its lower x boundary.
        lower_event = 2 * row
        upper_event = lower_event + 1
        next_event[lower_event] = heads[start]
        heads[start] = lower_event
        next_event[upper_event] = heads[end]
        heads[end] = upper_event

    tree = np.zeros(n + 1, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    for end_x in range(n + 1):
        if end_x:
            rank = rank_y[order_x[end_x - 1]] + 1
            while rank <= n:
                tree[rank] += 1
                rank += rank & -rank
        event = heads[end_x]
        while event >= 0:
            row = event // 2
            number = _prefix_count(tree, high_y[row]) - _prefix_count(tree, low_y[row])
            if event % 2:
                counts[row] += number
            else:
                counts[row] -= number
            event = next_event[event]
    return counts


@contextmanager
def accelerate_marginal_counts(estimator):
    """Use identical marginal counts, preserving NPEET's joint queries and RNG."""
    original = estimator.avgdigamma

    def avgdigamma(points, distances):
        radii = distances - 1e-15
        if np.min(radii) >= 0:
            if points.shape[1] == 1:
                return np.mean(digamma(count_intervals(points[:, 0], radii)))
            if points.shape[1] == 2:
                return np.mean(digamma(count_rectangles(points, radii)))
        return original(points, distances)

    estimator.avgdigamma = avgdigamma
    try:
        yield
    finally:
        estimator.avgdigamma = original
