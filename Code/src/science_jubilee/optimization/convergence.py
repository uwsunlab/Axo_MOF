"""
Convergence criteria for Bayesian optimization.

Provides multiple stopping criteria to determine when optimization has converged.
"""

from typing import List, Optional, Tuple

import numpy as np


def check_ei_convergence(
    ei_history: List[float],
    threshold: float = 0.001,
    window: int = 3
) -> Tuple[bool, str]:
    """
    Check if maximum Expected Improvement has plateaued.

    Parameters
    ----------
    ei_history : list of float
        History of maximum EI values per iteration
    threshold : float, default=0.001
        Convergence threshold
    window : int, default=3
        Number of recent iterations to check

    Returns
    -------
    converged : bool
        True if converged
    reason : str
        Description of convergence status
    """
    if len(ei_history) < window:
        return False, "Insufficient data"

    recent_ei = ei_history[-window:]
    max_ei = max(recent_ei)

    if max_ei < threshold:
        return True, f"Maximum EI ({max_ei:.6e}) below threshold ({threshold:.6e})"

    # Check if EI is decreasing
    is_decreasing = all(recent_ei[i] >= recent_ei[i+1] for i in range(len(recent_ei)-1))
    if is_decreasing and max_ei < 10 * threshold:
        return True, f"EI decreasing and below 10× threshold (max EI: {max_ei:.6e})"

    return False, f"EI not converged (max EI: {max_ei:.6e})"


def check_yield_plateau(
    y_history: np.ndarray,
    window: int = 5,
    threshold: float = 0.005
) -> Tuple[bool, str]:
    """
    Check if best yield hasn't improved in last N iterations.

    Parameters
    ----------
    y_history : np.ndarray
        All observed yields
    window : int, default=5
        Number of recent iterations to check
    threshold : float, default=0.005
        Minimum improvement threshold

    Returns
    -------
    converged : bool
        True if converged
    reason : str
        Description of convergence status
    """
    if len(y_history) < window:
        return False, "Insufficient data"

    # Get best yields over time
    best_yields = np.maximum.accumulate(y_history)
    recent_best = best_yields[-window:]

    # Check if best yield has improved
    improvement = recent_best[-1] - recent_best[0]
    rel_improvement = improvement / max(abs(recent_best[0]), 1e-9)

    if abs(rel_improvement) < threshold:
        return True, f"Yield plateau: improvement={improvement:.6f} < threshold ({threshold})"

    return False, f"Yield still improving (improvement: {improvement:.6f})"


def check_max_iterations(
    iteration: int,
    max_iterations: int
) -> Tuple[bool, str]:
    """
    Check if maximum number of iterations reached.

    Parameters
    ----------
    iteration : int
        Current iteration number
    max_iterations : int
        Maximum allowed iterations

    Returns
    -------
    converged : bool
        True if limit reached
    reason : str
        Description
    """
    if iteration >= max_iterations:
        return True, f"Maximum iterations reached ({max_iterations})"
    return False, f"Iterations: {iteration}/{max_iterations}"


def check_exploration_exhausted(
    X: np.ndarray,
    lattice_size: int,
    coverage_threshold: float = 0.10
) -> Tuple[bool, str]:
    """
    Check if search space has been sufficiently explored.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 3)
        Sampled compositions
    lattice_size : int
        Total number of points in feasible lattice
    coverage_threshold : float, default=0.10
        Fraction of space to consider "explored"

    Returns
    -------
    converged : bool
        True if space sufficiently explored
    reason : str
        Description
    """
    coverage = len(X) / lattice_size
    if coverage >= coverage_threshold:
        return True, f"Search space coverage: {coverage:.1%} >= {coverage_threshold:.1%}"
    return False, f"Search space coverage: {coverage:.1%}"


def suggest_stopping(
    X: np.ndarray,
    y: np.ndarray,
    iteration: int,
    max_iterations: int = 20,
    ei_history: Optional[List[float]] = None,
    lattice_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Comprehensive stopping criteria evaluation.

    Checks multiple convergence criteria and returns True if any are met.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 3)
        Sampled compositions
    y : np.ndarray, shape (n_samples,)
        Observed yields
    iteration : int
        Current iteration number
    max_iterations : int, default=20
        Maximum allowed iterations
    ei_history : list of float, optional
        History of maximum EI values
    lattice_size : int, optional
        Total lattice size for coverage calculation

    Returns
    -------
    should_stop : bool
        True if optimization should stop
    reason : str
        Convergence reason (or status if not converged)
    """
    # Check max iterations first (hard limit)
    stop, reason = check_max_iterations(iteration, max_iterations)
    if stop:
        return True, reason

    # Check EI convergence
    if ei_history is not None and len(ei_history) > 0:
        stop, reason = check_ei_convergence(ei_history, threshold=0.001, window=3)
        if stop:
            return True, reason

    # Check yield plateau
    if len(y) >= 5:
        stop, reason = check_yield_plateau(y, window=5, threshold=0.005)
        if stop:
            return True, reason

    # Check exploration coverage
    if lattice_size is not None:
        stop, reason = check_exploration_exhausted(X, lattice_size, coverage_threshold=0.10)
        if stop:
            return True, reason

    # Not converged
    status_parts = []
    status_parts.append(f"Iter: {iteration}/{max_iterations}")
    if ei_history and len(ei_history) > 0:
        status_parts.append(f"EI: {ei_history[-1]:.4e}")
    status_parts.append(f"Best yield: {np.max(y):.6f}")

    return False, " | ".join(status_parts)


def check_stagnation(
    y_history: np.ndarray,
    window: int = 10,
    std_threshold: float = 0.001
) -> Tuple[bool, str]:
    """
    Check if yields have stagnated (very low variance in recent samples).

    Parameters
    ----------
    y_history : np.ndarray
        All observed yields
    window : int, default=10
        Number of recent samples to check
    std_threshold : float, default=0.001
        Maximum allowed std deviation

    Returns
    -------
    stagnated : bool
        True if stagnated
    reason : str
        Description
    """
    if len(y_history) < window:
        return False, "Insufficient data"

    recent_yields = y_history[-window:]
    std_dev = np.std(recent_yields)

    if std_dev < std_threshold:
        return True, f"Yield variance stagnated (std={std_dev:.6f} < {std_threshold})"

    return False, f"Yield variance: {std_dev:.6f}"
