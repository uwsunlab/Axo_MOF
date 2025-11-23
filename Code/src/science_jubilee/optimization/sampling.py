"""
Initial sampling strategies for design of experiments (DOE) in ternary composition space.

Provides maximin distance sampling to ensure good coverage of the feasible region
before starting Bayesian optimization.
"""

from typing import Optional

import numpy as np


def build_feasible_lattice(
    total_ml: float = 8.0,
    step_ml: float = 0.1,
    min_co: float = 0.1,
    min_mim: float = 0.1,
    min_tea: float = 0.0
) -> np.ndarray:
    """
    Generate feasible lattice in ternary composition space.

    Constraint: Co + MIM + TEA = total_ml

    Parameters
    ----------
    total_ml : float, default=8.0
        Total mixture volume (mL)
    step_ml : float, default=0.1
        Grid resolution (mL)
    min_co : float, default=0.1
        Minimum Co volume (mL)
    min_mim : float, default=0.1
        Minimum MIM volume (mL)
    min_tea : float, default=0.0
        Minimum TEA volume (mL)

    Returns
    -------
    np.ndarray, shape (n_points, 3)
        Array of [Co, MIM, TEA] compositions on grid
    """
    grid_vals = np.round(np.arange(0.0, total_ml + 1e-9, step_ml), 10)
    pts = []

    for co in grid_vals:
        for mim in grid_vals:
            tea = total_ml - co - mim

            # TEA must be non-negative
            if tea < -1e-9:
                continue

            tea = np.round(tea, 10)

            # Ensure tea is on grid (multiple of step)
            if abs(tea / step_ml - np.round(tea / step_ml)) > 1e-9:
                continue

            # Enforce minimum constraints
            if co >= min_co - 1e-9 and mim >= min_mim - 1e-9 and tea >= min_tea - 1e-9:
                pts.append((co, mim, tea))

    return np.array(pts, dtype=float)


def maximin_scaled(
    grid_ml: np.ndarray,
    n_samples: int = 5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Greedy maximin sampling in normalized simplex coordinates.

    Selects points to maximize the minimum pairwise distance, ensuring
    good space-filling coverage.

    Parameters
    ----------
    grid_ml : np.ndarray, shape (n_candidates, 3)
        Candidate points [Co, MIM, TEA] in mL
    n_samples : int, default=5
        Number of samples to select
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray, shape (n_samples, 3)
        Selected samples in mL
    """
    if random_state is not None:
        np.random.seed(random_state)

    grid_ml = np.asarray(grid_ml, float)
    total = grid_ml.sum(axis=1, keepdims=True)

    # Normalize to simplex (divide by total)
    grid_normalized = grid_ml / total

    n_candidates = len(grid_normalized)
    if n_samples >= n_candidates:
        return grid_ml  # Return all points if requested more than available

    # Start with random point
    selected_idx = [np.random.randint(n_candidates)]

    # Greedy selection: maximize minimum distance to selected points
    for _ in range(n_samples - 1):
        # Compute distances from all candidates to selected points
        distances = []
        for i in range(n_candidates):
            if i in selected_idx:
                distances.append(-np.inf)  # Exclude already selected
            else:
                # Minimum distance to any selected point
                min_dist = min(
                    np.linalg.norm(grid_normalized[i] - grid_normalized[j])
                    for j in selected_idx
                )
                distances.append(min_dist)

        # Select point with maximum minimum distance
        next_idx = int(np.argmax(distances))
        selected_idx.append(next_idx)

    return grid_ml[selected_idx]


def initial_sampling(
    total_ml: float = 8.0,
    step_ml: float = 0.1,
    n_init: int = 5,
    min_co: float = 0.1,
    min_mim: float = 0.1,
    min_tea: float = 0.0,
    random_state: Optional[int] = 42
) -> np.ndarray:
    """
    Generate initial samples using maximin distance criterion.

    Parameters
    ----------
    total_ml : float, default=8.0
        Total mixture volume (mL)
    step_ml : float, default=0.1
        Grid resolution (mL)
    n_init : int, default=5
        Number of initial samples
    min_co : float, default=0.1
        Minimum Co volume (mL)
    min_mim : float, default=0.1
        Minimum MIM volume (mL)
    min_tea : float, default=0.0
        Minimum TEA volume (mL)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray, shape (n_init, 3)
        Initial sample compositions [Co, MIM, TEA]
    """
    # Build feasible lattice
    lattice = build_feasible_lattice(
        total_ml=total_ml,
        step_ml=step_ml,
        min_co=min_co,
        min_mim=min_mim,
        min_tea=min_tea
    )

    # Apply maximin sampling
    samples = maximin_scaled(
        lattice,
        n_samples=n_init,
        random_state=random_state
    )

    return samples


def validate_composition(
    co: float,
    mim: float,
    tea: float,
    total_ml: float = 8.0,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that composition sums to total_ml.

    Parameters
    ----------
    co : float
        Co volume (mL)
    mim : float
        MIM volume (mL)
    tea : float
        TEA volume (mL)
    total_ml : float, default=8.0
        Expected total volume
    tolerance : float, default=1e-6
        Numerical tolerance

    Returns
    -------
    bool
        True if composition is valid
    """
    total = co + mim + tea
    return abs(total - total_ml) < tolerance and co >= 0 and mim >= 0 and tea >= 0
