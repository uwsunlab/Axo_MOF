"""
Bayesian optimization module for MOF synthesis parameter optimization.

This module provides Gaussian Process modeling, Expected Improvement acquisition,
and ternary composition space visualization for optimizing ZIF synthesis conditions.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import ternary
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


def unit_scale(
    X: np.ndarray,
    bounds: Sequence[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale input variables to [0, 1]^d unit hypercube.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input variables in physical units
    bounds : list of (low, high) tuples
        Physical bounds for each variable

    Returns
    -------
    X_scaled : np.ndarray
        Scaled inputs in [0, 1]
    lo : np.ndarray
        Lower bounds
    hi : np.ndarray
        Upper bounds
    """
    X = np.asarray(X, float)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    w = hi - lo
    if np.any(w <= 0):
        raise ValueError("Invalid bounds: upper must be > lower.")
    return (X - lo) / w, lo, hi


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    normalise_y: bool = True,
    n_restarts: int = 20,
    random_state: int = 42
) -> Tuple[Dict, Dict]:
    """
    Fit a Gaussian Process (GP) model using a Matern(ν=2.5, ARD) kernel.

    Hyperparameters are optimized by maximizing the marginal likelihood
    with multiple random restarts.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input variables (e.g., [Co, MIM, TEA] in mL)
    y : np.ndarray, shape (n_samples,)
        Observed scalar responses (yields)
    bounds : list of (low, high) tuples
        Physical range for each variable; used for unit scaling to [0, 1]
    normalise_y : bool, default=True
        Whether to z-score y for numerical stability
    n_restarts : int, default=20
        Number of random restarts for hyperparameter optimization
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    fit : dict
        Contains:
        - 'gp': fitted GaussianProcessRegressor
        - 'Xu': scaled inputs
        - 'yproc': processed y values
        - 'bounds': original bounds
        - 'lo', 'hi': bound arrays
        - 'yscaler': StandardScaler object (or None)
    info : dict
        Diagnostic information:
        - 'nll': negative log-likelihood
        - 'kernel_': final kernel string
        - 'y_mean', 'y_std': normalization parameters
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()

    # Shape validation
    if X.ndim != 2 or len(bounds) != X.shape[1] or len(y) != len(X):
        raise ValueError("Shape mismatch between X, y, and bounds.")

    # Scale inputs to [0, 1]
    Xu, lo, hi = unit_scale(X, bounds)

    # Normalize y
    if normalise_y:
        ys = StandardScaler(with_mean=True, with_std=True)
        yp = ys.fit_transform(y.reshape(-1, 1)).ravel()
    else:
        ys = None
        yp = y.copy()

    # Define kernel: Constant * Matern(ν=2.5, ARD) + WhiteKernel
    k = (
        C(1.0, (0.1, 10))
        * Matern(
            length_scale=[0.5] * X.shape[1],
            length_scale_bounds=(0.05, 1.0),
            nu=2.5
        )
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
    )

    # Fit GP
    gp = GaussianProcessRegressor(
        kernel=k,
        normalize_y=False,  # We already normalized
        n_restarts_optimizer=int(n_restarts),
        random_state=int(random_state)
    )
    gp.fit(Xu, yp)

    # Diagnostics
    info = {
        "nll": float(-gp.log_marginal_likelihood_value_),
        "kernel_": str(gp.kernel_),
        "y_mean": None if ys is None else float(ys.mean_[0]),
        "y_std": None if ys is None else float(ys.scale_[0]),
    }

    fit = {
        "gp": gp,
        "Xu": Xu,
        "yproc": yp,
        "bounds": bounds,
        "lo": lo,
        "hi": hi,
        "yscaler": ys
    }

    return fit, info


def compute_ei(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Compute Expected Improvement (EI) for maximization.

    EI = (mu - y_best - xi) * Φ(z) + sigma * φ(z)
    where z = (mu - y_best - xi) / sigma

    Parameters
    ----------
    mu : np.ndarray
        Predictive mean
    sigma : np.ndarray
        Predictive standard deviation
    y_best : float
        Best observed value so far
    xi : float, default=0.01
        Exploration parameter

    Returns
    -------
    ei : np.ndarray
        Expected improvement values
    """
    mu = np.asarray(mu, float).ravel()
    sigma = np.asarray(sigma, float).ravel()

    improv = mu - (float(y_best) + float(xi))
    z = np.zeros_like(mu)
    mask = sigma > 0
    z[mask] = improv[mask] / sigma[mask]

    ei = np.zeros_like(mu)
    ei[mask] = improv[mask] * norm.cdf(z[mask]) + sigma[mask] * norm.pdf(z[mask])
    ei[~mask] = 0.0
    ei[ei < 0] = 0.0  # Numerical safety

    return ei


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
        Array of [Co, MIM, TEA] compositions
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


def greedy_ei_with_radius(
    lattice: np.ndarray,
    Xu: np.ndarray,
    ei: np.ndarray,
    X_seen: Optional[np.ndarray] = None,
    step_ml: float = 0.1,
    k: int = 5,
    radius_uc: float = 0.18
) -> List[int]:
    """
    Greedy top-k selection with local penalization to diversify batch.

    After each pick, zero EI within an L2 ball of radius_uc in unit space.

    Parameters
    ----------
    lattice : np.ndarray, shape (n, 3)
        Lattice points in physical units
    Xu : np.ndarray, shape (n, 3)
        Lattice points in unit space [0,1]³
    ei : np.ndarray, shape (n,)
        Expected improvement values
    X_seen : np.ndarray, optional
        Previously evaluated points to exclude
    step_ml : float, default=0.1
        Grid step size
    k : int, default=5
        Number of points to select
    radius_uc : float, default=0.18
        Exclusion radius in unit space

    Returns
    -------
    list of int
        Indices of selected points
    """
    ei_work = ei.copy()

    # Mask historical points
    if X_seen is not None and len(X_seen) > 0:
        Xs = np.round(np.asarray(X_seen, float) / step_ml) * step_ml
        seen = {tuple(r) for r in Xs}
        for i, row in enumerate(np.round(lattice / step_ml) * step_ml):
            if tuple(row) in seen:
                ei_work[i] = 0.0

    picked_idx = []
    for _ in range(int(k)):
        i = int(np.argmax(ei_work))
        if not np.isfinite(ei_work[i]) or ei_work[i] <= 0:
            break
        picked_idx.append(i)

        # Zero out EI within radius
        di = np.linalg.norm(Xu - Xu[i], axis=1)
        ei_work[di <= float(radius_uc)] = 0.0

    return picked_idx


def y_best_from_fit(fit: Dict) -> float:
    """
    Extract best observed y value from fit dictionary.

    Parameters
    ----------
    fit : dict
        Fit dictionary from fit_gp()

    Returns
    -------
    float
        Maximum observed y value (on original scale)
    """
    yp = np.asarray(fit["yproc"], float)
    s = fit["yscaler"]
    if s is None:
        return float(yp.max())
    else:
        return float(s.inverse_transform(yp.reshape(-1, 1)).ravel().max())


def lattice_mu_sigma_ei(
    fit: Dict,
    bounds: Sequence[Tuple[float, float]],
    total_ml: float = 8.0,
    step_ml: float = 0.1,
    xi: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lattice, predictive mean/std, and EI (reusable for plotting and selection).

    Parameters
    ----------
    fit : dict
        Fitted GP model dictionary
    bounds : list of (low, high) tuples
        Physical bounds
    total_ml : float, default=8.0
        Total mixture volume
    step_ml : float, default=0.1
        Grid resolution
    xi : float, default=0.01
        EI exploration parameter

    Returns
    -------
    lattice : np.ndarray, shape (n, 3)
        Feasible points in mL
    Xu : np.ndarray, shape (n, 3)
        Unit-scaled inputs
    mu : np.ndarray, shape (n,)
        Predictive mean (original scale)
    sd : np.ndarray, shape (n,)
        Predictive std (original scale)
    ei : np.ndarray, shape (n,)
        Expected improvement
    """
    lattice = build_feasible_lattice(total_ml=total_ml, step_ml=step_ml)
    Xu, _, _ = unit_scale(lattice, bounds)

    mu, sd = fit["gp"].predict(Xu, return_std=True)

    # Transform back to original scale
    if fit["yscaler"] is not None:
        mu = fit["yscaler"].inverse_transform(mu.reshape(-1, 1)).ravel()
        sd = sd * float(fit["yscaler"].scale_[0])

    ei = compute_ei(mu, sd, y_best=y_best_from_fit(fit), xi=xi)

    return lattice, Xu, mu.ravel(), sd.ravel(), ei.ravel()


def ternary_to_xy_lrt(lrt_ml: np.ndarray) -> np.ndarray:
    """
    Convert ternary coordinates to Cartesian XY.

    Parameters
    ----------
    lrt_ml : np.ndarray, shape (n, 3)
        Array of [left, right, top] coordinates (Co, MIM, TEA)

    Returns
    -------
    np.ndarray, shape (n, 2)
        Cartesian XY coordinates
    """
    lrt = np.asarray(lrt_ml, float)
    x = lrt[:, 1] + 0.5 * lrt[:, 2]  # x = right + top/2
    y = (np.sqrt(3) / 2.0) * lrt[:, 2]  # y = (√3/2) * top
    return np.column_stack([x, y])


def plot_sampling_space(scale_ml: float = 8.0) -> Tuple:
    """
    Create ternary diagram base with labels and gridlines.

    Parameters
    ----------
    scale_ml : float, default=8.0
        Total mixture volume (defines simplex size)

    Returns
    -------
    tax : ternary.TernaryAxesSubplot
        Ternary axis object
    scale : int
        Scale value
    """
    s = int(round(scale_ml))
    fig, tax = ternary.figure(scale=s)
    tax.gridlines(multiple=1, color="0.8", linewidth=0.8, zorder=0)
    tax.boundary(linewidth=2.0, zorder=0)
    tax.left_corner_label("", offset=0)

    # Co²⁺ label (bottom-left vertex)
    tax.annotate(
        r"Co$^{2+}$ (mL)",
        position=(0, 1, 0),
        fontsize=14,
        textcoords="offset points",
        xytext=(-35, -60),
        ha='center', va='center'
    )

    # 2-mIm label (bottom-right vertex)
    tax.right_corner_label("", offset=0)
    tax.annotate(
        r"2-mIm (mL)",
        position=(0, 1, 0),
        fontsize=14,
        xytext=(315, -60),
        textcoords="offset points",
        ha='center', va='center'
    )

    # TEA label (top vertex)
    tax.top_corner_label("TEA (mL)", fontsize=14, offset=0.2)

    tax.ticks(axis='lbr', multiple=1, linewidth=1,
              tick_formats="%d", fontsize=12, offset=0.015)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    plt.tight_layout()

    return tax, s


def plot_initial_samples(
    tax,
    X_ml: np.ndarray,
    marker_size: float = 60,
    marker_color: str = "red"
):
    """
    Plot existing samples on ternary diagram.

    Parameters
    ----------
    tax : ternary.TernaryAxesSubplot
        Ternary axis
    X_ml : np.ndarray, shape (n, 3)
        Sample compositions [Co, MIM, TEA]
    marker_size : float, default=60
        Marker size
    marker_color : str, default="red"
        Marker color
    """
    X_ml = np.asarray(X_ml, float)
    ternary_pts = [(row[1], row[2], row[0]) for row in X_ml]  # (MIM, TEA, Co)
    for p in ternary_pts:
        tax.scatter([p], marker='o', color=marker_color, s=marker_size,
                    edgecolors=marker_color, facecolors='none')


def plot_gp_heatmap(
    tax,
    fit: Dict,
    bounds: Sequence[Tuple[float, float]],
    which: str = "mean",
    scale_ml: float = 8.0,
    step_ml: float = 0.1
):
    """
    Plot smooth GP heatmap on ternary diagram using Gouraud shading.

    Parameters
    ----------
    tax : ternary.TernaryAxesSubplot
        Ternary axis
    fit : dict
        Fitted GP model
    bounds : list of (low, high) tuples
        Physical bounds
    which : str, default="mean"
        "mean" or "std" for predictive mean or standard deviation
    scale_ml : float, default=8.0
        Total mixture volume
    step_ml : float, default=0.1
        Grid resolution
    """
    # Construct fine lattice
    lattice = build_feasible_lattice(total_ml=scale_ml, step_ml=step_ml)

    # GP prediction on lattice
    Xu, _, _ = unit_scale(lattice, bounds)
    mu, sd = fit["gp"].predict(Xu, return_std=True)

    if fit["yscaler"] is not None:
        mu = fit["yscaler"].inverse_transform(mu.reshape(-1, 1)).ravel()
        sd = sd * float(fit["yscaler"].scale_[0])

    Z = mu if which == "mean" else sd
    norm = Normalize(vmin=0.0, vmax=np.max(Z))

    # Convert to Cartesian
    lrt = lattice[:, [0, 1, 2]]
    XY = ternary_to_xy_lrt(lrt)
    tri = Triangulation(XY[:, 0], XY[:, 1])

    # Plot gradient color map
    ax = tax.get_axes()
    tpc = ax.tripcolor(tri, Z, shading="gouraud", zorder=1, norm=norm)

    # Add colorbar
    cbar = plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04, shrink=0.9, anchor=(0.0, 1))
    cbar.set_label(f"Predictive {which}")

    # Re-draw ternary boundary
    tax.boundary(linewidth=2.0, zorder=3)

    return tax


def select_and_plot_next_batch(
    X: np.ndarray,
    y: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    k: int = 5,
    xi: float = 0.01,
    total_ml: float = 8.0,
    step_ml: float = 0.1,
    radius_uc: float = 0.18,
    random_state: int = 42,
    save_plot: Optional[str] = None
) -> Tuple[np.ndarray, Dict, Tuple]:
    """
    Fit GP, compute EI, select diversified batch, and plot EI heatmap.

    Parameters
    ----------
    X : np.ndarray, shape (n, 3)
        Observed compositions
    y : np.ndarray, shape (n,)
        Observed yields
    bounds : list of (low, high) tuples
        Physical bounds
    k : int, default=5
        Batch size
    xi : float, default=0.01
        EI exploration parameter
    total_ml : float, default=8.0
        Total mixture volume
    step_ml : float, default=0.1
        Grid resolution
    radius_uc : float, default=0.18
        Diversification radius in unit space
    random_state : int, default=42
        Random seed
    save_plot : str, optional
        Path to save plot

    Returns
    -------
    X_next : np.ndarray, shape (k, 3)
        Suggested next batch
    table : dict
        Summary table with Co, MIM, TEA, mu, sigma, EI
    cache : tuple
        (fit, info, (lattice, mu, sd, ei)) for reuse
    """
    # Fit GP
    fit, info = fit_gp(X, y, bounds, normalise_y=True,
                       n_restarts=10, random_state=random_state)

    # Compute lattice + EI once
    lattice, Xu, mu, sd, ei = lattice_mu_sigma_ei(
        fit, bounds, total_ml=total_ml, step_ml=step_ml, xi=xi
    )

    # Greedy selection with diversification
    idx = greedy_ei_with_radius(lattice, Xu, ei, X_seen=X,
                                 step_ml=step_ml, k=k, radius_uc=radius_uc)
    X_next = lattice[idx] if idx else np.empty((0, 3), float)

    # Plot EI heatmap
    tax, _ = plot_sampling_space(scale_ml=total_ml)
    lrt = lattice[:, [0, 1, 2]]
    XY = ternary_to_xy_lrt(lrt)
    tri = Triangulation(XY[:, 0], XY[:, 1])
    ax = tax.get_axes()
    tpc = ax.tripcolor(tri, ei, shading="gouraud", zorder=1)
    cbar = plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04, shrink=0.9, anchor=(0.0, 1))
    cbar.set_label("Expected Improvement")
    tax.boundary(linewidth=2.0, zorder=3)

    # Overlay existing samples (red circles)
    plot_initial_samples(tax, X)

    # Overlay new suggestions (yellow stars)
    if X_next.size:
        pts = [(r[1], r[2], r[0]) for r in X_next]
        tax.scatter(pts, marker='*', s=130, edgecolor='k',
                    facecolor='yellow', linewidths=0.8, zorder=5)

    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')

    plt.tight_layout()

    # Create summary table
    if X_next.size:
        table = {
            "Co_mL": X_next[:, 0],
            "MIM_mL": X_next[:, 1],
            "TEA_mL": X_next[:, 2],
            "mu": mu[idx],
            "sigma": sd[idx],
            "EI": ei[idx]
        }
    else:
        table = {
            "Co_mL": [], "MIM_mL": [], "TEA_mL": [],
            "mu": [], "sigma": [], "EI": []
        }

    return X_next, table, (fit, info, (lattice, mu, sd, ei))


def append_batch(
    X: np.ndarray,
    y: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate arrays after a batch is executed experimentally.

    Parameters
    ----------
    X : np.ndarray, shape (n, 3)
        Existing compositions
    y : np.ndarray, shape (n,)
        Existing yields
    X_new : np.ndarray, shape (m, 3)
        New compositions
    y_new : np.ndarray, shape (m,)
        New yields

    Returns
    -------
    X_all : np.ndarray, shape (n+m, 3)
        Combined compositions
    y_all : np.ndarray, shape (n+m,)
        Combined yields
    """
    if X_new.size == 0:
        return X, y
    X_all = np.vstack([X, X_new])
    y_all = np.concatenate([y, y_new.ravel()])
    return X_all, y_all
