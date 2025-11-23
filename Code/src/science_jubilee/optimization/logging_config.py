"""
Logging configuration for Bayesian optimization orchestration.

Provides comprehensive logging to both console and file for debugging,
monitoring, and audit trail purposes.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_bo_logger(
    log_dir: str,
    log_filename: str = "bo_optimization.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Configure logger for Bayesian optimization loop.

    Creates both console and file handlers with appropriate formatting.

    Parameters
    ----------
    log_dir : str
        Directory to save log file
    log_filename : str, default="bo_optimization.log"
        Log file name
    console_level : int, default=logging.INFO
        Console logging level
    file_level : int, default=logging.DEBUG
        File logging level

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("BayesianOptimization")
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    simple_formatter = logging.Formatter(
        fmt="%(levelname)-8s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / log_filename

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Log file: {log_file}")

    return logger


def log_iteration_start(logger: logging.Logger, iteration: int, n_samples: int):
    """
    Log iteration start.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    iteration : int
        Iteration number
    n_samples : int
        Total samples so far
    """
    logger.info("=" * 70)
    logger.info(f"ITERATION {iteration} START | Total samples: {n_samples}")
    logger.info("=" * 70)


def log_gp_fit(
    logger: logging.Logger,
    iteration: int,
    nll: float,
    kernel_str: str,
    y_mean: Optional[float],
    y_std: Optional[float]
):
    """
    Log Gaussian Process fitting results.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    iteration : int
        Iteration number
    nll : float
        Negative log-likelihood
    kernel_str : str
        Final kernel string
    y_mean : float, optional
        Y normalization mean
    y_std : float, optional
        Y normalization std
    """
    logger.info(f"GP Fit Results (Iteration {iteration}):")
    logger.info(f"  Negative Log-Likelihood: {nll:.6f}")
    logger.info(f"  Final Kernel: {kernel_str}")
    if y_mean is not None and y_std is not None:
        logger.info(f"  Y Normalization: mean={y_mean:.6f}, std={y_std:.6f}")


def log_ei_statistics(
    logger: logging.Logger,
    iteration: int,
    ei_max: float,
    ei_mean: float,
    ei_std: float
):
    """
    Log Expected Improvement statistics.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    iteration : int
        Iteration number
    ei_max : float
        Maximum EI
    ei_mean : float
        Mean EI
    ei_std : float
        Std of EI
    """
    logger.info(f"EI Statistics (Iteration {iteration}):")
    logger.info(f"  Max EI: {ei_max:.6e}")
    logger.info(f"  Mean EI: {ei_mean:.6e}")
    logger.info(f"  Std EI: {ei_std:.6e}")


def log_batch_selection(logger: logging.Logger, iteration: int, X_next, ei_values):
    """
    Log selected batch.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    iteration : int
        Iteration number
    X_next : np.ndarray
        Selected compositions
    ei_values : np.ndarray
        EI values for selected points
    """
    logger.info(f"Selected Batch (Iteration {iteration}):")
    for i, (comp, ei) in enumerate(zip(X_next, ei_values)):
        logger.info(f"  {i+1}. Co={comp[0]:.1f}, MIM={comp[1]:.1f}, TEA={comp[2]:.1f} | EI={ei:.4e}")


def log_yield_extraction(
    logger: logging.Logger,
    iteration: int,
    yields: dict,
    failed_vials: list
):
    """
    Log yield extraction results.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    iteration : int
        Iteration number
    yields : dict
        Vial yields
    failed_vials : list
        List of vials with failed fits
    """
    logger.info(f"Yield Extraction (Iteration {iteration}):")
    for vial_id in sorted(yields.keys()):
        status = " (FAILED FIT)" if vial_id in failed_vials else ""
        logger.info(f"  {vial_id}: I_max = {yields[vial_id]:.6f}{status}")

    if failed_vials:
        logger.warning(f"Failed fits for vials: {', '.join(failed_vials)}")


def log_convergence(
    logger: logging.Logger,
    reason: str,
    iteration: int,
    best_yield: float,
    best_composition
):
    """
    Log convergence.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    reason : str
        Convergence reason
    iteration : int
        Final iteration number
    best_yield : float
        Best yield found
    best_composition : np.ndarray
        Optimal composition
    """
    logger.info("=" * 70)
    logger.info("CONVERGENCE ACHIEVED")
    logger.info("=" * 70)
    logger.info(f"Reason: {reason}")
    logger.info(f"Total iterations: {iteration}")
    logger.info(f"Best yield: {best_yield:.6f}")
    logger.info(f"Best composition: Co={best_composition[0]:.1f}, "
                f"MIM={best_composition[1]:.1f}, TEA={best_composition[2]:.1f}")


def log_error(logger: logging.Logger, error_message: str, exception: Optional[Exception] = None):
    """
    Log error with optional exception details.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    error_message : str
        Error description
    exception : Exception, optional
        Exception object
    """
    logger.error(error_message)
    if exception:
        logger.exception("Exception details:", exc_info=exception)
