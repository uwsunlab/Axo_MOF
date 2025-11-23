"""Optimization module for Bayesian optimization and experiment planning."""

# Try importing optional dependencies
# These require scipy, scikit-learn, matplotlib, ternary
try:
    from .bayesian import (
        fit_gp,
        compute_ei,
        select_and_plot_next_batch,
        plot_sampling_space,
        plot_gp_heatmap
    )
    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

try:
    from .sampling import (
        build_feasible_lattice,
        initial_sampling
    )
    _SAMPLING_AVAILABLE = True
except ImportError:
    _SAMPLING_AVAILABLE = False

try:
    from .orchestrator import BayesianOptimizationLoop
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False

# Phase management modules (minimal dependencies)
from .phase_config import PhaseConfig, DEFAULT_PHASES, create_single_phase_config
from .phase_manager import PhaseManager

__all__ = [
    'PhaseConfig',
    'DEFAULT_PHASES',
    'create_single_phase_config',
    'PhaseManager'
]

# Add optional exports if available
if _BAYESIAN_AVAILABLE:
    __all__.extend([
        'fit_gp',
        'compute_ei',
        'select_and_plot_next_batch',
        'plot_sampling_space',
        'plot_gp_heatmap'
    ])

if _SAMPLING_AVAILABLE:
    __all__.extend([
        'build_feasible_lattice',
        'initial_sampling'
    ])

if _ORCHESTRATOR_AVAILABLE:
    __all__.append('BayesianOptimizationLoop')
