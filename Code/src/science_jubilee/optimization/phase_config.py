"""
Phase configuration for multi-phase Bayesian optimization.

Defines PhaseConfig dataclass and predefined phase templates for
exploration, refinement, and validation stages.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PhaseConfig:
    """
    Configuration for a single optimization phase.

    Attributes
    ----------
    name : str
        Phase identifier (e.g., "exploration", "refinement")
    description : str
        Human-readable description of the phase
    batch_size : int
        Number of samples per iteration in this phase
    xi : float
        Exploration parameter for Expected Improvement acquisition
        Higher xi = more exploration, lower xi = more exploitation
    bounds : list of (low, high) tuples, optional
        Search space bounds for [Co, MIM, TEA] in mL
        If None, bounds will be computed dynamically
    min_iterations : int, default=3
        Minimum iterations before phase can transition
    max_iterations : int, default=10
        Maximum iterations for this phase (hard limit)
    performance_threshold : float, optional
        Minimum yield required to enter this phase
        If None, no yield requirement
    convergence_ei_threshold : float, default=0.01
        EI threshold below which phase is considered converged
    """
    name: str
    description: str
    batch_size: int
    xi: float
    bounds: Optional[List[Tuple[float, float]]] = None
    min_iterations: int = 3
    max_iterations: int = 10
    performance_threshold: Optional[float] = None
    convergence_ei_threshold: float = 0.01

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.xi < 0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if self.min_iterations < 1:
            raise ValueError(f"min_iterations must be >= 1, got {self.min_iterations}")
        if self.max_iterations < self.min_iterations:
            raise ValueError(f"max_iterations ({self.max_iterations}) must be >= min_iterations ({self.min_iterations})")
        if self.performance_threshold is not None and self.performance_threshold < 0:
            raise ValueError(f"performance_threshold must be non-negative, got {self.performance_threshold}")


# Predefined phase templates for ZIF synthesis optimization

DEFAULT_PHASES = [
    PhaseConfig(
        name="exploration",
        description="Broad screening of composition space with large batches",
        batch_size=10,
        xi=0.05,  # High exploration
        bounds=[(0.0, 8.0), (0.0, 8.0), (0.0, 8.0)],  # Full space
        min_iterations=3,
        max_iterations=5,
        performance_threshold=None,  # No yield requirement to enter
        convergence_ei_threshold=0.01
    ),

    PhaseConfig(
        name="refinement",
        description="Focused optimization of promising regions with medium batches",
        batch_size=5,
        xi=0.01,  # Balanced exploration/exploitation
        bounds=None,  # Computed dynamically around best region
        min_iterations=5,
        max_iterations=10,
        performance_threshold=0.1,  # Need yield > 0.1 to enter
        convergence_ei_threshold=0.001
    ),

    PhaseConfig(
        name="validation",
        description="Fine-tuning optimal conditions with small batches",
        batch_size=2,
        xi=0.001,  # Pure exploitation
        bounds=None,  # Tight bounds around optimum
        min_iterations=2,
        max_iterations=5,
        performance_threshold=0.5,  # Need yield > 0.5 to enter
        convergence_ei_threshold=0.0005
    )
]


def create_single_phase_config(
    batch_size: int = 5,
    max_iterations: int = 20,
    xi: float = 0.01
) -> List[PhaseConfig]:
    """
    Create a single-phase configuration for backward compatibility.

    Parameters
    ----------
    batch_size : int, default=5
        Number of samples per iteration
    max_iterations : int, default=20
        Maximum iterations
    xi : float, default=0.01
        Exploration parameter

    Returns
    -------
    list of PhaseConfig
        Single-element list with one phase
    """
    return [
        PhaseConfig(
            name="optimization",
            description="Single-phase optimization",
            batch_size=batch_size,
            xi=xi,
            bounds=[(0.0, 8.0), (0.0, 8.0), (0.0, 8.0)],
            min_iterations=1,
            max_iterations=max_iterations,
            performance_threshold=None,
            convergence_ei_threshold=0.001
        )
    ]
