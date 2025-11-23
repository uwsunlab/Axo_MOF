"""
User notification system for human-in-loop operations.

Provides clear, actionable instructions for vial loading, experiment monitoring,
and iteration progress.
"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def print_separator(char: str = "=", length: int = 70):
    """Print a separator line."""
    print(char * length)


def notify_vial_loading(
    vial_assignments: Dict[str, Tuple[float, float, float]],
    iteration: int
):
    """
    Display vial loading instructions to operator.

    Parameters
    ----------
    vial_assignments : dict
        Mapping: vial_id -> (Co_mL, MIM_mL, TEA_mL)
    iteration : int
        Current iteration number
    """
    print_separator()
    print(f"VIAL LOADING REQUIRED - Iteration {iteration}")
    print_separator()
    print("Prepare the following vials:\n")

    # Group by slot
    slot2_vials = {k: v for k, v in vial_assignments.items() if k.startswith('A')}
    slot5_vials = {k: v for k, v in vial_assignments.items() if k.startswith('B')}

    if slot2_vials:
        print("Slot 2:")
        for vial_id in sorted(slot2_vials.keys()):
            co, mim, tea = vial_assignments[vial_id]
            print(f"  {vial_id}: Co²⁺={co:.1f} mL, 2-mIm={mim:.1f} mL, TEA={tea:.1f} mL")
        print()

    if slot5_vials:
        print("Slot 5:")
        for vial_id in sorted(slot5_vials.keys()):
            co, mim, tea = vial_assignments[vial_id]
            print(f"  {vial_id}: Co²⁺={co:.1f} mL, 2-mIm={mim:.1f} mL, TEA={tea:.1f} mL")
        print()

    print("Checklist:")
    print("  [ ] Load clean vials into specified slots")
    print("  [ ] Ensure precursor reservoirs are filled")
    print("  [ ] Ensure solvent reservoir is filled")
    print("  [ ] Verify all tools are in parking positions")
    print()
    print_separator()


def wait_for_confirmation(
    message: str = "Press ENTER when ready...",
    timeout_seconds: Optional[int] = None
) -> bool:
    """
    Wait for operator to press ENTER.

    Parameters
    ----------
    message : str, default="Press ENTER when ready..."
        Prompt message
    timeout_seconds : int, optional
        Timeout in seconds (not implemented - waits indefinitely)

    Returns
    -------
    bool
        True if user confirmed
    """
    try:
        input(message)
        return True
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return False


def notify_experiment_start(
    experiment_name: str,
    n_vials: int,
    estimated_duration_hours: float
):
    """
    Notify user that experiment is starting.

    Parameters
    ----------
    experiment_name : str
        Experiment identifier
    n_vials : int
        Number of vials in batch
    estimated_duration_hours : float
        Estimated experiment duration (hours)
    """
    print_separator()
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print_separator()
    print(f"Number of vials: {n_vials}")
    print(f"Estimated duration: {estimated_duration_hours:.1f} hours")
    print()
    print("The system will now:")
    print("  1. Record spectrum references (dark/white)")
    print("  2. Dispense solvents to all vials")
    print("  3. Dispense metal precursor to all vials")
    print("  4. For each vial:")
    print("     - Dispense organic precursor")
    print("     - Mix solution")
    print("     - Record T0 spectrum")
    print("  5. Monitor spectra at regular intervals")
    print()
    print("Do NOT remove vials until experiment is complete!")
    print_separator()


def notify_experiment_complete(
    experiment_name: str,
    experiment_dir: str,
    duration_hours: float
):
    """
    Notify user that experiment has completed.

    Parameters
    ----------
    experiment_name : str
        Experiment identifier
    experiment_dir : str
        Path to experiment data directory
    duration_hours : float
        Actual experiment duration (hours)
    """
    print_separator()
    print(f"EXPERIMENT COMPLETE: {experiment_name}")
    print_separator()
    print(f"Actual duration: {duration_hours:.1f} hours")
    print(f"Data saved to: {experiment_dir}")
    print()
    print("Next steps:")
    print("  [ ] Remove vials from deck")
    print("  [ ] Clean tools if needed")
    print("  [ ] Wait for yield extraction to complete")
    print()
    print_separator()


def notify_yield_extraction(n_vials: int):
    """
    Notify user that yield extraction is in progress.

    Parameters
    ----------
    n_vials : int
        Number of vials to process
    """
    print_separator()
    print("EXTRACTING YIELDS")
    print_separator()
    print(f"Processing {n_vials} vials...")
    print("  - Reading spectral data")
    print("  - Parsing timestamps from operations log")
    print("  - Fitting Gualtieri models")
    print("  - Extracting I_max values")
    print()
    print("This may take 1-2 minutes...")
    print_separator()


def notify_yields_extracted(
    yields: Dict[str, float],
    best_vial: str,
    best_yield: float
):
    """
    Display extracted yields with summary.

    Parameters
    ----------
    yields : dict
        Mapping: vial_id -> yield (I_max)
    best_vial : str
        Vial ID with highest yield
    best_yield : float
        Maximum yield in this batch
    """
    print_separator()
    print("YIELDS EXTRACTED")
    print_separator()
    print()

    for vial_id in sorted(yields.keys()):
        marker = " ★" if vial_id == best_vial else ""
        print(f"  {vial_id}: I_max = {yields[vial_id]:.6f}{marker}")

    print()
    print(f"Best vial in this batch: {best_vial} (I_max = {best_yield:.6f})")
    print_separator()


def notify_iteration_complete(
    iteration: int,
    best_yield_overall: float,
    best_composition: np.ndarray,
    ei_max: float
):
    """
    Display iteration summary.

    Parameters
    ----------
    iteration : int
        Iteration number
    best_yield_overall : float
        Best yield observed so far (across all iterations)
    best_composition : np.ndarray, shape (3,)
        Composition [Co, MIM, TEA] with best yield
    ei_max : float
        Maximum expected improvement for next iteration
    """
    print_separator()
    print(f"ITERATION {iteration} COMPLETE")
    print_separator()
    print()
    print(f"Best yield so far: {best_yield_overall:.6f}")
    print(f"Best composition: Co={best_composition[0]:.1f} mL, "
          f"MIM={best_composition[1]:.1f} mL, TEA={best_composition[2]:.1f} mL")
    print()
    print(f"Max EI for next batch: {ei_max:.4e}")
    print()
    print_separator()


def notify_convergence(
    reason: str,
    final_best_yield: float,
    final_best_composition: np.ndarray,
    total_experiments: int
):
    """
    Notify user that optimization has converged.

    Parameters
    ----------
    reason : str
        Convergence reason
    final_best_yield : float
        Best yield found
    final_best_composition : np.ndarray, shape (3,)
        Optimal composition [Co, MIM, TEA]
    total_experiments : int
        Total number of experiments performed
    """
    print_separator("*")
    print("OPTIMIZATION CONVERGED")
    print_separator("*")
    print()
    print(f"Reason: {reason}")
    print(f"Total experiments: {total_experiments}")
    print()
    print("OPTIMAL CONDITIONS:")
    print(f"  Co²⁺: {final_best_composition[0]:.1f} mL")
    print(f"  2-methylimidazole: {final_best_composition[1]:.1f} mL")
    print(f"  Triethylamine: {final_best_composition[2]:.1f} mL")
    print(f"  Best yield (I_max): {final_best_yield:.6f}")
    print()
    print_separator("*")


def notify_error(error_message: str, details: Optional[str] = None):
    """
    Display error message to user.

    Parameters
    ----------
    error_message : str
        Brief error description
    details : str, optional
        Detailed error information
    """
    print_separator("!")
    print("ERROR")
    print_separator("!")
    print(f"\n{error_message}\n")
    if details:
        print("Details:")
        print(details)
        print()
    print_separator("!")


def display_progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress:",
    bar_length: int = 40
):
    """
    Display a progress bar in the terminal.

    Parameters
    ----------
    current : int
        Current progress value
    total : int
        Total value
    prefix : str, default="Progress:"
        Prefix text
    bar_length : int, default=40
        Length of progress bar
    """
    if total == 0:
        return

    progress = current / total
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)

    percent = progress * 100
    print(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)

    if current == total:
        print()  # New line when complete


def notify_phase_transition_proposal(
    current_phase,
    next_phase,
    summary: Dict,
    reasons: List,
    approval_mode: str
):
    """
    Display phase transition notification.

    Parameters
    ----------
    current_phase : PhaseConfig
        Current phase configuration
    next_phase : PhaseConfig
        Next phase configuration
    summary : dict
        Phase summary statistics
    reasons : list of str
        Reasons for transition
    approval_mode : str
        "manual" or "automatic"
    """
    print()
    print_separator()
    print("PHASE TRANSITION PROPOSED")
    print_separator()
    print()
    print(f"Current Phase: {current_phase.name.upper()}")
    print(f"  Description: {current_phase.description}")
    print(f"  Iterations completed: {summary['iterations']}")
    print(f"  Samples in phase: {summary['samples_in_phase']}")
    print(f"  Best yield in phase: {summary['best_yield_in_phase']:.6f}")
    print()
    print(f"Next Phase: {next_phase.name.upper()}")
    print(f"  Description: {next_phase.description}")
    print(f"  Batch size: {next_phase.batch_size} samples")
    print(f"  Exploration parameter (xi): {next_phase.xi}")
    if next_phase.bounds:
        print(f"  Search space: Co=[{next_phase.bounds[0][0]:.1f},{next_phase.bounds[0][1]:.1f}], "
              f"MIM=[{next_phase.bounds[1][0]:.1f},{next_phase.bounds[1][1]:.1f}], "
              f"TEA=[{next_phase.bounds[2][0]:.1f},{next_phase.bounds[2][1]:.1f}]")
    else:
        print(f"  Search space: Will be computed dynamically")
    print()
    print("Transition Criteria Met:")
    for reason in reasons:
        print(f"  ✓ {reason}")
    print()
    comp = summary['best_composition_overall']
    print(f"Best Overall: Yield={summary['best_yield_overall']:.6f}, "
          f"Composition=[Co={comp[0]:.1f}, MIM={comp[1]:.1f}, TEA={comp[2]:.1f}]")
    print()
    print(f"Review plots at: {summary['plots_dir']}")
    print()

    if approval_mode == "manual":
        print("Mode: MANUAL APPROVAL REQUIRED")
    else:
        print("Mode: AUTOMATIC APPROVAL (proceeding automatically)")

    print_separator()
    print()


def request_phase_approval(next_phase) -> bool:
    """
    Request user approval for phase transition.

    Parameters
    ----------
    next_phase : PhaseConfig
        Phase to transition to

    Returns
    -------
    bool
        True if approved, False if rejected
    """
    while True:
        response = input(f"Approve transition to '{next_phase.name}' phase? [Y/n/info]: ").strip().lower()

        if response in ['y', 'yes', '']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response == 'info':
            print()
            print(f"Phase: {next_phase.name}")
            print(f"Description: {next_phase.description}")
            print(f"Batch size: {next_phase.batch_size}")
            print(f"Max iterations: {next_phase.max_iterations}")
            print(f"Exploration (xi): {next_phase.xi}")
            if next_phase.performance_threshold:
                print(f"Entry requirement: yield > {next_phase.performance_threshold}")
            print()
        else:
            print("Invalid response. Please enter Y (yes), n (no), or 'info' for details.")
