"""
Bayesian Optimization Orchestrator for autonomous MOF synthesis.

This module provides the main BayesianOptimizationLoop class that coordinates:
- Initial sampling
- Synthesis plan generation
- Experiment execution
- Yield extraction
- GP model updating
- Batch selection
- Convergence checking
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from science_jubilee.analysis.gualtieri import batch_extract_yields
from science_jubilee.optimization.bayesian import (
    fit_gp, select_and_plot_next_batch, append_batch, build_feasible_lattice
)
from science_jubilee.optimization.sampling import initial_sampling
from science_jubilee.optimization.convergence import suggest_stopping
from science_jubilee.optimization.phase_config import PhaseConfig
from science_jubilee.optimization.phase_manager import PhaseManager
from science_jubilee.optimization.logging_config import (
    setup_bo_logger, log_iteration_start, log_gp_fit, log_ei_statistics,
    log_batch_selection, log_yield_extraction, log_convergence, log_error
)
from science_jubilee.optimization.notifications import (
    notify_vial_loading, wait_for_confirmation, notify_experiment_start,
    notify_experiment_complete, notify_yield_extraction, notify_yields_extracted,
    notify_iteration_complete, notify_convergence, notify_error,
    notify_phase_transition_proposal, request_phase_approval
)
from science_jubilee.utils.synthesis_plan import (
    create_synthesis_plan_json, get_vial_assignments_from_plan
)


class BayesianOptimizationLoop:
    """
    Autonomous Bayesian optimization loop for ZIF synthesis.

    This class orchestrates the complete workflow:
    1. Generate initial samples (or load from saved state)
    2. Create synthesis plan JSON
    3. Wait for operator to load vials
    4. Execute synthesis experiment
    5. Extract yields from spectral data
    6. Update GP model
    7. Select next batch
    8. Check convergence
    9. Repeat until converged or max iterations
    """

    def __init__(
        self,
        machine,
        experiment,
        phases: List[PhaseConfig],
        approval_mode: str = "manual",
        output_dir: str = "optimization_results",
        n_initial_samples: int = 5,
        convergence_threshold: float = 0.001,
        state_file: str = "bo_state.json",
        operator_name: str = "Operator"
    ):
        """
        Initialize Bayesian optimization loop with multi-phase support.

        Parameters
        ----------
        machine : Machine
            Jubilee machine instance
        experiment : Experiment
            Experiment instance for running synthesis
        phases : list of PhaseConfig
            Ordered list of optimization phases
        approval_mode : str, default="manual"
            "manual" or "automatic" - controls ALL phase transition approvals
        output_dir : str, default="optimization_results"
            Directory for saving results
        n_initial_samples : int, default=5
            Number of initial samples
        convergence_threshold : float, default=0.001
            EI threshold for convergence
        state_file : str, default="bo_state.json"
            Filename for state persistence
        operator_name : str, default="Operator"
            Operator name for synthesis plans
        """
        self.machine = machine
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.n_initial_samples = n_initial_samples
        self.convergence_threshold = convergence_threshold
        self.state_file = self.output_dir / state_file
        self.operator_name = operator_name

        # Multi-phase management
        self.phase_manager = PhaseManager(phases, approval_mode)

        # Bounds from first phase, or use default
        self.bounds = self.phase_manager.current_phase.bounds or [(0.0, 8.0), (0.0, 8.0), (0.0, 8.0)]

        # State variables
        self.X: Optional[np.ndarray] = None  # (n, 3) compositions
        self.y: Optional[np.ndarray] = None  # (n,) yields
        self.iteration: int = 0
        self.ei_history: List[float] = []
        self.converged: bool = False
        self.convergence_reason: str = ""

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_bo_logger(str(self.output_dir))

        self.logger.info("BayesianOptimizationLoop initialized (multi-phase)")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Number of phases: {len(phases)}")
        self.logger.info(f"Approval mode: {approval_mode}")
        self.logger.info(f"Initial phase: {self.phase_manager.current_phase.name}")
        self.logger.info(f"Initial bounds: {self.bounds}")
        self.logger.info(f"Initial batch size: {self.phase_manager.current_phase.batch_size}")

    def initialize(self) -> np.ndarray:
        """
        Generate initial samples using maximin sampling.

        Returns
        -------
        np.ndarray, shape (n_initial_samples, 3)
            Initial sample compositions
        """
        self.logger.info(f"Generating {self.n_initial_samples} initial samples...")

        X_init = initial_sampling(
            total_ml=8.0,
            step_ml=0.1,
            n_init=self.n_initial_samples,
            min_co=0.1,
            min_mim=0.1,
            min_tea=0.0,
            random_state=42
        )

        self.logger.info(f"Generated {len(X_init)} initial samples")
        for i, comp in enumerate(X_init):
            self.logger.debug(f"  {i+1}. Co={comp[0]:.1f}, MIM={comp[1]:.1f}, TEA={comp[2]:.1f}")

        return X_init

    def create_next_synthesis_plan(self, X_next: np.ndarray) -> str:
        """
        Generate synthesis plan JSON for next batch.

        Parameters
        ----------
        X_next : np.ndarray, shape (k, 3)
            Batch compositions [Co, MIM, TEA]

        Returns
        -------
        str
            Path to created JSON file
        """
        phase_name = self.phase_manager.current_phase.name
        experiment_name = f"BO-{phase_name}-iter{self.phase_manager.phase_iteration}"
        output_path = self.output_dir / f"synthesis_plan_{phase_name}_iter{self.phase_manager.phase_iteration}.json"

        self.logger.info(f"Creating synthesis plan: {output_path}")
        self.logger.info(f"Phase: {phase_name}, iteration: {self.phase_manager.phase_iteration}")

        plan_path = create_synthesis_plan_json(
            X_batch=X_next,
            output_path=str(output_path),
            experiment_name=experiment_name,
            operator_name=self.operator_name,
            spectrum_interval_mins=15,
            max_spectrum_records=13,
            metal_precursor_name="Co+2",
            organic_precursor_name="2-methylimidazole",
            solvent_name="triethylamine/methanol"
        )

        self.logger.info(f"Synthesis plan created: {plan_path}")
        return plan_path

    def wait_for_vial_loading(self, plan_path: str):
        """
        Notify operator to load vials and wait for confirmation.

        Parameters
        ----------
        plan_path : str
            Path to synthesis plan JSON
        """
        vial_assignments = get_vial_assignments_from_plan(plan_path)

        notify_vial_loading(vial_assignments, self.iteration)
        self.logger.info("Waiting for operator to load vials...")

        confirmed = wait_for_confirmation("Press ENTER when vials are loaded and ready...")

        if not confirmed:
            raise KeyboardInterrupt("User cancelled vial loading")

        self.logger.info("Vial loading confirmed")

    def run_synthesis(self, plan_path: str) -> Tuple[str, float]:
        """
        Execute synthesis via Experiment.make_batch().

        Parameters
        ----------
        plan_path : str
            Path to synthesis plan JSON

        Returns
        -------
        experiment_dir : str
            Path to experiment data directory
        duration_hours : float
            Actual experiment duration (hours)
        """
        vial_assignments = get_vial_assignments_from_plan(plan_path)
        n_vials = len(vial_assignments)

        # Estimate duration
        estimated_duration_hours = (13 * 15) / 60.0  # max_records * interval / 60

        notify_experiment_start(
            experiment_name=f"BO-iter{self.iteration}",
            n_vials=n_vials,
            estimated_duration_hours=estimated_duration_hours
        )

        self.logger.info(f"Starting experiment with {n_vials} vials...")
        start_time = time.time()

        # Run experiment
        try:
            self.experiment.make_batch(plan_path)
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            raise

        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600.0

        # Get experiment directory from experiment object
        experiment_dir = self.experiment.experiment_dir

        notify_experiment_complete(
            experiment_name=f"BO-iter{self.iteration}",
            experiment_dir=experiment_dir,
            duration_hours=duration_hours
        )

        self.logger.info(f"Experiment complete. Duration: {duration_hours:.2f} hours")
        self.logger.info(f"Data saved to: {experiment_dir}")

        return experiment_dir, duration_hours

    def extract_yields(self, experiment_dir: str) -> Tuple[np.ndarray, List[str]]:
        """
        Automatically extract yields from completed experiment.

        Parameters
        ----------
        experiment_dir : str
            Path to experiment data directory

        Returns
        -------
        y_batch : np.ndarray, shape (k,)
            Extracted yields (I_max values)
        failed_vials : list of str
            Vial IDs with failed fits
        """
        batch_size = self.phase_manager.current_phase.batch_size
        notify_yield_extraction(n_vials=batch_size)
        self.logger.info("Extracting yields from spectral data...")

        try:
            yields_dict = batch_extract_yields(
                experiment_dir=experiment_dir,
                wavelength_nm=400.0
            )
        except Exception as e:
            self.logger.error(f"Yield extraction failed: {e}", exc_info=True)
            raise

        # Sort by vial ID to ensure consistent ordering
        sorted_vials = sorted(yields_dict.keys())
        y_batch = np.array([yields_dict[v] for v in sorted_vials], dtype=float)

        # Identify failed fits (yield = 0.0)
        failed_vials = [v for v in sorted_vials if yields_dict[v] <= 0.0]

        best_vial = max(yields_dict, key=yields_dict.get)
        best_yield = yields_dict[best_vial]

        notify_yields_extracted(yields_dict, best_vial, best_yield)

        log_yield_extraction(self.logger, self.iteration, yields_dict, failed_vials)

        self.logger.info(f"Extracted {len(y_batch)} yields. Failed fits: {len(failed_vials)}")

        return y_batch, failed_vials

    def update_model(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update dataset and refit GP model.

        Parameters
        ----------
        X_new : np.ndarray, shape (k, 3)
            New compositions
        y_new : np.ndarray, shape (k,)
            New yields
        """
        self.logger.info("Updating GP model...")

        # Append new data
        if self.X is None or self.y is None:
            self.X = X_new
            self.y = y_new
        else:
            self.X, self.y = append_batch(self.X, self.y, X_new, y_new)

        self.logger.info(f"Total samples: {len(self.X)}")

        # Fit GP
        fit, info = fit_gp(self.X, self.y, self.bounds, normalise_y=True, n_restarts=20)

        log_gp_fit(
            self.logger, self.iteration,
            nll=info["nll"],
            kernel_str=info["kernel_"],
            y_mean=info["y_mean"],
            y_std=info["y_std"]
        )

        # Generate visualization plots
        self._save_visualization_plots(fit)

    def check_and_execute_phase_transition(self, ei_max: float) -> bool:
        """
        Check if phase transition should occur and execute if approved.

        Parameters
        ----------
        ei_max : float
            Maximum Expected Improvement value

        Returns
        -------
        bool
            True if transition occurred, False otherwise
        """
        # Check if we should transition
        should_transition, reasons = self.phase_manager.check_transition_criteria(
            iteration=self.phase_manager.phase_iteration,
            best_yield=float(np.max(self.y)),
            ei_max=ei_max
        )

        if not should_transition:
            self.logger.info(f"Phase transition check: {reasons[0]}")
            return False

        # Get next phase
        next_phase = self.phase_manager.get_next_phase()
        if next_phase is None:
            self.logger.info("No more phases available")
            return False

        # Generate phase summary
        plots_dir = str(self.output_dir / "plots")
        summary = self.phase_manager.generate_phase_summary(
            X=self.X,
            y=self.y,
            iteration=self.phase_manager.phase_iteration,
            plots_dir=plots_dir
        )

        # Notify user
        notify_phase_transition_proposal(
            current_phase=self.phase_manager.current_phase,
            next_phase=next_phase,
            summary=summary,
            reasons=reasons,
            approval_mode=self.phase_manager.approval_mode
        )

        # Get approval (automatic or manual)
        if self.phase_manager.approval_mode == "manual":
            approved = request_phase_approval(next_phase)
            if not approved:
                self.logger.info(f"Phase transition to '{next_phase.name}' rejected by user")
                return False
        else:
            self.logger.info(f"Phase transition to '{next_phase.name}' approved automatically")
            approved = True

        if approved:
            # Update bounds if next phase has dynamic bounds
            if next_phase.bounds is None:
                self.logger.info("Computing refinement bounds for next phase...")
                new_bounds = self.phase_manager.compute_refinement_bounds(
                    X=self.X,
                    y=self.y,
                    top_percentile=0.2,
                    margin=0.5
                )
                next_phase.bounds = new_bounds
                self.logger.info(f"New bounds: {new_bounds}")

            # Execute transition
            self.phase_manager.transition_to_next_phase(total_samples=len(self.X))
            self.bounds = self.phase_manager.current_phase.bounds

            self.logger.info(f"Transitioned to phase '{self.phase_manager.current_phase.name}'")
            self.logger.info(f"New batch size: {self.phase_manager.current_phase.batch_size}")
            self.logger.info(f"New xi: {self.phase_manager.current_phase.xi}")

            return True

        return False

    def _save_visualization_plots(self, fit: Dict):
        """
        Save GP mean, std, and EI heatmaps.

        Parameters
        ----------
        fit : dict
            Fitted GP model dictionary
        """
        from science_jubilee.optimization.bayesian import (
            plot_sampling_space, plot_gp_heatmap, plot_initial_samples
        )

        plots_dir = self.output_dir / "plots" / f"iter{self.iteration}"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Mean heatmap
        tax, _ = plot_sampling_space(scale_ml=8.0)
        plot_gp_heatmap(tax, fit, self.bounds, which="mean", step_ml=0.1)
        plot_initial_samples(tax, self.X, marker_color="red")
        plt.savefig(plots_dir / "gp_mean.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Std heatmap
        tax, _ = plot_sampling_space(scale_ml=8.0)
        plot_gp_heatmap(tax, fit, self.bounds, which="std", step_ml=0.1)
        plot_initial_samples(tax, self.X, marker_color="red")
        plt.savefig(plots_dir / "gp_std.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved visualization plots to {plots_dir}")

    def select_next_batch(self) -> Tuple[np.ndarray, Dict, float]:
        """
        Select next batch using Bayesian optimization with phase-specific parameters.

        Returns
        -------
        X_next : np.ndarray, shape (k, 3)
            Next batch compositions
        table : dict
            Summary table
        ei_max : float
            Maximum EI value
        """
        # Get phase-specific parameters
        batch_size = self.phase_manager.current_phase.batch_size
        xi = self.phase_manager.current_phase.xi

        self.logger.info(f"Selecting next batch using BO (batch_size={batch_size}, xi={xi})...")

        phase_name = self.phase_manager.current_phase.name
        plots_dir = self.output_dir / "plots" / f"{phase_name}_iter{self.phase_manager.phase_iteration}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        ei_plot_path = plots_dir / "ei_heatmap.png"

        X_next, table, cache = select_and_plot_next_batch(
            X=self.X,
            y=self.y,
            bounds=self.bounds,
            k=batch_size,
            xi=xi,
            total_ml=8.0,
            step_ml=0.1,
            radius_uc=0.18,
            random_state=42,
            save_plot=str(ei_plot_path)
        )

        plt.close('all')  # Clean up plots

        # Extract EI statistics
        fit, info, (lattice, mu, sd, ei) = cache
        ei_max = float(np.max(ei))
        ei_mean = float(np.mean(ei))
        ei_std = float(np.std(ei))

        self.ei_history.append(ei_max)

        log_ei_statistics(self.logger, self.iteration, ei_max, ei_mean, ei_std)
        log_batch_selection(self.logger, self.iteration, X_next, table["EI"])

        return X_next, table, ei_max

    def check_convergence(self) -> Tuple[bool, str]:
        """
        Check if optimization has converged (phase-aware).

        Returns
        -------
        converged : bool
            True if converged
        reason : str
            Convergence reason or status
        """
        lattice = build_feasible_lattice(total_ml=8.0, step_ml=0.1)
        lattice_size = len(lattice)

        # Use phase-specific max iterations
        max_iterations = self.phase_manager.current_phase.max_iterations

        converged, reason = suggest_stopping(
            X=self.X,
            y=self.y,
            iteration=self.phase_manager.phase_iteration,
            max_iterations=max_iterations,
            ei_history=self.ei_history,
            lattice_size=lattice_size
        )

        if converged:
            self.logger.info(f"Phase convergence achieved: {reason}")
        else:
            self.logger.info(f"Phase convergence status: {reason}")

        return converged, reason

    def save_state(self):
        """Persist X, y, iteration, and phase state to JSON for resume capability."""
        state = {
            "iteration": self.iteration,
            "X": self.X.tolist() if self.X is not None else None,
            "y": self.y.tolist() if self.y is not None else None,
            "ei_history": self.ei_history,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "phase_state": self.phase_manager.get_state(),
            "bounds": self.bounds,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"State saved to {self.state_file}")
        self.logger.info(f"Current phase: {self.phase_manager.current_phase.name}")

    def load_state(self):
        """Resume from saved state including phase state."""
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file not found: {self.state_file}")

        with open(self.state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.iteration = state["iteration"]
        self.X = np.array(state["X"]) if state["X"] else None
        self.y = np.array(state["y"]) if state["y"] else None
        self.ei_history = state["ei_history"]
        self.converged = state["converged"]
        self.convergence_reason = state["convergence_reason"]

        # Restore phase state
        if "phase_state" in state:
            self.phase_manager.set_state(state["phase_state"])

        # Restore bounds
        if "bounds" in state:
            self.bounds = state["bounds"]

        self.logger.info(f"State loaded from {self.state_file}")
        self.logger.info(f"Resuming from iteration {self.iteration}")
        self.logger.info(f"Current phase: {self.phase_manager.current_phase.name}")

    def run_iteration(self) -> Dict:
        """
        Execute one complete BO iteration with phase-aware logic.

        Returns
        -------
        dict
            Iteration summary
        """
        phase_name = self.phase_manager.current_phase.name
        phase_iter = self.phase_manager.phase_iteration

        log_iteration_start(
            self.logger,
            self.iteration,
            len(self.X) if self.X is not None else 0
        )
        self.logger.info(f"Phase: {phase_name}, phase iteration: {phase_iter}")

        try:
            # Select next batch (or use initial samples)
            if self.X is None or self.y is None:
                X_next = self.initialize()
                ei_max = np.nan
            else:
                X_next, table, ei_max = self.select_next_batch()

            # Create synthesis plan
            plan_path = self.create_next_synthesis_plan(X_next)

            # Wait for vial loading
            self.wait_for_vial_loading(plan_path)

            # Run synthesis
            experiment_dir, duration_hours = self.run_synthesis(plan_path)

            # Extract yields
            y_next, failed_vials = self.extract_yields(experiment_dir)

            # Update model
            self.update_model(X_next, y_next)

            # Check for phase transition (if not initial samples)
            phase_transitioned = False
            if self.X is not None and not np.isnan(ei_max):
                phase_transitioned = self.check_and_execute_phase_transition(ei_max)

            # Check convergence
            converged, reason = self.check_convergence()

            # Notify iteration complete
            best_idx = int(np.argmax(self.y))
            notify_iteration_complete(
                iteration=self.iteration,
                best_yield_overall=float(self.y[best_idx]),
                best_composition=self.X[best_idx],
                ei_max=ei_max if not np.isnan(ei_max) else 0.0
            )

            # Save state
            self.save_state()

            summary = {
                "iteration": self.iteration,
                "phase_name": phase_name,
                "phase_iteration": phase_iter,
                "phase_transitioned": phase_transitioned,
                "n_samples_total": len(self.X),
                "best_yield": float(np.max(self.y)),
                "ei_max": float(ei_max) if not np.isnan(ei_max) else None,
                "converged": converged,
                "reason": reason,
                "experiment_dir": experiment_dir,
                "duration_hours": duration_hours
            }

            # Increment counters
            self.iteration += 1
            self.phase_manager.phase_iteration += 1

            return summary

        except Exception as e:
            log_error(self.logger, f"Iteration {self.iteration} failed", exception=e)
            notify_error(f"Iteration {self.iteration} failed", details=str(e))
            raise

    def run(self):
        """
        Main autonomous multi-phase loop. Runs until final phase converges.
        """
        self.logger.info("=" * 70)
        self.logger.info("MULTI-PHASE BAYESIAN OPTIMIZATION STARTING")
        self.logger.info("=" * 70)
        self.logger.info(f"Total phases: {len(self.phase_manager.phases)}")
        self.logger.info(f"Approval mode: {self.phase_manager.approval_mode}")

        try:
            # Continue until converged or no more phases
            while not self.converged:
                summary = self.run_iteration()

                # Check if current phase has converged
                if summary["converged"]:
                    # If we're on the last phase, we're done
                    if not self.phase_manager.has_next_phase:
                        self.converged = True
                        self.convergence_reason = f"Final phase '{self.phase_manager.current_phase.name}' converged: {summary['reason']}"
                        break
                    else:
                        # Phase converged but more phases available
                        # Phase transition will be handled in next iteration
                        self.logger.info(f"Phase '{self.phase_manager.current_phase.name}' converged, but more phases available")

            # Final summary
            if self.converged:
                best_idx = int(np.argmax(self.y))
                notify_convergence(
                    reason=self.convergence_reason,
                    final_best_yield=float(self.y[best_idx]),
                    final_best_composition=self.X[best_idx],
                    total_experiments=len(self.X)
                )

                log_convergence(
                    self.logger,
                    reason=self.convergence_reason,
                    iteration=self.iteration,
                    best_yield=float(self.y[best_idx]),
                    best_composition=self.X[best_idx]
                )

            self.logger.info("=" * 70)
            self.logger.info("MULTI-PHASE BAYESIAN OPTIMIZATION COMPLETE")
            self.logger.info(f"Total iterations: {self.iteration}")
            self.logger.info(f"Total experiments: {len(self.X) if self.X is not None else 0}")
            self.logger.info("=" * 70)

        except KeyboardInterrupt:
            self.logger.warning("Optimization interrupted by user")
            self.save_state()
            print("\nState saved. You can resume later.")

        except Exception as e:
            log_error(self.logger, "Optimization loop failed", exception=e)
            self.save_state()
            raise
