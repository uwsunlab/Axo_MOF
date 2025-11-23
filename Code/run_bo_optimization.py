#!/usr/bin/env python
"""
Autonomous Multi-Phase Bayesian Optimization for ZIF Synthesis

This script provides the main entry point for running autonomous
multi-phase Bayesian optimization of ZIF synthesis conditions.

Usage Examples:
    # Multi-phase with manual approval (default)
    python run_bo_optimization.py --new --approval-mode manual

    # Multi-phase with automatic approval
    python run_bo_optimization.py --new --approval-mode automatic

    # Single-phase mode (backward compatible)
    python run_bo_optimization.py --new --phases single --batch-size 5 --max-iterations 20

    # Resume from saved state
    python run_bo_optimization.py --resume

The system will:
1. Generate initial samples (or load from saved state)
2. Execute synthesis batches with phase-specific parameters
3. Extract yields from spectral data
4. Update Gaussian Process model
5. Select next batch using Expected Improvement
6. Check phase transition criteria
7. Prompt for approval (manual mode) or auto-transition (automatic mode)
8. Repeat until final phase converges

Phase Configurations:
- Multi-phase (default): Exploration (broad search) -> Refinement (focused) -> Validation (fine-tuning)
- Single-phase: Traditional BO with fixed batch size throughout
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from science_jubilee.Machine import Machine
from science_jubilee.Experiment import Experiment
from science_jubilee.optimization.orchestrator import BayesianOptimizationLoop
from science_jubilee.optimization.phase_config import (
    PhaseConfig, DEFAULT_PHASES, create_single_phase_config
)


def setup_hardware():
    """
    Initialize and configure the Jubilee machine with all tools and labware.

    Returns
    -------
    machine : Machine
        Configured machine instance
    experiment : Experiment
        Configured experiment instance
    """
    print("=" * 70)
    print("HARDWARE SETUP")
    print("=" * 70)

    # Initialize machine
    print("Connecting to Jubilee at 192.168.1.2...")
    machine = Machine(address='192.168.1.2')

    print("Homing all axes...")
    machine.home_all()
    machine.move_to(z=200)  # Safe Z position

    # Load deck
    print("Loading deck configuration...")
    deck = machine.load_deck("lab_automation_deck")

    # Load labware
    print("Loading labware...")

    solvents = machine.load_labware(
        'uwsunlab_2_wellplate_60000ul_slot3.json',
        slot=3, has_lid_on_top=False, currentLiquidVolume=60
    )
    solvents.load_manualOffset()

    precursors = machine.load_labware(
        'uwsunlab_2_wellplate_60000ul_slot1.json',
        slot=1, has_lid_on_top=False, currentLiquidVolume=60
    )
    precursors.load_manualOffset()

    # Load sample labware (3 views: single syringe, dual syringe, spectrometer)
    samples2_ssy = machine.load_labware(
        'uwsunlab_10_wellplate_14000ul_ssy_1.json',
        slot=2, has_lid_on_top=False
    )
    samples2_ssy.load_manualOffset()

    samples2_sy = machine.load_labware(
        'uwsunlab_10_wellplate_14000ul_sy_1.json',
        slot=2, has_lid_on_top=False
    )
    samples2_sy.load_manualOffset()

    samples2_spec = machine.load_labware(
        'uwsunlab_10_wellplate_14000ul_spec_1.json',
        slot=2, has_lid_on_top=False
    )
    samples2_spec.load_manualOffset()

    samples5_ssy = machine.load_labware(
        'uwsunlab_10_wellplate_14000ul_ssy_2.json',
        slot=5, has_lid_on_top=False
    )
    samples5_ssy.load_manualOffset()

    samples5_sy = machine.load_labware(
        'uwsunlab_10_wellplate_14000ul_sy_2.json',
        slot=5, has_lid_on_top=False
    )
    samples5_sy.load_manualOffset()

    samples5_spec = machine.load_labware(
        'uwsunlab_10_wellplate_14000ul_spec_2.json',
        slot=5, has_lid_on_top=False
    )
    samples5_spec.load_manualOffset()

    # Load tools
    print("Loading tools...")

    from science_jubilee.tools.Syringe import Syringe
    from science_jubilee.tools.Double_Syringe import DoubleSyringe
    from science_jubilee.tools.Vacuum_Gripper import VacuumGripper
    from science_jubilee.tools.Oceandirect_axo import Spectrometer

    single_syringe = Syringe(
        index=0,
        name='single_syringe',
        config='single_syringe'
    )
    machine.load_tool(single_syringe)

    dual_syringe = DoubleSyringe(
        index=2,
        name='Dual_Syringe',
        config='10cc_syringe'
    )
    machine.load_tool(dual_syringe)

    gripper = VacuumGripper(
        index=4,
        name='Vacuum_Gripper',
        vacuum_pin=0,
        limit_switch_pin=2
    )
    machine.load_tool(gripper)

    spectrometer = Spectrometer(
        index=3,
        name='Spectrometer',
        base_dir=r"C:\Axo\science-jubilee\axo\spectrum_data",
        experiment_name='BO_Optimization',  # Will be overwritten
        operator_name='BO_System',
        target_compound='ZIF',
        experiment_notes='Automated Bayesian optimization',
        solvent='triethylamine/methanol',
        temperature_c=25
    )
    machine.load_tool(spectrometer)

    # Define vacuum gripper locations
    vacuum_location = {
        0: {"loc": (84, 51, 0), "labwares_list": []},
        1: {"loc": (225, 57, 0), "labwares_list": [precursors]},
        2: {"loc": (84, 147, 0), "labwares_list": [samples2_ssy, samples2_sy, samples2_spec]},
        3: {"loc": (225, 149, 0), "labwares_list": [solvents]},
        4: {"loc": (79, 244, 0), "labwares_list": []},
        5: {"loc": (221, 244, 0), "labwares_list": [samples5_ssy, samples5_sy, samples5_spec]}
    }

    # Organize labware and tools
    all_labwares = {
        "slot1": {"precursors": precursors},
        "slot2": {
            "samples2_ssy": samples2_ssy,
            "samples2_sy": samples2_sy,
            "samples2_spec": samples2_spec
        },
        "slot3": {"solvents": solvents},
        "slot4": {"vacuum_location": vacuum_location},
        "slot5": {
            "samples5_ssy": samples5_ssy,
            "samples5_sy": samples5_sy,
            "samples5_spec": samples5_spec
        }
    }

    all_tools = {
        "single_syringe": single_syringe,
        "dual_syringe": dual_syringe,
        "spectrometer": spectrometer,
        "gripper": gripper
    }

    # Create experiment
    experiment = Experiment(
        machine=machine,
        deck=deck,
        all_tools=all_tools,
        all_labwares=all_labwares
    )

    print("Hardware setup complete!")
    print("=" * 70)
    print()

    return machine, experiment


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Phase Bayesian Optimization for ZIF Synthesis"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start new optimization"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results",
        help="Output directory for results (default: optimization_results)"
    )
    parser.add_argument(
        "--approval-mode",
        type=str,
        choices=["manual", "automatic"],
        default="manual",
        help="Phase transition approval mode: 'manual' (prompt user) or 'automatic' (no prompts) (default: manual)"
    )
    parser.add_argument(
        "--phases",
        type=str,
        choices=["multi", "single"],
        default="multi",
        help="Phase configuration: 'multi' (3-phase: exploration->refinement->validation) or 'single' (1-phase) (default: multi)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="For single-phase mode: maximum iterations (default: 20). Ignored in multi-phase mode."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="For single-phase mode: experiments per batch (default: 5). Ignored in multi-phase mode."
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="Operator",
        help="Operator name (default: Operator)"
    )

    args = parser.parse_args()

    if not args.new and not args.resume:
        parser.error("Must specify either --new or --resume")

    if args.new and args.resume:
        parser.error("Cannot specify both --new and --resume")

    try:
        # Setup hardware
        machine, experiment = setup_hardware()

        # Configure phases
        if args.phases == "multi":
            phases = DEFAULT_PHASES
            print(f"\nUsing multi-phase configuration (3 phases)")
            print(f"  1. Exploration: batch_size=10, xi=0.05, iterations=3-5")
            print(f"  2. Refinement: batch_size=5, xi=0.01, iterations=5-10")
            print(f"  3. Validation: batch_size=2, xi=0.001, iterations=2-5")
        else:
            phases = create_single_phase_config(
                batch_size=args.batch_size,
                max_iterations=args.max_iterations,
                xi=0.01
            )
            print(f"\nUsing single-phase configuration")
            print(f"  batch_size={args.batch_size}, max_iterations={args.max_iterations}")

        print(f"Approval mode: {args.approval_mode}")
        print()

        # Create orchestrator
        orchestrator = BayesianOptimizationLoop(
            machine=machine,
            experiment=experiment,
            phases=phases,
            approval_mode=args.approval_mode,
            output_dir=args.output_dir,
            n_initial_samples=5,
            convergence_threshold=0.001,
            state_file="bo_state.json",
            operator_name=args.operator
        )

        # Resume if requested
        if args.resume:
            print("Resuming from saved state...")
            orchestrator.load_state()

        # Run optimization loop
        print("\nStarting multi-phase Bayesian optimization...\n")
        orchestrator.run()

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {args.output_dir}")
        print("Check the log file for detailed information.")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("State has been saved. You can resume later with --resume flag.")
        sys.exit(0)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("\nCheck the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
