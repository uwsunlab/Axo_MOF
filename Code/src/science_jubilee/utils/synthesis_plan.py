"""
Utilities for generating synthesis plan JSON files from Bayesian optimization outputs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def create_synthesis_plan_json(
    X_batch: np.ndarray,
    output_path: str,
    experiment_name: str,
    operator_name: str = "Operator",
    spectrum_interval_mins: int = 15,
    max_spectrum_records: int = 13,
    metal_precursor_name: str = "Co+2",
    organic_precursor_name: str = "2-methylimidazole",
    solvent_name: str = "triethylamine/methanol",
    experiment_description: Optional[str] = None,
    approximate_time: Optional[str] = None
) -> str:
    """
    Convert Bayesian optimization batch to synthesis plan JSON.

    Maps compositions to vial slots:
    - Batch size ≤5: Use slot2 (A1-A5)
    - Batch size 6-10: Use slot2 + slot5 (A1-A5, B1-B5)

    Parameters
    ----------
    X_batch : np.ndarray, shape (k, 3)
        Array of [Co, MIM, TEA] compositions in mL
    output_path : str
        Path to save JSON file
    experiment_name : str
        Experiment identifier (e.g., "BO-7")
    operator_name : str, default="Operator"
        Operator name
    spectrum_interval_mins : int, default=15
        Time between spectrum measurements (minutes)
    max_spectrum_records : int, default=13
        Maximum number of spectrum readings per vial
    metal_precursor_name : str, default="Co+2"
        Metal precursor name
    organic_precursor_name : str, default="2-methylimidazole"
        Organic linker name
    solvent_name : str, default="triethylamine/methanol"
        Solvent name
    experiment_description : str, optional
        Description of experiment
    approximate_time : str, optional
        Estimated experiment duration

    Returns
    -------
    str
        Path to created JSON file

    Raises
    ------
    ValueError
        If batch size > 10 (exceeds available slots)
    """
    X_batch = np.asarray(X_batch, float)
    k = len(X_batch)

    if k > 10:
        raise ValueError(f"Batch size {k} exceeds maximum of 10 vials (2 slots × 5 vials)")

    # Assign vials to slots
    vial_assignments: Dict[str, Dict[str, Dict[str, float]]] = {}

    slot2_vials = ["A1", "A2", "A3", "A4", "A5"]
    slot5_vials = ["B1", "B2", "B3", "B4", "B5"]

    if k <= 5:
        # Use only slot2
        vial_assignments["slot2"] = {}
        for i in range(k):
            co, mim, tea = X_batch[i]
            vial_assignments["slot2"][slot2_vials[i]] = {
                "metal_precursor_vol": float(co),
                "organic_precursor_vol": float(mim),
                "solvent_vol": float(tea)
            }
    else:
        # Use both slots
        vial_assignments["slot2"] = {}
        vial_assignments["slot5"] = {}

        for i in range(5):
            co, mim, tea = X_batch[i]
            vial_assignments["slot2"][slot2_vials[i]] = {
                "metal_precursor_vol": float(co),
                "organic_precursor_vol": float(mim),
                "solvent_vol": float(tea)
            }

        for i in range(5, k):
            co, mim, tea = X_batch[i]
            vial_assignments["slot5"][slot5_vials[i - 5]] = {
                "metal_precursor_vol": float(co),
                "organic_precursor_vol": float(mim),
                "solvent_vol": float(tea)
            }

    # Build synthesis plan
    now = datetime.now()
    plan = {
        **vial_assignments,
        "spectrum_record_interval_mins": spectrum_interval_mins,
        "max_spectrum_records": max_spectrum_records,
        "Experiment": {
            "operator_name": operator_name,
            "experiment_name": experiment_name,
            "experiment_date": now.strftime("%d %B %Y"),
            "experiment_time": now.strftime("%H:%M"),
            "experiment_description": experiment_description or f"Synthesis of {k} vials of MOF",
            "approximate_time": approximate_time or f"{(max_spectrum_records * spectrum_interval_mins / 60):.1f} hours",
            "metal_precursor_name": metal_precursor_name,
            "organic_precursor_name": organic_precursor_name,
            "solvent_name": solvent_name
        }
    }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=4)

    return str(output_path)


def get_vial_assignments_from_plan(plan_path: str) -> Dict[str, tuple]:
    """
    Parse synthesis plan JSON and extract vial assignments.

    Parameters
    ----------
    plan_path : str
        Path to synthesis plan JSON

    Returns
    -------
    dict
        Mapping: vial_id -> (Co_mL, MIM_mL, TEA_mL)
    """
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)

    assignments = {}

    for slot_key in ["slot2", "slot5"]:
        if slot_key in plan:
            for vial_id, volumes in plan[slot_key].items():
                co = volumes["metal_precursor_vol"]
                mim = volumes["organic_precursor_vol"]
                tea = volumes["solvent_vol"]
                assignments[vial_id] = (co, mim, tea)

    return assignments


def extract_batch_from_plan(plan_path: str) -> np.ndarray:
    """
    Extract composition batch from synthesis plan JSON.

    Parameters
    ----------
    plan_path : str
        Path to synthesis plan JSON

    Returns
    -------
    np.ndarray, shape (k, 3)
        Array of [Co, MIM, TEA] compositions
    """
    assignments = get_vial_assignments_from_plan(plan_path)

    # Sort by vial ID to ensure consistent ordering
    sorted_vials = sorted(assignments.keys())

    X_batch = np.array([assignments[vial] for vial in sorted_vials], dtype=float)

    return X_batch
