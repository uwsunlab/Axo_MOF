# Reaction-aware control of autonomous batch synthesis

> **Reaction-aware autonomous batch synthesis for ZIF-67 crystallization using robotic liquid handling, in situ UV–Vis spectroscopy, and adaptive Bayesian optimization**

This repository contains the software, analysis workflows, and supporting assets for **Axo**, a modular autonomous platform for solution-phase batch synthesis. The platform integrates robotic reagent delivery, programmable tool changing, cyclic in situ UV–Vis measurements, kinetic model fitting, and reaction-aware Bayesian optimization to co-optimize **what formulation to test** and **how the experiment should be executed**.

---

## 🌟 Key Features

### Reaction-Aware Adaptive Execution
- **Adaptive scheduling**: 
  - **Throughput-prioritized exploration**: 5-vial batch, 15 min measurement interval
  - **Measurement-prioritized refinement**: 2-vial batch, 5 min measurement interval
- **Reaction-aware trigger**: Switch modes automatically when early reaction progress exceeds a threshold
- **Flexible approval modes**: Manual (interactive) or automatic (fully autonomous)
- **State persistence**: Resume interrupted runs from checkpoints

### Automated Synthesis & Characterization
- **Robotic liquid handling**: Precise dispensing with single/dual syringe systems
- **In situ monitoring**: Cyclic UV–Vis spectroscopy integrated into the batch synthesis workflow
- **Autonomous mixing**: Automated mixing and reaction control
- **Programmable multi-vial workflows**: Batch synthesis and cyclic monitoring across multiple reaction vials

### Intelligent Data Analysis
- **Gualtieri kinetic fitting**: Automated MOF growth curve analysis
- **Reaction-progress extraction**: Automatic extraction of extent of reaction `α(t)` and nucleation rate constant `k_n`
- **Gaussian Process modeling**: Uncertainty-aware surrogate models
- **Expected Improvement**: Efficient exploration-exploitation balance

### Production-Ready Engineering
- **Comprehensive logging**: Experiment metadata, synthesis recipes, operation logs, reference spectra, and time-resolved UV–Vis outputs
- **Error handling**: Graceful handling of excluded or failed fits (`no_reaction`, `no_target_feature`)
- **Hardware safety**: Homing checks, collision avoidance, safe Z movements
- **Configuration management**: JSON-based hardware and experiment configs

---

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing](#testing)

---

## 🏗️ System Architecture

The platform is organized into **four main abstraction layers**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Layer                         │
│  High-level workflows, BO orchestration, kinetic analysis   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Deck & Labware Layer                       │
│     Spatial management, well positions, volume tracking     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Tool Layer                             │
│   Single Syringe, Dual Syringe, Spectrometer, Gripper       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Machine Layer                            │
│    G-code execution, motion control, hardware interface     │
└─────────────────────────────────────────────────────────────┘
```

**Communication**: HTTP REST API to Duet 3 controller (RepRapFirmware)

---

## 💻 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: Modified Jubilee 3D printer with Duet 3 controller
- **Network**: Local network connection to Jubilee (default: 192.168.1.2)

### Dependencies

Install required Python packages:

```bash
# Core scientific computing
pip install numpy scipy scikit-learn pandas

# Visualization
pip install matplotlib python-ternary

# Optional: Jupyter for notebook interface
pip install jupyter
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Axo_MOF
   ```

2. **Install the package** (optional, for development):
   ```bash
   cd Code
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python -c "import numpy, scipy, sklearn, matplotlib; print('✓ Dependencies OK')"
   ```

---

## 🔧 Hardware Requirements

### Core Platform
- **Jubilee 3D printer** (modified for lab automation)
- **Duet 3 controller** running RepRapFirmware
- **Network connection** (Ethernet or WiFi)

### Tools (mounted on tool changer)
- **T0**: Single syringe (precise liquid dispensing)
- **T2**: Dual syringe (parallel dispensing)
- **T3**: Ocean Optics spectrometer (UV-Vis, 200-1100 nm)
- **T4**: Vacuum gripper (lid handling)

### Labware
- SLAS-standard microplates (6-slot deck)
- 10-well sample vials (14 mL capacity)
- 2-well precursor reservoirs (60 mL capacity)

---

## 🚀 Quick Start

### Option 1: Command-Line Interface (Recommended for Production)

**optimization with manual approval:**
```bash
cd Code
python run_bo_optimization.py --new --approval-mode manual
```

**Fully autonomous (automatic mode transitions):**
```bash
python run_bo_optimization.py --new --approval-mode automatic
```

**traditional BO:**
```bash
python run_bo_optimization.py --new --phases single --batch-size 5 --max-iterations 20
```

**Resume interrupted run:**
```bash
python run_bo_optimization.py --resume
```

### Option 2: Jupyter Notebook (Recommended for Interactive Use)

```bash
cd Code
jupyter notebook multi_phase_bo_orchestration.ipynb
```

The notebook provides:
- Step-by-step hardware initialization
- Interactive configuration
- In situ monitoring
- Built-in visualization and analysis

---

## 📖 Usage

### Adaptive Execution

The system runs through **two optimization modes**:

| Mode | Goal | Batch Size | Measurement Interval (min) | Exploration (xi) | 
|-------|------|------------|----------------------------|------------------|
| **Exploration** | Broad screening | 5 | 15 | 0.01 | 
| **Refinement** | Focused optimization | 2 | 5 | 0.01 | 

**Mode Transition Criteria:**
- Switch from exploration to refinement when any sample reaches early reaction progress `α(t1) > 0.5`

### Command-Line Arguments

```bash
python run_bo_optimization.py [OPTIONS]

Required (choose one):
  --new                   Start new optimization
  --resume                Resume from saved state

Multi-Phase Options:
  --approval-mode {manual,automatic}
                          Phase transition approval mode (default: manual)
  --phases {multi,single}
                          Phase configuration (default: multi)

Single-Phase Options (ignored in multi-phase):
  --batch-size N          Experiments per batch (default: 5)
  --max-iterations N      Maximum iterations (default: 20)

General Options:
  --output-dir DIR        Output directory (default: optimization_results)
  --operator NAME         Operator name (default: Operator)
```

### Workflow Example

```bash
# Start new multi-phase optimization with manual approval
python run_bo_optimization.py --new --approval-mode manual --operator "Alice"

# System will:
# 1. Generate 5 initial samples (maximin sampling)
# 2. Prompt you to load vials
# 3. Run automated synthesis + spectroscopy
# 4. Extract kinetic descriptors using Gualtieri fitting
# 5. Update GP model
# 6. Enter exploration phase (5 vials, 15 min interval)
# 7. Prompt for approval when criteria met
# 8. Transition to refinement phase (2 vials, 5 min interval)
# 9. Converge and report optimal conditions
```

### Jupyter Notebook Workflow

See [multi_phase_bo_orchestration.ipynb](Code/multi_phase_bo_orchestration.ipynb) for:
- Interactive hardware setup and calibration
- Configuration with validation
- Progress visualization
- Post-optimization analysis and plotting

---

## 📁 Project Structure

```
Axo_MOF/
├── Code/
│   ├── src/
│   │   └── science_jubilee/
│   │       ├── Machine.py              # Core machine controller
│   │       ├── Experiment.py           # High-level experiment workflows
│   │       ├── tools/                  # Tool implementations
│   │       │   ├── Syringe.py
│   │       │   ├── Double_Syringe.py
│   │       │   ├── Oceandirect_axo.py  # Spectrometer
│   │       │   └── Vacuum_Gripper.py
│   │       ├── labware/                # Labware definitions
│   │       ├── decks/                  # Deck configurations
│   │       ├── analysis/               # Data analysis modules
│   │       │   └── gualtieri.py        # Kinetic fitting
│   │       ├── optimization/           # Bayesian optimization
│   │       │   ├── bayesian.py         # GP modeling, EI
│   │       │   ├── sampling.py         # Initial sampling
│   │       │   ├── phase_config.py     # Phase definitions
│   │       │   ├── phase_manager.py    # Phase transitions
│   │       │   ├── orchestrator.py     # Main BO loop
│   │       │   ├── convergence.py      # Stopping criteria
│   │       │   ├── notifications.py    # User interface
│   │       │   └── logging_config.py   # Structured logging
│   │       └── utils/
│   │           └── synthesis_plan.py   # Experiment plan generation
│   │
│   ├── bayesian.ipynb                  # BO analysis and visualization
│   ├── draft_synthesis_plan.json       # Example synthesis-plan configuration
│   ├── gualtieri.ipynb                 # Kinetic fitting workflow
│   ├── mof_synthesis.ipynb             # Manual synthesis workflows
│   ├── multi_phase_bo_orchestration.ipynb  # Interactive orchestration notebook
│   ├── run_bo_optimization.py          # Main CLI entry point
│   └── sampling.ipynb                  # Initial/maximin sampling workflow
|
├── Dataset/                            # Experimental data
│   └── {experiment_name}_{timestamp}/
│       ├── references/                # Background/reference spectra
│       ├── spectra/                   # Time-resolved spectral outputs
│       ├── experiment_metadata.json   # Experiment-level metadata
│       ├── mof_recipes.json           # Per-vial synthesis recipes
│       └── operations_log.txt         # Chronological robotic execution log
│
└── README.md                           # This file
```

---

## 📚 Documentation

### For Users
- **[Quick Start Notebook](Code/multi_phase_bo_orchestration.ipynb)**: Interactive guide for running experiments

### Example Notebooks
- **[mof_synthesis.ipynb](Code/mof_synthesis.ipynb)**: Manual synthesis workflows and calibration
- **[bayesian.ipynb](Code/bayesian.ipynb)**: BO analysis and visualization
- **[gualtieri.ipynb](Code/gualtieri.ipynb)**: Spectral analysis and kinetic fitting
- **[sampling.ipynb](Code/sampling.ipynb)**: Initial space-filling / maximin sampling
---

## 🧪 Testing

1. **Dry run** (test communication, no synthesis):
   ```bash
   python run_bo_optimization.py --new --phases single --batch-size 1 --max-iterations 1
   # Press Ctrl+C when prompted to load vials
   ```

2. **Single iteration** (full workflow test):
   ```bash
   python run_bo_optimization.py --new --phases single --batch-size 2 --max-iterations 1
   ```

3. **Multi-phase test**:
   ```bash
   python run_bo_optimization.py --new --approval-mode manual
   ```


