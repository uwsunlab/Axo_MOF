# Axo_MOF: Autonomous Multi-Phase Bayesian Optimization for MOF Synthesis

> **An automated Metal-Organic Framework (MOF) synthesis and optimization platform combining robotics, spectroscopy, and machine learning for autonomous materials discovery**

Built on a modified Jubilee 3D printer with integrated liquid handling, UV-Vis spectroscopy, and intelligent experiment planning through multi-phase Bayesian optimization.

---

## 🌟 Key Features

### Multi-Phase Bayesian Optimization
- **3-phase progressive optimization**: Exploration → Refinement → Validation
- **Adaptive batch sizing**: Large batches (10) for exploration, small batches (2) for validation
- **Dynamic search space**: Automatically narrows to promising regions
- **Flexible approval modes**: Manual (interactive) or automatic (fully autonomous)
- **State persistence**: Resume interrupted runs from checkpoints

### Automated Synthesis & Characterization
- **Robotic liquid handling**: Precise dispensing with single/dual syringe systems
- **Real-time monitoring**: In-situ UV-Vis spectroscopy during synthesis
- **Autonomous mixing**: Automated stirring and reaction control
- **Multi-vial workflows**: Parallel synthesis of up to 20 samples

### Intelligent Data Analysis
- **Gualtieri kinetic fitting**: Automated MOF growth curve analysis
- **Yield extraction**: Automatic I_max extraction from spectral time series
- **Gaussian Process modeling**: Uncertainty-aware surrogate models
- **Expected Improvement**: Efficient exploration-exploitation balance

### Production-Ready Engineering
- **Comprehensive logging**: Detailed operation logs and experiment tracking
- **Error recovery**: Graceful handling of failed fits (yield = 0.0)
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
│                    Experiment Layer                          │
│  High-level workflows, BO orchestration, yield extraction   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Deck & Labware Layer                        │
│     Spatial management, well positions, volume tracking     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Tool Layer                              │
│   Single Syringe, Dual Syringe, Spectrometer, Gripper      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Machine Layer                             │
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

**Multi-phase optimization with manual approval:**
```bash
cd Code
python run_bo_optimization.py --new --approval-mode manual
```

**Fully autonomous (automatic phase transitions):**
```bash
python run_bo_optimization.py --new --approval-mode automatic
```

**Single-phase mode (traditional BO):**
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
- Real-time progress monitoring
- Built-in visualization and analysis

---

## 📖 Usage

### Multi-Phase Bayesian Optimization

The system runs through **three optimization phases**:

| Phase | Goal | Batch Size | Exploration (xi) | Iterations |
|-------|------|------------|------------------|------------|
| **Exploration** | Broad screening | 10 | 0.05 (high) | 3-5 |
| **Refinement** | Focused optimization | 5 | 0.01 (medium) | 5-10 |
| **Validation** | Fine-tuning | 2 | 0.001 (low) | 2-5 |

**Phase Transition Criteria:**
- Minimum iterations reached
- Maximum iterations reached (hard limit)
- Performance threshold met (yield > threshold)
- Expected Improvement converged

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
# 4. Extract yields using Gualtieri fitting
# 5. Update GP model
# 6. Enter exploration phase (10 vials × ~5 iterations)
# 7. Prompt for approval when criteria met
# 8. Transition to refinement phase (5 vials × ~10 iterations)
# 9. Transition to validation phase (2 vials × ~5 iterations)
# 10. Converge and report optimal conditions
```

### Jupyter Notebook Workflow

See [multi_phase_bo_orchestration.ipynb](Code/multi_phase_bo_orchestration.ipynb) for:
- Interactive hardware setup and calibration
- Configuration with validation
- Real-time progress visualization
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
│   ├── run_bo_optimization.py          # Main CLI entry point
│   ├── multi_phase_bo_orchestration.ipynb  # Interactive notebook
│   ├── mof_synthesis.ipynb             # Manual synthesis workflows
│   ├── bayesian optimisation.ipynb     # BO analysis & visualization
|
├── Dataset/                            # Experimental data
│   └── {experiment_name}_{timestamp}/
│       ├── operations_log.txt
│       ├── {vial}_{time}min.npy        # Spectral data
│       └── {vial}_{time}min.png
│
└── README.md                           # This file
```

---

## 📚 Documentation

### For Users
- **[Quick Start Notebook](Code/multi_phase_bo_orchestration.ipynb)**: Interactive guide for running experiments

### Example Notebooks
- **[mof_synthesis.ipynb](Code/mof_synthesis.ipynb)**: Manual synthesis workflows and calibration
- **[bayesian optimisation.ipynb](Code/bayesian%20optimisation.ipynb)**: BO analysis and visualization
- **[spectra data and Gualtieri fitting.ipynb](Code/spectra%20data%20and%20Gualtieri%20fitting.ipynb)**: Spectral analysis

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


