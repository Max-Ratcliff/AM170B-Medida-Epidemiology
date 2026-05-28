# MEDIDA: Model Error Discovery with Iterative Data Assimilation

MEDIDA is an object-oriented Python framework for discovering and correcting structural errors in dynamical systems, specifically focused on epidemiology and physics models. It leverages symbolic regression (using Relevance Vector Machines) and Data Assimilation to bridge the gap between imperfect theoretical models and observed data.

## Features

- **Structural Discovery**: Identify missing terms in ODE systems.
- **Library-Based Regression**: Support for polynomial, PDE, and custom feature libraries.
- **Data Assimilation**: Integrated Ensemble Kalman Filter (EnKF) for state estimation in noisy environments.
- **Epidemiology Focus**: Pre-built systems for SIR, SIRS, SIRD, SEIR, and saturated nonlinear incidence models.
- **Physics Models**: Support for Kuramoto-Sivashinsky and Lorenz-63 systems.

## Project Structure

- `medida/`: Core library package.
  - `systems.py`: Definitions of dynamical systems.
  - `libraries.py`: Feature libraries for regression.
  - `regression.py`: RVM and Ridge-based sparse regression.
  - `assimilation.py`: Data assimilation algorithms (EnKF).
  - `framework.py`: MEDIDA core logic.
  - `integrators.py`: Numerical solvers (RK4, Euler).
  - `metrics.py`: Error metrics and formatting tools.
- `examples/`: Verification scripts and usage examples.
- `figures/`: Generated diagnostic plots.

## Installation

It is recommended to use a virtual environment to manage dependencies.

### 1. Create a Virtual Environment

```bash
# Unix/macOS
python3 -m venv venv

# Windows
python -m venv venv
```

### 2. Activate the Environment

```bash
# Unix/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Run Verification Examples

Reproduces the synthetic verification cases for SIR, SIRS, SIRD, and Nonlinear incidence models:

```bash
python examples/run_examples.py
```

### Run Real-World COVID-19 Analysis

Analyzes historical COVID-19 data for Italy to discover structural model errors in a standard SIR model. This script downloads the latest data from Our World in Data (OWID) automatically:

```bash
python examples/covid_analysis.py
```

Diagnostic plots will be saved in the `figures/` directory.

## References

This project is based on the following research:

- **MEDIDA Framework**: [Discovery of interpretable structural model errors by combining Bayesian sparse regression and data assimilation](https://doi.org/10.1063/5.0091282) (Mojgani et al., 2022)
- **EnKF Tutorial**: [Understanding the Ensemble Kalman Filter](http://dx.doi.org/10.1080/00031305.2016.1141709) (Katzfuss et al., 2016)
