# MEDIDA: Model Error Discovery with Iterative Data Assimilation

MEDIDA is an object-oriented Python framework for discovering and correcting structural errors in dynamical systems, specifically focused on epidemiology and physics models. It leverages **Sparse Bayesian Regression** (via Relevance Vector Machines) and **Data Assimilation** (via the Ensemble Kalman Filter) to bridge the gap between imperfect theoretical models and observed data.

## Features

- **Structural Discovery**: Identify missing terms in ODE and PDE systems by learning corrections from sparse, noisy observations.
- **Library-Based Regression**: Support for polynomial, PDE (spectral), and custom nonlinear feature libraries.
- **Data Assimilation**: Integrated Ensemble Kalman Filter (EnKF) for robust state estimation and denoising in high-noise environments.
- **High-Performance PDE Support**: Specialized `KSSystem` implementation using **ETDRK4** for stable integration of the Kuramoto-Sivashinsky equation.
- **Real-World Analysis**: Built-in support for analyzing COVID-19 pandemic data from Our World in Data (OWID).

## Project Structure

- `medida/`: Core library package.
  - `framework.py`: MEDIDA core discovery logic and Result classes.
  - `systems.py`: Definitions of dynamical systems (SIR family, Lorenz, KS).
  - `libraries.py`: Feature libraries for symbolic regression.
  - `regression.py`: RVM and Ridge-based sparse Bayesian regression.
  - `assimilation.py`: Data assimilation algorithms (EnKF).
  - `integrators.py`: Numerical solvers (RK4, Euler, Batch integration).
  - `metrics.py`: Mathematical error metrics (normalized Eq. 16) and formatting.
- `scripts/`: Verification scripts and real-world analysis.
- `outputs/`: Generated diagnostic plots and results.

## Installation

### 1. Set Up the Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
cd medida_epidemiology

# Create a virtual environment
python3 -m venv .venv

# Activate the environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
 (if needed)

```bash
pip install -r requirements.txt
```

## Usage

### Run Mathematical Verification

Reproduces the synthetic verification cases from the MEDIDA paper (SIR, SIRS, SIRD, Lorenz-63, and KS PDE):

```bash
python scripts/verification.py
```

### Run COVID-19 Analysis

Analyzes historical data for Italy to discover structural corrections for a naive SIR model:

```bash
python scripts/covid_analysis.py
```

Results and diagnostic plots will be saved in `outputs/figures/`.

## References

This project is a modular implementation of the research described in:

- **MEDIDA Framework**: [Discovery of interpretable structural model errors by combining Bayesian sparse regression and data assimilation](https://doi.org/10.1063/5.0091282) (Mojgani et al., 2022).
- **EnKF Tutorial**: [Understanding the Ensemble Kalman Filter](http://dx.doi.org/10.1080/00031305.2016.1141709) (Katzfuss et al., 2016).
- **KS PDE Integration**: *Exponential Time Differencing Runge-Kutta Methods* (Cox & Matthews, 2002).
