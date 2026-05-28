# MEDIDA for Epidemiology

UCSC AM 170B math modeling research project by Alejandro Munoz, Alexander
Spetzler, Anirudh Raja, Matteo Tasso, and Max Ratcliff.

This repository applies MEDIDA to synthetic epidemic models and historical
COVID-19 data. In the 2022 paper,
MEDIDA stands for **Model Error Discovery with Interpretability and Data
Assimilation**. The method was introduced on a chaotic Kuramoto-Sivashinsky
PDE test case, so our project asks whether the same model-error discovery idea
can be moved from a physics setting into epidemiological compartment models.

The main research question is:

> Can MEDIDA be applied beyond physics systems to recover missing or inaccurate
> terms in epidemiological models and improve their predictive accuracy?

The goal is not to forecast current disease conditions. The COVID portion uses
old OWID data from March 1, 2020 through January 15, 2021 as a real-data case
study: can a sparse, interpretable correction learned from historical epidemic
trajectories reduce the one-step error of a simple SIR model?

The project also includes synthetic checks on SIR-family systems, Lorenz-63,
and the Kuramoto-Sivashinsky PDE to show that the implementation can recover
known missing model terms when ground truth is available. The KS example is
especially important because it reproduces the kind of physics validation used
in the original MEDIDA paper before moving to epidemic models.

## Repository Layout

- `medida/`: reusable MEDIDA implementation.
- `scripts/verification.py`: synthetic model-error recovery experiments.
- `scripts/covid_analysis.py`: historical COVID analysis and global transfer
  checks.
- `scripts/utils.py`: shared plotting, experiment, and sweep helpers.
- `scripts/antagonistic_analysis.py`: bias check for improvement factors.
- `data/owid-covid-data.csv`: cached OWID historical COVID dataset.
- `outputs/`: generated figures and CSV summaries.

## Environment

Use the project virtual environment from the repo root:

```bash
cd medida_epidemiology
source .venv/bin/activate
pip install -r requirements.txt
```

The plotting scripts force Matplotlib's noninteractive `Agg` backend, so they
can generate PNGs from a terminal or headless session.

## Reproducing Results

Run all synthetic verification figures:

```bash
python scripts/verification.py
```

Run the historical COVID analysis for any country name available in the OWID
dataset:

```bash
python scripts/covid_analysis.py --train-country "Country Name"
```

Add `--sweep` to evaluate the learned correction across all available OWID
countries:

```bash
python scripts/covid_analysis.py --train-country "Country Name" --sweep
```

The sweep writes `sweep_results.csv`, `global_map.png`,
`top50_landscape.png`, `bottom50_landscape.png`, and
`failure_analysis_grid.png` in the corresponding country folder.

<details>
<summary>Example country calls used in this project</summary>

```bash
python scripts/covid_analysis.py --train-country Italy --sweep
python scripts/covid_analysis.py --train-country Sweden --sweep
python scripts/covid_analysis.py --train-country Germany --sweep
python scripts/covid_analysis.py --train-country India --sweep
python scripts/covid_analysis.py --train-country "South Africa" --sweep
```

</details>

## Modeling Pipeline

1. Build an imperfect model. For COVID, this is a standard SIR model with
   constant transmission rate `beta` and recovery rate `gamma = 1 / 14`.
2. Estimate one-step model error:
   `delta_u = (observed_next - model_step(observed_current)) / dt`.
3. Regress that error against a sparse feature library using RVM-style sparse
   regression.
4. Add the discovered correction to the imperfect model coefficients.
5. Compare the imperfect and corrected one-step predictions.

This follows the paper's main logic: estimate the model-error tendency from
short model integrations, then identify a sparse closed-form correction from a
candidate library. If observations are noisy, an EnKF can be used to estimate a
cleaner analysis state before regression.

For COVID, reported cases are converted into approximate SIR compartments:

- `I(t)`: 14-day rolling sum of smoothed new cases, multiplied by 4 to account
  for undercounting.
- `R(t)`: cumulative estimated infections minus active infectious count.
- `S(t)`: population minus estimated active and recovered counts.

These are simple modeling assumptions for a class project, not clinical
estimates.

## Outputs

Generated results are written under `outputs/`.

Synthetic outputs:

- `outputs/synthetic/sirs/`: SIRS recovery tests.
- `outputs/synthetic/sird/`: SIRD recovery tests.
- `outputs/synthetic/nonlinear_sir/`: nonlinear incidence recovery test.
- `outputs/synthetic/seir_from_sir/`: hidden-exposed-compartment experiments.
- `outputs/synthetic/hidden_e/`: noisy hidden-variable stress test.
- `outputs/synthetic/lorenz/`: Lorenz-63 controlled model-error experiments.
- `outputs/synthetic/ks_pde/`: Kuramoto-Sivashinsky PDE reconstruction.
- `outputs/summary/sweeps/`: parameter-sweep heatmaps across synthetic systems.

COVID outputs:

Each folder under `outputs/covid/{country}/` contains discovered correction
figures, one-step residual plots, transfer checks, and optional global sweep
results. `sweep_results.csv` stores the country-level improvement factors used
by the maps and ranking plots.

This repository includes pre-generated outputs for:

- `outputs/covid/italy/`
- `outputs/covid/sweden/`
- `outputs/covid/germany/`
- `outputs/covid/india/`
- `outputs/covid/south_africa/`

These countries were chosen as representative test cases for our project, but
the analysis script is not limited to them.

## Limitations

- COVID results are based on historical reported-case data and simple
  compartment estimates, not direct measurements of susceptible, infectious,
  and recovered populations.
- The 4x undercount multiplier and 14-day infectious window are modeling
  assumptions.
- Global maps show one-step prediction improvement, not long-term forecasting
  accuracy.
- A correction trained on one country can help many countries but can also fail;
  the bottom-50 and failure-grid plots should be kept visible in the project
  record.
- Data assimilation is not uniformly beneficial in every synthetic setup. In
  the current Lorenz noisy benchmark, direct noisy regression beats the EnKF
  configuration for several perturbations.

## References

- Mojgani et al., "Discovery of interpretable structural model errors by
  combining Bayesian sparse regression and data assimilation", Chaos, 2022.
- Katzfuss et al., "Understanding the Ensemble Kalman Filter", The American
  Statistician, 2016.
