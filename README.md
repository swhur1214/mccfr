# MCCFR

Monte Carlo Counterfactual Regret Minimization final project.

## Contents

- `src/`: self-contained Kuhn/Leduc Poker, vanilla CFR, external/outcome-sampling MCCFR, and evaluation helpers.
- `tests/`: unit tests and convergence sanity checks.
- `notebooks/01_external_sampling_mccfr_kuhn_poker.ipynb`: Kuhn sanity-check experiment.
- `notebooks/02_external_sampling_mccfr_leduc_poker.ipynb`: main Leduc experiment comparing CFR, external sampling, and outcome sampling.
- `report/main.tex`: project report.

## Run

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/02_external_sampling_mccfr_leduc_poker.ipynb --ExecutePreprocessor.timeout=900
pdflatex -interaction=nonstopmode main
```

Run the LaTeX command from `report/`.
