# MCCFR

Monte Carlo Counterfactual Regret Minimization final project.

## Contents

- `src/`: self-contained Kuhn Poker, vanilla CFR, external-sampling MCCFR, and evaluation helpers.
- `tests/`: unit tests and convergence sanity checks.
- `notebooks/01_external_sampling_mccfr_kuhn_poker.ipynb`: experiment notebook.
- `report/main.tex`: project report.

## Run

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_external_sampling_mccfr_kuhn_poker.ipynb --ExecutePreprocessor.timeout=600
pdflatex -interaction=nonstopmode main
```

Run the LaTeX command from `report/`.
