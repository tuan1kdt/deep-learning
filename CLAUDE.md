# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.14 with a local virtual environment at `.venv/`. Always activate it before running scripts:

```bash
source .venv/bin/activate
```

Run any day's script directly:

```bash
python day1/01_linear_regression_mse.py
```

Generated plot files (`*.png`) are gitignored and recreated by running the script.

## Repository structure

This is a deep learning study journal. Each `dayN/` directory contains:
- One or more numbered Python scripts (`NN_topic_name.py`) — standalone, executable examples built from scratch with NumPy (no high-level frameworks unless introduced later).
- A `README.md` written in Vietnamese that covers the day's theory: core concepts, relevant math, loss functions, and homework assignments.

Scripts are self-contained teaching examples: they define the model, loss, gradients, and training loop inline, then print results and save a plot. There is no shared library code between days.

## Conventions

- **No ML frameworks** for fundamentals — NumPy only, unless a day explicitly introduces PyTorch/JAX/etc.
- Matplotlib uses the `"Agg"` headless backend so plots render to `.png` files without a GUI.
- Each script is meant to be read top-to-bottom as a tutorial; inline comments explain the *why* behind each step.
- New days follow the same pattern: `dayN/NN_topic.py` + `dayN/README.md`.
