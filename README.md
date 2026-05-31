# Math 104C Final Project - Part II

This folder contains the code, tables, figures, and report source for
Project 2: finite-difference methods for partial differential equations.

## Contents

- `src/project2_pde_methods.py`: reusable Python script that implements the PDE
  solvers and generates all numerical results.
- `tests/test_project2_pde_methods.py`: sanity tests for grids, exact solutions,
  stability behavior, and convergence checks.
- `outputs/tables/`: generated CSV and LaTeX tables.
- `outputs/figures/`: generated plots.
- `report/project2_report.tex`: LaTeX source for the PDF report.
- `report/project2_report.pdf`: compiled report after running the build steps.

## Methods Compared

- Five-point finite-difference method for elliptic PDEs
- Successive over-relaxation iteration for elliptic linear systems
- Forward-Difference method for the heat equation
- Backward-Difference method for the heat equation
- Crank-Nicolson method for the heat equation

## Reproduce the Results

From this folder:

```bash
python3 src/project2_pde_methods.py
python3 -m unittest discover -s tests
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report/build report/project2_report.tex
cp report/build/project2_report.pdf report/project2_report.pdf
```

The Python script creates all files in `outputs/`. The LaTeX build creates the
final PDF report.

## Submission Note

Before submitting, replace the placeholder code link in
`report/project2_report.tex` with the repository, Drive, or Colab link that the
instructor can open.
