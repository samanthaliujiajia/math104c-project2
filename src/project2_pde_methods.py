"""Generate finite-difference PDE experiments for Math 104C Project 2."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "math104c_mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as spla


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLE_DIR = OUTPUTS / "tables"
FIGURE_DIR = OUTPUTS / "figures"

ELLIPTIC_MESHES = (0.25, 0.125, 0.0625)
HEAT_FINAL_TIME = 0.1


@dataclass(frozen=True)
class EllipticProblem:
    key: str
    name: str
    equation: str
    boundary: Callable[[float, float], float]
    source: Callable[[np.ndarray, np.ndarray], np.ndarray]
    exact: Callable[[np.ndarray, np.ndarray], np.ndarray]
    sign: int


@dataclass(frozen=True)
class EllipticResult:
    problem: EllipticProblem
    method: str
    h: float
    x: np.ndarray
    y: np.ndarray
    numerical: np.ndarray
    exact: np.ndarray
    iterations: int | None
    residual: float | None


@dataclass(frozen=True)
class HeatCase:
    key: str
    name: str
    mode: int
    h: float
    k: float
    t_end: float = HEAT_FINAL_TIME

    @property
    def lam(self) -> float:
        return self.k / self.h**2


@dataclass(frozen=True)
class HeatResult:
    case: HeatCase
    method: str
    x: np.ndarray
    t: np.ndarray
    numerical: np.ndarray
    exact: np.ndarray


LAPLACE_PROBLEM = EllipticProblem(
    key="A",
    name="Laplace equation with linear boundary data",
    equation="u_xx + u_yy = 0",
    boundary=lambda x, y: 100.0 * x * y,
    source=lambda x, y: np.zeros_like(x, dtype=float),
    exact=lambda x, y: 100.0 * x * y,
    sign=1,
)

POISSON_PROBLEM = EllipticProblem(
    key="C1",
    name="Poisson equation with manufactured solution",
    equation="u_xx + u_yy = -2 pi^2 sin(pi x) sin(pi y)",
    boundary=lambda x, y: 0.0,
    source=lambda x, y: -2.0 * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y),
    exact=lambda x, y: np.sin(math.pi * x) * np.sin(math.pi * y),
    sign=1,
)

HEAT_CASES = (
    HeatCase("B_stable_h0.1", "Problem B stable forward case", 1, 0.1, 0.004),
    HeatCase("B_refined_h0.05", "Problem B refined stable case", 1, 0.05, 0.001),
    HeatCase("B_unstable_h0.1", "Problem B unstable forward case", 1, 0.1, 0.01),
    HeatCase("C2_stable_h0.1", "Problem C2 different initial condition", 2, 0.1, 0.004),
    HeatCase("C2_refined_h0.05", "Problem C2 refined different initial condition", 2, 0.05, 0.001),
)

REFINEMENT_CASES = {"B_stable_h0.1", "B_refined_h0.05", "C2_stable_h0.1", "C2_refined_h0.05"}


def interior_count(h: float) -> int:
    n = int(round(1.0 / h))
    if not np.isclose(n * h, 1.0):
        raise ValueError(f"h={h} must evenly divide the unit interval")
    return n - 1


def elliptic_grid(h: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(1.0 / h))
    points = np.linspace(0.0, 1.0, n + 1)
    return points, points.copy()


def interior_index(i: int, j: int, n_inner: int) -> int:
    return (j - 1) * n_inner + (i - 1)


def build_elliptic_system(problem: EllipticProblem, h: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    n_inner = interior_count(h)
    size = n_inner * n_inner
    matrix = sparse.lil_matrix((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    for j in range(1, n_inner + 1):
        y = j * h
        for i in range(1, n_inner + 1):
            x = i * h
            row = interior_index(i, j, n_inner)
            matrix[row, row] = -4.0
            rhs[row] = h**2 * float(problem.source(np.array(x), np.array(y)))
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 1 <= ni <= n_inner and 1 <= nj <= n_inner:
                    matrix[row, interior_index(ni, nj, n_inner)] = 1.0
                else:
                    rhs[row] -= problem.boundary(ni * h, nj * h)
    return matrix.tocsr(), rhs


def fill_elliptic_solution(
    problem: EllipticProblem, h: float, interior_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = elliptic_grid(h)
    numerical = np.zeros((len(y), len(x)), dtype=float)
    n_inner = interior_count(h)
    for j, yj in enumerate(y):
        for i, xi in enumerate(x):
            if i == 0 or j == 0 or i == len(x) - 1 or j == len(y) - 1:
                numerical[j, i] = problem.boundary(float(xi), float(yj))
            else:
                numerical[j, i] = interior_values[interior_index(i, j, n_inner)]
    xx, yy = np.meshgrid(x, y)
    exact = problem.exact(xx, yy)
    return x, y, numerical, exact


def solve_elliptic_direct(problem: EllipticProblem, h: float) -> EllipticResult:
    matrix, rhs = build_elliptic_system(problem, h)
    values = spla.spsolve(matrix, rhs)
    x, y, numerical, exact = fill_elliptic_solution(problem, h, values)
    residual = float(np.linalg.norm(matrix @ values - rhs, ord=np.inf))
    return EllipticResult(problem, "Direct", h, x, y, numerical, exact, None, residual)


def solve_elliptic_sor(
    problem: EllipticProblem,
    h: float,
    omega: float = 1.6,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> EllipticResult:
    matrix, rhs = build_elliptic_system(problem, h)
    dense = matrix.toarray()
    values = np.zeros_like(rhs)

    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        previous = values.copy()
        for row in range(len(values)):
            sigma = dense[row, :] @ values - dense[row, row] * values[row]
            gauss_seidel_value = (rhs[row] - sigma) / dense[row, row]
            values[row] = (1.0 - omega) * values[row] + omega * gauss_seidel_value
        residual = float(np.linalg.norm(matrix @ values - rhs, ord=np.inf))
        if residual < tolerance or np.linalg.norm(values - previous, ord=np.inf) < tolerance:
            break

    x, y, numerical, exact = fill_elliptic_solution(problem, h, values)
    return EllipticResult(problem, f"SOR omega={omega:g}", h, x, y, numerical, exact, iteration, residual)


def heat_mesh(case: HeatCase) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(1.0 / case.h))
    m = int(round(case.t_end / case.k))
    if not np.isclose(n * case.h, 1.0):
        raise ValueError(f"h={case.h} must evenly divide [0,1]")
    if not np.isclose(m * case.k, case.t_end):
        raise ValueError(f"k={case.k} must evenly divide [0,{case.t_end}]")
    return np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, case.t_end, m + 1)


def heat_initial(case: HeatCase, x: np.ndarray) -> np.ndarray:
    return np.sin(case.mode * math.pi * x)


def heat_exact(case: HeatCase, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    xx, tt = np.meshgrid(x, t)
    return np.exp(-(case.mode * math.pi) ** 2 * tt) * np.sin(case.mode * math.pi * xx)


def forward_difference(case: HeatCase) -> HeatResult:
    x, t = heat_mesh(case)
    u = np.zeros((len(t), len(x)), dtype=float)
    u[0, :] = heat_initial(case, x)
    lam = case.lam
    for n in range(len(t) - 1):
        u[n + 1, 1:-1] = (
            lam * u[n, :-2] + (1.0 - 2.0 * lam) * u[n, 1:-1] + lam * u[n, 2:]
        )
    return HeatResult(case, "Forward Difference", x, t, u, heat_exact(case, x, t))


def heat_matrix(n_inner: int, lam: float, method: str) -> sparse.csr_matrix:
    off = np.ones(n_inner - 1)
    if method == "backward":
        diagonal = (1.0 + 2.0 * lam) * np.ones(n_inner)
        offdiag = -lam * off
    elif method == "cn_left":
        diagonal = (1.0 + lam) * np.ones(n_inner)
        offdiag = -0.5 * lam * off
    elif method == "cn_right":
        diagonal = (1.0 - lam) * np.ones(n_inner)
        offdiag = 0.5 * lam * off
    else:
        raise ValueError(method)
    return sparse.diags((offdiag, diagonal, offdiag), offsets=(-1, 0, 1), format="csr")


def backward_difference(case: HeatCase) -> HeatResult:
    x, t = heat_mesh(case)
    n_inner = len(x) - 2
    matrix = heat_matrix(n_inner, case.lam, "backward")
    u = np.zeros((len(t), len(x)), dtype=float)
    u[0, :] = heat_initial(case, x)
    for n in range(len(t) - 1):
        u[n + 1, 1:-1] = spla.spsolve(matrix, u[n, 1:-1])
    return HeatResult(case, "Backward Difference", x, t, u, heat_exact(case, x, t))


def crank_nicolson(case: HeatCase) -> HeatResult:
    x, t = heat_mesh(case)
    n_inner = len(x) - 2
    left = heat_matrix(n_inner, case.lam, "cn_left")
    right = heat_matrix(n_inner, case.lam, "cn_right")
    u = np.zeros((len(t), len(x)), dtype=float)
    u[0, :] = heat_initial(case, x)
    for n in range(len(t) - 1):
        u[n + 1, 1:-1] = spla.spsolve(left, right @ u[n, 1:-1])
    return HeatResult(case, "Crank-Nicolson", x, t, u, heat_exact(case, x, t))


HEAT_METHODS = (forward_difference, backward_difference, crank_nicolson)


def final_error(result: EllipticResult | HeatResult) -> float:
    if isinstance(result, EllipticResult):
        return float(np.max(np.abs(result.exact - result.numerical)))
    return float(np.max(np.abs(result.exact[-1, :] - result.numerical[-1, :])))


def max_error(result: EllipticResult | HeatResult) -> float:
    return float(np.max(np.abs(result.exact - result.numerical)))


def observed_orders(summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in summary.groupby(group_columns):
        sorted_group = group.sort_values("h", ascending=False)
        errors = sorted_group["max_error"].to_numpy(dtype=float)
        hs = sorted_group["h"].to_numpy(dtype=float)
        order = float("nan")
        if len(errors) >= 2 and np.all(errors > 0):
            local_orders = np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
            order = float(np.mean(local_orders))
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(dict(zip(group_columns, keys), observed_order=order))
    return pd.DataFrame(rows)


def latex_float(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1000 or (0 < abs(x) < 0.001):
        return f"{x:.3e}"
    return f"{x:.6f}"


def latex_cell(value: object) -> str:
    text = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        text = text.replace(old, new)
    return text


def latex_heading(column: str) -> str:
    headings = {
        "problem": "problem",
        "case": "case",
        "method": "method",
        "h": "$h$",
        "k": "$k$",
        "lambda": "$\\lambda$",
        "final_error": "final error",
        "max_error": "max error",
        "iterations": "iterations",
        "residual": "residual",
        "observed_order": "observed order",
        "x": "$x$",
        "y": "$y$",
        "t": "$t$",
        "exact": "exact",
        "numerical": "numerical",
        "absolute_error": "abs. error",
    }
    return headings.get(column, latex_cell(column))


def write_latex_table(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    formatted = df.loc[:, columns].copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(latex_float)
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule"]
    lines.append(" & ".join(latex_heading(column) for column in columns) + " \\\\")
    lines.append("\\midrule")
    for row in formatted.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_cell(value) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_elliptic(result: EllipticResult) -> None:
    xx, yy = np.meshgrid(result.x, result.y)
    values = [
        ("exact", result.exact),
        ("numerical", result.numerical),
        ("absolute_error", np.abs(result.exact - result.numerical)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    for ax, (title, data) in zip(axes, values):
        mesh = ax.pcolormesh(xx, yy, data, shading="auto", cmap="viridis")
        ax.set_title(title.replace("_", " ").title())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.suptitle(f"{result.problem.key}: {result.method}, h={result.h:g}")
    filename = f"problem_{result.problem.key}_{result.method.split()[0]}_h{result.h:g}".replace(".", "p")
    fig.savefig(FIGURE_DIR / f"{filename}.pdf")
    fig.savefig(FIGURE_DIR / f"{filename}.png", dpi=200)
    plt.close(fig)


def plot_heat_final(results: list[HeatResult], case_key: str) -> None:
    selected = [r for r in results if r.case.key == case_key]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(selected[0].x, selected[0].exact[-1, :], color="black", linewidth=2.0, label="Exact")
    for result in selected:
        ax.plot(result.x, result.numerical[-1, :], marker="o", markersize=3, label=result.method)
    case = selected[0].case
    ax.set_title(f"{case.key}: final time t={case.t_end:g}, lambda={case.lam:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{case.key}_final_profiles.pdf")
    fig.savefig(FIGURE_DIR / f"{case.key}_final_profiles.png", dpi=200)
    plt.close(fig)


def plot_heat_evolution(result: HeatResult) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    mesh = ax.pcolormesh(result.x, result.t, result.numerical, shading="auto", cmap="viridis")
    ax.set_title(f"{result.case.key}: {result.method}")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(mesh, ax=ax, label="u")
    fig.tight_layout()
    filename = f"{result.case.key}_{result.method.replace(' ', '_').replace('-', '_')}_evolution"
    fig.savefig(FIGURE_DIR / f"{filename}.pdf")
    fig.savefig(FIGURE_DIR / f"{filename}.png", dpi=200)
    plt.close(fig)


def plot_convergence(summary: pd.DataFrame, problem: str, filename: str) -> None:
    data = summary[summary["problem"] == problem]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for method, group in data.groupby("method"):
        if group["h"].nunique() < 2:
            continue
        group = group.sort_values("h", ascending=False)
        ax.loglog(group["h"], group["max_error"], marker="o", label=method)
    ax.invert_xaxis()
    ax.set_title(f"{problem}: max error vs mesh size")
    ax.set_xlabel("h")
    ax.set_ylabel("max absolute error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{filename}.pdf")
    fig.savefig(FIGURE_DIR / f"{filename}.png", dpi=200)
    plt.close(fig)


def elliptic_frame(result: EllipticResult) -> pd.DataFrame:
    xx, yy = np.meshgrid(result.x, result.y)
    return pd.DataFrame(
        {
            "problem": result.problem.key,
            "method": result.method,
            "h": result.h,
            "x": xx.ravel(),
            "y": yy.ravel(),
            "exact": result.exact.ravel(),
            "numerical": result.numerical.ravel(),
            "absolute_error": np.abs(result.exact - result.numerical).ravel(),
        }
    )


def heat_frame(result: HeatResult) -> pd.DataFrame:
    xx, tt = np.meshgrid(result.x, result.t)
    return pd.DataFrame(
        {
            "problem": "B" if result.case.key.startswith("B") else "C2",
            "case": result.case.key,
            "method": result.method,
            "h": result.case.h,
            "k": result.case.k,
            "lambda": result.case.lam,
            "x": xx.ravel(),
            "t": tt.ravel(),
            "exact": result.exact.ravel(),
            "numerical": result.numerical.ravel(),
            "absolute_error": np.abs(result.exact - result.numerical).ravel(),
        }
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    elliptic_results: list[EllipticResult] = []
    elliptic_frames: list[pd.DataFrame] = []
    elliptic_summary_rows = []
    for problem in (LAPLACE_PROBLEM, POISSON_PROBLEM):
        for h in ELLIPTIC_MESHES:
            for result in (
                solve_elliptic_direct(problem, h),
                solve_elliptic_sor(problem, h),
            ):
                elliptic_results.append(result)
                elliptic_frames.append(elliptic_frame(result))
                elliptic_summary_rows.append(
                    {
                        "problem": result.problem.key,
                        "method": result.method,
                        "h": result.h,
                        "final_error": final_error(result),
                        "max_error": max_error(result),
                        "iterations": result.iterations if result.iterations is not None else 0,
                        "residual": result.residual if result.residual is not None else 0.0,
                    }
                )
                plot_elliptic(result)

    heat_results: list[HeatResult] = []
    heat_frames: list[pd.DataFrame] = []
    heat_summary_rows = []
    for case in HEAT_CASES:
        for method in HEAT_METHODS:
            result = method(case)
            heat_results.append(result)
            heat_frames.append(heat_frame(result))
            heat_summary_rows.append(
                {
                    "problem": "B" if case.key.startswith("B") else "C2",
                    "case": case.key,
                    "method": result.method,
                    "h": case.h,
                    "k": case.k,
                    "lambda": case.lam,
                    "final_error": final_error(result),
                    "max_error": max_error(result),
                }
            )
            if case.key in {"B_stable_h0.1", "B_unstable_h0.1", "C2_stable_h0.1"}:
                plot_heat_evolution(result)

    for key in {result.case.key for result in heat_results}:
        plot_heat_final(heat_results, key)

    elliptic_all = pd.concat(elliptic_frames, ignore_index=True)
    heat_all = pd.concat(heat_frames, ignore_index=True)
    elliptic_summary = pd.DataFrame(elliptic_summary_rows)
    heat_summary = pd.DataFrame(heat_summary_rows)
    elliptic_orders = observed_orders(elliptic_summary, ["problem", "method"])
    heat_orders = observed_orders(heat_summary[heat_summary["case"].isin(REFINEMENT_CASES)], ["problem", "method"])

    elliptic_all.to_csv(TABLE_DIR / "elliptic_all_mesh_results.csv", index=False)
    heat_all.to_csv(TABLE_DIR / "heat_all_mesh_results.csv", index=False)
    elliptic_summary.to_csv(TABLE_DIR / "elliptic_error_summary.csv", index=False)
    heat_summary.to_csv(TABLE_DIR / "heat_error_summary.csv", index=False)
    elliptic_orders.to_csv(TABLE_DIR / "elliptic_observed_orders.csv", index=False)
    heat_orders.to_csv(TABLE_DIR / "heat_observed_orders.csv", index=False)

    write_latex_table(
        elliptic_summary,
        TABLE_DIR / "elliptic_error_summary.tex",
        ["problem", "method", "h", "max_error", "iterations", "residual"],
    )
    write_latex_table(
        heat_summary,
        TABLE_DIR / "heat_error_summary.tex",
        ["problem", "case", "method", "h", "k", "lambda", "final_error", "max_error"],
    )
    write_latex_table(
        elliptic_orders,
        TABLE_DIR / "elliptic_observed_orders.tex",
        ["problem", "method", "observed_order"],
    )
    write_latex_table(
        heat_orders,
        TABLE_DIR / "heat_observed_orders.tex",
        ["problem", "method", "observed_order"],
    )

    selected_elliptic = elliptic_all[
        (elliptic_all["h"] == 0.125)
        & (elliptic_all["method"] == "Direct")
        & (elliptic_all["x"].isin([0.25, 0.5, 0.75]))
        & (elliptic_all["y"].isin([0.25, 0.5, 0.75]))
    ].sort_values(["problem", "y", "x"])
    selected_heat = heat_all[
        (heat_all["case"].isin(["B_stable_h0.1", "C2_stable_h0.1"]))
        & (heat_all["method"].isin(["Forward Difference", "Backward Difference", "Crank-Nicolson"]))
        & np.isclose(heat_all["t"], HEAT_FINAL_TIME)
    ].sort_values(["problem", "case", "method", "x"])

    write_latex_table(
        selected_elliptic.head(18),
        TABLE_DIR / "selected_elliptic_mesh_values.tex",
        ["problem", "x", "y", "exact", "numerical", "absolute_error"],
    )
    write_latex_table(
        selected_heat.groupby(["problem", "case", "method"]).head(4),
        TABLE_DIR / "selected_heat_mesh_values.tex",
        ["problem", "case", "method", "x", "t", "exact", "numerical", "absolute_error"],
    )

    plot_convergence(elliptic_summary, "A", "problem_A_convergence")
    plot_convergence(elliptic_summary, "C1", "problem_C1_convergence")
    refinement_summary = heat_summary[heat_summary["case"].isin(REFINEMENT_CASES)]
    plot_convergence(refinement_summary, "B", "problem_B_convergence")
    plot_convergence(refinement_summary, "C2", "problem_C2_convergence")

    print(f"Wrote Project 2 results to {OUTPUTS}")


if __name__ == "__main__":
    main()
