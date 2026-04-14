#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refined paper-style figure generator for segmentation evaluation CSV.

Input:
    evaluation_results_5folds_full.csv

Output:
    paper_figures_refined/
        01_boxplots_all_metrics.pdf / .svg
        02_grouped_bars_with_ci.pdf / .svg
        03_scatter_correlation_by_model.pdf / .svg
        04_padding_sensitivity.pdf / .svg
        summary_by_model_fold.csv

Design goals:
    - Clean ICCV / Nature / JBHI-style layout
    - Minimal visual noise
    - Muted palette, thin spines, no heavy grid
    - Use fold-wise mean + 95% CI for summary bars
    - Scatter plot emphasizes correlation / feasibility with R²-like annotation
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- Style -------------------------

MODEL_DISPLAY = {
    "ST-SAM": "ST-SAM",
    "MSA_SAM2": "MSA-SAM2",
    "Baseline_SAM2": "Baseline-SAM2",
    "LoRA_SAM2": "LoRA-SAM2",
}

MODEL_ORDER = ["ST-SAM", "MSA_SAM2", "Baseline_SAM2", "LoRA_SAM2"]
MODALITY_ORDER = ["Colour", "Infrared"]

PALETTE = {
    "ST-SAM": "#3B5B92",
    "MSA_SAM2": "#5B8F74",
    "Baseline_SAM2": "#B07A4E",
    "LoRA_SAM2": "#8A6F9E",
    "Colour": "#4C72B0",
    "Infrared": "#DD8452",
}


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 12.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#222222",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "grid.color": "#D9D9D9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def prettify_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def save_fig(fig: plt.Figure, outbase: Path) -> None:
    for ext in ("pdf", "svg"):
        fig.savefig(outbase.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)


# ------------------------- Data helpers -------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in ["Padding", "Model", "Modality"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def model_order_from_data(df: pd.DataFrame) -> List[str]:
    if "Model" not in df.columns:
        return MODEL_ORDER
    order = (
        df.groupby("Model")["Dice"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # keep only known models first, then anything unexpected
    known = [m for m in MODEL_ORDER if m in order]
    extra = [m for m in order if m not in MODEL_ORDER]
    return known + extra


def detect_padding_order(values: Iterable[str]) -> List[str]:
    vals = [str(v).strip() for v in values]
    numeric = []
    special = []
    for v in vals:
        try:
            numeric.append((float(v), v))
        except Exception:
            if v not in special:
                special.append(v)
    numeric_sorted = [v for _, v in sorted(numeric, key=lambda x: x[0])]
    if "YOLO" in special:
        special = [v for v in special if v != "YOLO"] + ["YOLO"]
    return numeric_sorted + [v for v in special if v not in numeric_sorted]


def fold_level_summary(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    per_fold = df.groupby(["Fold", "Model"], as_index=False, observed=True)[list(metrics)].mean(numeric_only=True)
    rows = []
    for model, g in per_fold.groupby("Model"):
        row = {"Model": model}
        for m in metrics:
            vals = g[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals))
            if len(vals) > 1:
                sem = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals)))
                row[f"{m}_ci95"] = 1.96 * sem
            else:
                row[f"{m}_ci95"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return mean, 0.0
    sem = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return mean, 1.96 * sem


def linreg_ci(x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple OLS line + approximate 95% confidence band.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        yhat = np.full_like(xs, np.nan, dtype=float)
        return yhat, yhat, yhat

    xbar = x.mean()
    ybar = y.mean()
    sxx = np.sum((x - xbar) ** 2)
    if sxx <= 0:
        yhat = np.full_like(xs, ybar, dtype=float)
        return yhat, yhat, yhat

    b1 = np.sum((x - xbar) * (y - ybar)) / sxx
    b0 = ybar - b1 * xbar
    yhat = b0 + b1 * xs

    resid = y - (b0 + b1 * x)
    s2 = np.sum(resid ** 2) / max(n - 2, 1)
    se = np.sqrt(s2 * (1.0 / n + (xs - xbar) ** 2 / sxx))
    band = 1.96 * se
    return yhat, yhat - band, yhat + band


# ------------------------- Plot 1: boxplots -------------------------

def plot_boxplots_all_metrics(df: pd.DataFrame, outdir: Path) -> None:
    metrics = ["Dice", "IoU", "Recall", "Precision", "HD95", "ASD"]
    model_order = model_order_from_data(df)

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.2))
    axes = axes.ravel()
    modalities = [m for m in MODALITY_ORDER if m in df["Modality"].unique()] if "Modality" in df.columns else []

    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_axis_off()
            continue

        x = np.arange(len(model_order))
        if modalities and len(modalities) >= 2:
            offsets = np.linspace(-0.18, 0.18, len(modalities))
            box_width = 0.28
            handles = []
            for i, mod in enumerate(modalities):
                series = [
                    df[(df["Model"] == model) & (df["Modality"] == mod)][metric].dropna().to_numpy()
                    for model in model_order
                ]
                bp = ax.boxplot(
                    series,
                    positions=x + offsets[i],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="#1A1A1A", linewidth=1.2),
                    whiskerprops=dict(color="#666666", linewidth=0.9),
                    capprops=dict(color="#666666", linewidth=0.9),
                    boxprops=dict(linewidth=0.9, color="#3A3A3A"),
                )
                color = PALETTE.get(mod, "#999999")
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                    patch.set_edgecolor("#2D2D2D")
                    patch.set_linewidth(0.9)
                handles.append(plt.Line2D([0], [0], color=color, lw=7, alpha=0.7))
        else:
            series = [df[df["Model"] == model][metric].dropna().to_numpy() for model in model_order]
            bp = ax.boxplot(
                series,
                positions=x,
                widths=0.42,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="#1A1A1A", linewidth=1.2),
                whiskerprops=dict(color="#666666", linewidth=0.9),
                capprops=dict(color="#666666", linewidth=0.9),
                boxprops=dict(linewidth=0.9, color="#3A3A3A"),
            )
            handles = []
            for patch, model in zip(bp["boxes"], model_order):
                color = PALETTE.get(model, "#999999")
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
                patch.set_edgecolor("#2D2D2D")
                patch.set_linewidth(0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_order], rotation=0)
        ax.set_title(metric, pad=8)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        prettify_spines(ax)

        if metric in {"Dice", "IoU", "Recall", "Precision"}:
            ax.set_ylim(0.0, 1.02)
        else:
            ax.set_yscale("symlog", linthresh=1.0)
            ax.tick_params(axis="y", which="minor", length=0)

    if modalities and len(modalities) >= 2:
        fig.legend(
            handles,
            modalities[:len(handles)],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(4, len(modalities)),
            frameon=False,
            handlelength=1.8,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_fig(fig, outdir / "01_boxplots_all_metrics")
    plt.close(fig)


# ------------------------- Plot 2: grouped bars -------------------------

def plot_grouped_bars(df: pd.DataFrame, outdir: Path) -> None:
    model_order = model_order_from_data(df)
    high_metrics = ["Dice", "IoU", "Recall", "Precision"]
    low_metrics = ["HD95", "ASD"]

    high_summary = fold_level_summary(df, high_metrics)
    low_summary = fold_level_summary(df, low_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.6))
    x = np.arange(len(model_order))

    # Panel A: higher-is-better
    width = 0.18
    for i, metric in enumerate(high_metrics):
        means = [float(high_summary.loc[high_summary["Model"] == m, f"{metric}_mean"].values[0]) for m in model_order]
        errs = [float(high_summary.loc[high_summary["Model"] == m, f"{metric}_ci95"].values[0]) for m in model_order]
        shift = (i - (len(high_metrics) - 1) / 2) * width
        axes[0].bar(
            x + shift,
            means,
            width=width,
            yerr=errs,
            capsize=3.5,
            color="#7A8CA8",
            edgecolor="#2B2B2B",
            linewidth=0.6,
            alpha=0.92,
            label=metric,
            error_kw=dict(elinewidth=0.9, ecolor="#333333"),
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_order])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Model")
    axes[0].set_title("Higher-is-better", pad=10)
    axes[0].grid(True, axis="y")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    prettify_spines(axes[0])
    axes[0].text(
        -0.10, 1.02, "A",
        transform=axes[0].transAxes,
        fontweight="bold",
        fontsize=12,
        va="bottom",
        ha="right",
    )

    # Panel B: lower-is-better
    width2 = 0.28
    for i, metric in enumerate(low_metrics):
        means = [float(low_summary.loc[low_summary["Model"] == m, f"{metric}_mean"].values[0]) for m in model_order]
        errs = [float(low_summary.loc[low_summary["Model"] == m, f"{metric}_ci95"].values[0]) for m in model_order]
        shift = (i - (len(low_metrics) - 1) / 2) * width2
        axes[1].bar(
            x + shift,
            means,
            width=width2,
            yerr=errs,
            capsize=3.5,
            color="#B39A7D",
            edgecolor="#2B2B2B",
            linewidth=0.6,
            alpha=0.92,
            label=metric,
            error_kw=dict(elinewidth=0.9, ecolor="#333333"),
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_order])
    axes[1].set_ylabel("Distance (pixels)")
    axes[1].set_xlabel("Model")
    axes[1].set_title("Lower-is-better", pad=10)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].grid(True, axis="y")
    axes[1].legend(frameon=False, loc="upper left")
    prettify_spines(axes[1])
    axes[1].text(
        -0.10, 1.02, "B",
        transform=axes[1].transAxes,
        fontweight="bold",
        fontsize=12,
        va="bottom",
        ha="right",
    )

    fig.subplots_adjust(top=0.89, wspace=0.26)
    fig.text(
        0.5, 0.975,
        "Fold-level mean ± 95% CI",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    save_fig(fig, outdir / "02_grouped_bars_with_ci")
    plt.close(fig)


# ------------------------- Plot 3: scatter / correlation -------------------------

def plot_scatter_correlation(df: pd.DataFrame, outdir: Path) -> None:
    """
    Faceted correlation plot:
      x = ASD, y = Dice
      one panel per model
      annotate Pearson r and R²
    """
    model_order = model_order_from_data(df)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 11.0), sharex=True, sharey=True)
    axes = axes.ravel()

    # Robust axis range
    x_all = df["ASD"].to_numpy(dtype=float)
    y_all = df["Dice"].to_numpy(dtype=float)
    x_all = x_all[np.isfinite(x_all)]
    y_all = y_all[np.isfinite(y_all)]
    x_min = max(0.0, float(np.nanpercentile(x_all, 1)))
    x_max = float(np.nanpercentile(x_all, 99.2))
    y_min = float(np.nanpercentile(y_all, 1))
    y_max = float(np.nanpercentile(y_all, 99.8))

    for ax, model in zip(axes, model_order):
        sub = df[df["Model"] == model][["ASD", "Dice"]].dropna().copy()
        x = sub["ASD"].to_numpy(dtype=float)
        y = sub["Dice"].to_numpy(dtype=float)

        # Subsample only for display; fit uses all points
        display_n = 6500
        if len(sub) > display_n:
            sub = sub.sample(display_n, random_state=42)
        color = PALETTE.get(model, "#4C72B0")

        ax.scatter(
            sub["ASD"].to_numpy(dtype=float),
            sub["Dice"].to_numpy(dtype=float),
            s=7,
            alpha=0.14,
            color=color,
            edgecolors="none",
            rasterized=True,
        )

        if len(x) >= 3 and np.nanstd(x) > 0:
            xs = np.linspace(float(np.nanpercentile(x, 1)), float(np.nanpercentile(x, 99)), 200)
            yhat, ylo, yhi = linreg_ci(x, y, xs)
            ax.plot(xs, yhat, color=color, linewidth=1.8)
            ax.fill_between(xs, ylo, yhi, color=color, alpha=0.12, linewidth=0)

            r = float(np.corrcoef(x, y)[0, 1])
            r2 = r ** 2
            ax.text(
                0.04, 0.06,
                f"$r$ = {r:.3f}\n$R^2$ = {r2:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
            )

        ax.set_title(MODEL_DISPLAY.get(model, model), pad=6)
        ax.set_xlabel("ASD (pixels)")
        ax.set_ylabel("Dice")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(max(0.0, y_min), min(1.0, y_max))
        ax.grid(True, alpha=0.55)
        prettify_spines(ax)
        ax.text(
            0.02, 0.98, model.replace("_", "-"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color="#555555",
        )

    fig.tight_layout()
    save_fig(fig, outdir / "03_scatter_correlation_by_model")
    plt.close(fig)


# ------------------------- Plot 4: padding sensitivity -------------------------

def plot_padding_sensitivity(df: pd.DataFrame, outdir: Path) -> None:
    if "Padding" not in df.columns:
        warnings.warn("No Padding column found; skip padding plot.")
        return

    model_order = model_order_from_data(df)
    pad_order = detect_padding_order(df["Padding"].dropna().unique())
    metrics = ["Dice", "IoU", "Recall", "Precision"]

    trend = df.copy()
    trend["Padding"] = pd.Categorical(trend["Padding"], categories=pad_order, ordered=True)
    fold_pad = trend.groupby(["Fold", "Model", "Padding"], as_index=False, observed=True)[metrics].mean(numeric_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2), sharex=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        for model in model_order:
            sub = fold_pad[fold_pad["Model"] == model].copy()
            g = sub.groupby("Padding", observed=True)[metric]
            means = g.mean()
            stds = g.std(ddof=1)
            xs = np.arange(len(pad_order), dtype=float)
            ys = np.array([means.get(p, np.nan) for p in pad_order], dtype=float)
            es = np.array([stds.get(p, np.nan) for p in pad_order], dtype=float)
            fold_n = max(1, sub["Fold"].nunique())
            ci = 1.96 * es / math.sqrt(fold_n)

            color = PALETTE.get(model, "#4C72B0")
            ax.plot(xs, ys, linewidth=1.8, marker="o", markersize=4.3, color=color, label=MODEL_DISPLAY.get(model, model))
            ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.12, linewidth=0)

        ax.set_xticks(np.arange(len(pad_order)))
        ax.set_xticklabels(pad_order)
        ax.set_xlabel("Padding")
        ax.set_ylabel(metric)
        ax.set_title(metric, pad=8)
        ax.grid(True, axis="y")
        ax.set_ylim(0.0, 1.05)
        prettify_spines(ax)

    axes[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    save_fig(fig, outdir / "04_padding_sensitivity")
    plt.close(fig)


# ------------------------- Summary -------------------------

def export_summary(df: pd.DataFrame, outdir: Path) -> None:
    metrics = ["Dice", "IoU", "Recall", "Precision", "HD95", "ASD"]
    fold_summary = df.groupby(["Fold", "Model"], as_index=False)[metrics].mean(numeric_only=True)
    fold_summary.to_csv(outdir / "summary_by_model_fold.csv", index=False)


# ------------------------- Main -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="evaluation_results_5folds_full.csv")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = Path(args.output_dir) if args.output_dir else csv_path.parent / "paper_figures_refined"
    outdir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    df = normalize_columns(pd.read_csv(csv_path))

    required = {"Fold", "Model", "Dice", "IoU", "Recall", "Precision", "HD95", "ASD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    export_summary(df, outdir)
    plot_boxplots_all_metrics(df, outdir)
    plot_grouped_bars(df, outdir)
    plot_scatter_correlation(df, outdir)
    plot_padding_sensitivity(df, outdir)

    print(f"Figures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
