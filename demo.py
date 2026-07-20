"""End-to-end demo on planted synthetic data.

Generates a dataset, builds Matrix A, runs the harness in comparison mode,
prints the metric table / ranking / selected method, prints the before-after
change in the Tier-1 metrics, and writes Matrix A, Matrix B, the metric table,
and a couple of QC figures into an output directory.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import synthetic
from common import detect_bridges, load_dataset
from batch_correct import METHOD_LABELS
from harness import UNCORRECTED_LABEL, build_matrix_A, method_summary, run

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
SEED = 7


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating synthetic data (seed %d) ..." % SEED)
    abundance_path, samples_path = synthetic.write(DATA_DIR, seed=SEED)
    X, meta = load_dataset(abundance_path, samples_path)
    print(f"  {X.shape[0]} proteins x {X.shape[1]} samples, "
          f"{meta['Batch'].nunique()} batches, {meta['Stratum'].nunique()} strata")
    _report_bridges(meta)

    X_A = build_matrix_A(X)
    result = run(X_A, meta, mode="comparison")

    print("\n=== Metric table (one row per method x stratum) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(result.metrics.to_string(index=False))

    print("\n=== Per-method averages across strata (ranked) ===")
    summary = method_summary(result.metrics).loc[
        [m for m in result.ranking] + [UNCORRECTED_LABEL]
    ]
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print(summary.to_string())

    print("\nRanking (best first): " + " > ".join(result.ranking))
    if result.disqualified:
        print("Disqualified (broke missingness): " + ", ".join(result.disqualified))
    print(f"Selected method: {result.selected_method} "
          f"({METHOD_LABELS[result.selected_method]})")

    _before_after(result)

    # write matrices and metric table
    X_A.to_csv(os.path.join(OUT_DIR, "matrix_A_uncorrected.csv"))
    result.matrix_B.to_csv(os.path.join(OUT_DIR, "matrix_B_corrected.csv"))
    result.metrics.to_csv(os.path.join(OUT_DIR, "qc_metrics.csv"), index=False)

    _figure_pca(result, meta, os.path.join(OUT_DIR, "pca_batch_before_after.png"))
    _figure_bridge(result, meta, os.path.join(OUT_DIR, "bridge_scatter_before_after.png"))

    print(f"\nWrote matrices, metric table, and figures to {OUT_DIR}")


def _report_bridges(meta):
    for stratum in pd.unique(meta["Stratum"]):
        br = detect_bridges(meta.loc[meta["Stratum"] == stratum])
        print(f"  stratum {stratum}: {len(br.bridge_subjects)} bridge subjects, "
              f"reference batch = {br.ref_batch}")


def _before_after(result):
    summary = method_summary(result.metrics)
    before = summary.loc[UNCORRECTED_LABEL]
    after = summary.loc[result.selected_method]
    print("\n=== Before / after (average across strata) ===")
    print(f"{'metric':<20}{'uncorrected':>14}{'corrected':>14}   direction")
    for metric, direction, good in (
        ("bridge_mad", "lower is better", after["bridge_mad"] < before["bridge_mad"]),
        ("pca_batch_r2", "lower is better", after["pca_batch_r2"] < before["pca_batch_r2"]),
        ("bio_preservation", "higher is better",
         after["bio_preservation"] >= before["bio_preservation"] - 0.02),
    ):
        flag = "ok" if good else "CHECK"
        print(f"{metric:<20}{before[metric]:>14.4f}{after[metric]:>14.4f}   "
              f"{direction} [{flag}]")


def _complete_case(matrix, samples):
    arr = matrix[samples].to_numpy(dtype=float)
    complete = ~np.isnan(arr).any(axis=1)
    return arr[complete, :].T                      # samples x proteins


def _figure_pca(result, meta, path):
    samples = list(result.matrix_A.columns)
    batch = meta.loc[samples, "Batch"].to_numpy()
    batches = sorted(np.unique(batch))
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, matrix, title in (
        (axes[0], result.matrix_A, "Matrix A (uncorrected)"),
        (axes[1], result.matrix_B, f"Matrix B ({result.selected_method})"),
    ):
        M = _complete_case(matrix, samples)
        coords = PCA(n_components=2).fit_transform(M - M.mean(axis=0))
        for i, b in enumerate(batches):
            sel = batch == b
            ax.scatter(coords[sel, 0], coords[sel, 1], s=22, alpha=0.8,
                       color=cmap(i % 10), label=f"batch {b}")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("PCA coloured by batch (batch clusters should collapse after)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _figure_bridge(result, meta, path):
    # pick a bridge subject and two of its batches (reference vs another)
    stratum = pd.unique(meta["Stratum"])[0]
    meta_str = meta.loc[meta["Stratum"] == stratum]
    br = detect_bridges(meta_str)
    if not br.bridge_subjects:
        return
    subj = br.bridge_subjects[0]
    locs = br.by_subject[subj]
    ref_batch = br.ref_batch
    other = next(b for b in sorted(locs) if b != ref_batch)
    s_ref = locs[ref_batch][0]
    s_other = locs[other][0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, matrix, title in (
        (axes[0], result.matrix_A, "before (Matrix A)"),
        (axes[1], result.matrix_B, f"after ({result.selected_method})"),
    ):
        x = matrix[s_ref].to_numpy(dtype=float)
        y = matrix[s_other].to_numpy(dtype=float)
        both = ~np.isnan(x) & ~np.isnan(y)
        ax.scatter(x[both], y[both], s=10, alpha=0.4)
        lo = np.nanmin([x[both].min(), y[both].min()])
        hi = np.nanmax([x[both].max(), y[both].max()])
        ax.plot([lo, hi], [lo, hi], color="k", lw=1)
        mad = float(np.mean(np.abs(x[both] - y[both])))
        ax.set_title(f"{title}\nbridge subject {subj}, MAD={mad:.3f}")
        ax.set_xlabel(f"reference batch {ref_batch}")
        ax.set_ylabel(f"batch {other}")
    fig.suptitle("Bridge replicate agreement (points should tighten onto y = x)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
