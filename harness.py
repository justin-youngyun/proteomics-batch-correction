"""The harness: run methods per stratum, score them, pick a winner, keep A and B.

Two matrices are kept on purpose and they are not interchangeable:

  Matrix A  uncorrected (median-normalized only). This is the correct input for
            downstream differential expression, where batch belongs in the
            model as a covariate. Correcting first would double-count it.
  Matrix B  corrected, for visualization, clustering, and QC, where you cannot
            put batch in a model and want it physically removed.

Modes:
  comparison  run every requested method, build the metric table, auto-select.
  production  run only the default method, no comparison.

Correction is done independently per stratum; the reference batch is chosen per
stratum from its bridge subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from batch_correct import DEFAULT_METHOD, METHOD_REGISTRY
from common import detect_bridges, median_normalize
from qc_metrics import compute_all

UNCORRECTED_LABEL = "uncorrected"

TIER1_COLUMNS = ["bridge_mad", "pca_batch_r2", "bio_preservation"]
ALL_METRIC_COLUMNS = TIER1_COLUMNS + [
    "group_separation",
    "missingness_preserved",
    "bridge_coverage",
]


@dataclass
class HarnessResult:
    matrix_A: pd.DataFrame
    matrix_B: pd.DataFrame
    metrics: pd.DataFrame
    selected_method: str
    ranking: list = field(default_factory=list)
    disqualified: list = field(default_factory=list)
    ref_batches: dict = field(default_factory=dict)
    mode: str = "comparison"


def build_matrix_A(X):
    """Matrix A: per-sample median-normalized, still uncorrected."""
    return median_normalize(X)


def run(X_A, meta, mode="comparison", methods=None, default_method=DEFAULT_METHOD):
    """Run the harness on a prepared Matrix A.

    Returns a HarnessResult. Matrix B is guaranteed to share Matrix A's shape,
    index, and columns.
    """
    if mode not in ("comparison", "production"):
        raise ValueError(f"unknown mode: {mode!r}")
    if mode == "production":
        methods = [default_method]
    elif methods is None:
        methods = list(METHOD_REGISTRY.keys())
    for m in methods:
        if m not in METHOD_REGISTRY:
            raise KeyError(f"unknown method: {m!r} (have {list(METHOD_REGISTRY)})")

    strata = list(pd.unique(meta["Stratum"]))
    ref_batches = {}
    corrected = {m: {} for m in methods}     # method -> {stratum: corrected df}
    rows = []

    for stratum in strata:
        samples = meta.index[meta["Stratum"] == stratum]
        X_str = X_A[samples]
        meta_str = meta.loc[samples]
        ref = detect_bridges(meta_str).ref_batch
        ref_batches[stratum] = ref

        # uncorrected baseline row (Matrix A scored against itself)
        rows.append(_metric_row(UNCORRECTED_LABEL, stratum, X_str, X_str, meta_str, ref))

        for m in methods:
            Xc = METHOD_REGISTRY[m](X_str, meta_str, ref)
            Xc = Xc.reindex(index=X_str.index, columns=X_str.columns)
            corrected[m][stratum] = Xc
            rows.append(_metric_row(m, stratum, Xc, X_str, meta_str, ref))

    metrics = pd.DataFrame(rows, columns=["method", "stratum"] + ALL_METRIC_COLUMNS)

    if mode == "production":
        selected, ranking, disq = default_method, [default_method], []
    else:
        selected, ranking, disq = select_method(metrics, methods)

    matrix_B = _assemble(X_A, corrected[selected])
    assert matrix_B.shape == X_A.shape, "Matrix B shape differs from Matrix A"
    assert list(matrix_B.index) == list(X_A.index), "Matrix B index differs from Matrix A"
    assert list(matrix_B.columns) == list(X_A.columns), "Matrix B columns differ from Matrix A"

    return HarnessResult(
        matrix_A=X_A,
        matrix_B=matrix_B,
        metrics=metrics,
        selected_method=selected,
        ranking=ranking,
        disqualified=disq,
        ref_batches=ref_batches,
        mode=mode,
    )


def _metric_row(method, stratum, X_corr, X_unc, meta_str, ref):
    row = {"method": method, "stratum": stratum}
    row.update(compute_all(X_corr, X_unc, meta_str, ref))
    return row


def _assemble(X_A, per_stratum):
    """Stitch per-stratum corrected matrices back into one full matrix."""
    out = X_A.copy()
    for _stratum, df in per_stratum.items():
        out.loc[:, df.columns] = df.to_numpy()
    return out.reindex(index=X_A.index, columns=X_A.columns)


def select_method(metrics, methods):
    """Apply the decision rule and return (selected, ranking, disqualified).

    1. disqualify any method that breaks missingness preservation on any stratum
    2. among the rest, minimize average bridge_mad
    3. tiebreak on lower average pca_batch_r2
    4. tiebreak on higher average bio_preservation
    """
    disqualified = []
    for m in methods:
        sub = metrics[metrics["method"] == m]
        if not bool(sub["missingness_preserved"].all()):
            disqualified.append(m)
    eligible = [m for m in methods if m not in disqualified]
    if not eligible:                              # nothing preserved missingness
        eligible = list(methods)
        disqualified = []

    def key(m):
        sub = metrics[metrics["method"] == m]
        return (
            _nanmean(sub["bridge_mad"]),
            _nanmean(sub["pca_batch_r2"]),
            -_nanmean(sub["bio_preservation"]),
        )

    ranking = sorted(eligible, key=key)
    full_ranking = ranking + [m for m in methods if m in disqualified]
    return ranking[0], full_ranking, disqualified


def _nanmean(series):
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(vals)):
        return np.inf
    return float(np.nanmean(vals))


def method_summary(metrics, methods=None):
    """Per-method averages across strata, in ranked order, for reporting."""
    if methods is None:
        methods = [m for m in metrics["method"].unique() if m != UNCORRECTED_LABEL]
    rows = []
    for m in list(metrics["method"].unique()):
        sub = metrics[metrics["method"] == m]
        rows.append({
            "method": m,
            "bridge_mad": _nanmean(sub["bridge_mad"]),
            "pca_batch_r2": _nanmean(sub["pca_batch_r2"]),
            "bio_preservation": _nanmean(sub["bio_preservation"]),
            "group_separation": _nanmean(sub["group_separation"]),
            "missingness_ok": bool(sub["missingness_preserved"].all()),
            "bridge_coverage": _nanmean(sub["bridge_coverage"]),
        })
    return pd.DataFrame(rows).set_index("method")
