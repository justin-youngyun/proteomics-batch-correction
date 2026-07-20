"""Run the harness on your own CSVs.

    python run.py --abundance data/abundance.csv --samples data/samples.csv \
        --outdir outputs

With no --method it runs comparison mode (all methods, auto-select). With
--method NAME it runs production mode using that method as the fixed default.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from batch_correct import METHOD_LABELS, METHOD_REGISTRY
from common import load_dataset
from harness import build_matrix_A, method_summary, run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abundance", required=True, help="abundance.csv (Protein + samples)")
    ap.add_argument("--samples", required=True, help="samples.csv (Sample, Batch, Group, Subject, Stratum)")
    ap.add_argument("--outdir", default="outputs", help="where to write matrices and metrics")
    ap.add_argument("--method", default=None, choices=sorted(METHOD_REGISTRY),
                    help="fixed method (production mode); omit for comparison mode")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X, meta = load_dataset(args.abundance, args.samples)
    X_A = build_matrix_A(X)

    if args.method:
        result = run(X_A, meta, mode="production", default_method=args.method)
    else:
        result = run(X_A, meta, mode="comparison")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(result.metrics.to_string(index=False))
        if result.mode == "comparison":
            print("\nRanking: " + " > ".join(result.ranking))
            print(method_summary(result.metrics).to_string())
    print(f"\nSelected method: {result.selected_method} "
          f"({METHOD_LABELS[result.selected_method]})")

    X_A.to_csv(os.path.join(args.outdir, "matrix_A_uncorrected.csv"))
    result.matrix_B.to_csv(os.path.join(args.outdir, "matrix_B_corrected.csv"))
    result.metrics.to_csv(os.path.join(args.outdir, "qc_metrics.csv"), index=False)
    print(f"Wrote Matrix A, Matrix B, and qc_metrics.csv to {args.outdir}")


if __name__ == "__main__":
    main()
