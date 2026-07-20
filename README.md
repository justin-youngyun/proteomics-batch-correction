# proteomics-batch-correction

A tool for correcting batch effects in a protein-by-sample abundance matrix
when some subjects were re-run across batches as bridge replicates.

It runs several correction methods behind one interface, scores each one
against the bridge replicates with a fixed set of QC metrics, and either
auto-selects the winner or runs a fixed default in production.

The tool produces two matricies:

- **Matrix A**, uncorrected (only median-normalized). 
- **Matrix B**, corrected. 
Matrix B keeps the same shape, proteins, and samples as Matrix A.

The downstream differential-expression and enrichment analysis that takes Matrix A
is in [proteomics-pipeline](https://github.com/justin-youngyun/proteomics-pipeline).

## Five methods

All of them run per stratum (for example per tissue or region) and share the
signature `correct(X, meta, ref_batch) -> X_corrected`.

| name | what it does |
| --- | --- |
| `bridge_median` | one global additive offset per batch, estimated from the bridge replicates. The baseline. |
| `limma_rbe` | per-protein regression that removes the batch factor while protecting the biological design (this is limma's `removeBatchEffect`). The default. |
| `combat_param` | parametric empirical-Bayes ComBat, location and scale (Johnson, Li & Rabinovic 2007). |
| `combat_bridge` | ComBat mean-only: it corrects the location shift and leaves the scale. |
| `median_polish` | per-batch effects from a Tukey median polish of the per-protein per-batch medians. |

`bridge_median` and `median_polish` apply a single number per batch, so they
cannot touch protein-specific batch effects. `limma_rbe` and the two ComBat
variants work per protein, and on realistic data they win. That contrast is the
point of the comparison.

## The six metrics

- `bridge_mad` (lower is better): median over bridge replicate pairs of the
  mean absolute difference across shared proteins.
- `pca_batch_r2` (lower is better): PERMANOVA R² of batch on complete-case
  Euclidean distances.
- `bio_preservation` (higher is better): correlation of per-protein group effect
  sizes between the corrected data and the batch-adjusted uncorrected data.
- `group_separation` (higher is better): mean silhouette width of the group
  labels in complete-case space.
- `missingness_preserved` (must stay true): the corrected NaN pattern equals the
  uncorrected one. A method that breaks it is disqualified.
- `bridge_coverage` (diagnostic): fraction of proteins detected in every bridge
  replicate, so you know how much the bridge metrics rest on.

## Auto-selection

In comparison mode it ranks the methods by an explicit rule:

1. disqualify any method that breaks missingness preservation on any stratum;
2. among the rest, minimize average `bridge_mad`;
3. break ties on lower `pca_batch_r2`;
4. break ties on higher `bio_preservation`.

Production mode skips all of that and runs the fixed default, `limma_rbe`.

## Input schema

`abundance.csv`: a leading `Protein` column, then one column per sample. Values
are log2 abundances. Blank means not detected; a literal `0` is read as not
detected too.

`samples.csv`: `Sample`, `Batch` (integer), `Group` (biological group),
`Subject` (bridge subjects share one id across batches), and `Stratum`. An
optional `Sex` column is added to the protected design if present.

Bridge subjects and the reference batch are computed, not configured. Within a
stratum, a subject appearing in more than one batch is a bridge; the reference
batch defaults to the one holding the most bridge subjects.

## Running it

```bash
pip install -r requirements.txt

# planted synthetic data, full comparison, figures
python demo.py

# your own data
python run.py --abundance data/abundance.csv --samples data/samples.csv --outdir outputs
python run.py --abundance data/abundance.csv --samples data/samples.csv --method limma_rbe
```

`demo.py` writes Matrix A, Matrix B, the metric table, and two QC figures (PCA
by batch before and after, and a bridge replicate scatter before and after) into
`outputs/`.

## Synthetic data

`synthetic.py` builds a dataset with a known planted truth so the demo can check
the methods rather than just run them: a per-protein additive batch effect with
a per-protein per-batch scale wobble, a biological group effect on a slice of
proteins, bridge subjects with identical biology across batches, and
left-censored plus random missingness. It takes a seed, so runs are
reproducible.

## Files

- `batch_correct.py` — the five methods and the registry
- `qc_metrics.py` — the six metrics
- `harness.py` — comparison/production modes, the decision rule, Matrix A and B
- `common.py` — IO, median normalization, bridge detection, design helpers
- `synthetic.py` — the planted-truth data generator
- `demo.py` — the end-to-end demo
- `run.py` — CLI for your own CSVs

## Notes

ComBat, `removeBatchEffect`, the median polish, and the PERMANOVA R² are
implemented here in numpy rather than pulled from R or a specialized package,
so the whole thing runs on numpy, pandas, scikit-learn, and matplotlib. ComBat
handles missing values by imputing each protein's gaps with its row minimum,
correcting, then restoring the original gaps; zero-variance proteins pass
through untouched. When a group sits entirely in one batch the ComBat step warns
that empirical-Bayes shrinkage can absorb the confounded biology.
