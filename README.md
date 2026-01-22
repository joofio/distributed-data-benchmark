# distributed-data-benchmark

Snapshot-only, privacy-preserving benchmarking pipeline for institutions using consensus clustering and within-peer benchmarking.

## Quick start

1) Install deps:

```bash
pip install -r requirements.txt
```

2) Run the pipeline on the toy snapshot dataset:

```bash
python -m src.run_experiments --config configs/default.yml
```

Outputs are saved under `reports/tables` and `reports/figures`.

## Expected CSV format

Required columns:
- `institution_id` (string)
- feature columns (mixed types; listed in config)
- target KPI columns for benchmarking (numeric; listed in config)

Optional columns:
- descriptor columns for rule-based peers (e.g., `size_tier`)

See `tests/fixtures/toy_snapshot.csv` for a minimal example.

## Config overview

`configs/default.yml` includes:
- dataset path and optional column rename mapping
- feature columns (numeric + categorical)
- KPI targets and optional descriptor columns
- preprocessing options (ordinal vs one-hot encoding)
- consensus clustering settings (bootstraps, jitter, feature bootstrap)
- benchmarking options (outlier thresholds, kNN sizes)
- perturbation evaluation settings
- K selection settings and output paths

Representation options:
- `numeric`
- `categorical` (uses k-modes unless one-hot encoding is selected)
- `mixed_encoded`
- `mixed_separated`

## Outputs

Tables in `reports/tables`:
- `peer_assignments.csv`
- `coassignment_matrix.csv`
- `stability_summary.csv`
- `benchmark_results.csv`
- `perturbation_eval.csv`
- `k_sweep_summary.csv`

Figures in `reports/figures`:
- `coassignment_heatmap.png`
- `stability_vs_k.png`
- `benchmark_percentiles_<KPI>.png`
- `perturbation_detection_curve_<KPI>.png`

Summary JSON:
- `reports/summary.json`

## Synthetic institution generation

Use `create_datasets.ipynb` to turn patient-level UCI data into synthetic institutions.
The notebook exposes a `SPLIT_STRATEGY` option that controls how patients are assigned
to institutions, which directly affects heterogeneity across institutions:

- `random`: shuffle then split evenly (low structural heterogeneity).
- `stratified`: balance institutions by a key column (defaults to `target_col`), which
  smooths KPI prevalence differences.
- `clustered`: sort by a continuous column and slice, which amplifies institutional
  differences along that variable.

## Adding local UCI datasets

Add a local CSV path in the config:

```yaml
dataset:
  type: uci
  path: /path/to/local.csv
  mapping:
    original_id_col: institution_id
    original_numeric_col: feature_numeric_1
```

Update the `features` and `targets` sections so the loader can validate the schema.
