# Distributed Data Benchmark

Privacy-preserving institutional benchmarking via consensus clustering.

## Project Structure

```
distributed-data-benchmark/
├── src/                    # Core library code
│   ├── benchmark.py        # Benchmarking functions (percentiles, z-scores)
│   ├── clustering.py       # Clustering algorithms (kmeans, PAM, hierarchical)
│   ├── consensus.py        # Consensus clustering with bootstrap
│   ├── data.py             # Data loading and validation
│   ├── plots.py            # Publication-quality figures
│   ├── privacy.py          # Privacy analysis (LDP, re-identification risk)
│   ├── synthetic.py        # Synthetic data generation
│   └── ...
│
├── scripts/
│   ├── experiments/        # Data generation & analysis scripts
│   │   ├── run_experiments.py
│   │   ├── baseline_comparison.py
│   │   ├── ground_truth_validation.py
│   │   ├── multi_kpi_analysis.py
│   │   └── reidentification_analysis.py
│   └── figures/            # Figure generation scripts
│       └── generate_figures.py
│
├── configs/                # Configuration files (YAML)
│   ├── obscare.yml         # ObsCare dataset config
│   ├── hcv.yml             # HCV dataset config
│   └── ...
│
├── data/                   # Input datasets (CSV)
│
├── results/                # Output results
│   ├── tables/             # CSV output tables
│   ├── figures/            # Generated figures
│   └── <dataset>/          # Per-dataset results
│
├── tests/                  # Unit tests
│
├── docs/                   # Documentation
│   ├── CLAUDE.md           # AI assistant context
│   └── IMPLEMENTATION_REVIEW.md
│
└── app/                    # Interactive app
    └── streamlit_app.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run baseline comparison
python scripts/experiments/baseline_comparison.py --config configs/obscare.yml

# Run full experiments
python scripts/experiments/run_experiments.py --config configs/obscare.yml
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `baseline_comparison.py` | Compare clustering methods (kmeans, PAM, k-prototypes, hierarchical) |
| `ground_truth_validation.py` | Validate clustering on synthetic data with known structure |
| `multi_kpi_analysis.py` | Analyze peer group validity across multiple KPIs |
| `reidentification_analysis.py` | Privacy risk assessment |

## Configuration

All experiments are configured via YAML files in `configs/`. Key sections:
- `features`: Define numeric and categorical columns
- `targets`: KPIs to benchmark
- `consensus`: Bootstrap and clustering parameters
- `output`: Result paths

See `docs/CLAUDE.md` for detailed methodology documentation.

## Data Provenance: Tables & Figures

This section documents the relationship between paper tables/figures and their source data.

### Tables

| Table | Label | Data Source | Generating Script |
|-------|-------|-------------|-------------------|
| 1 | `tab:datasets` | Static (materials.tex) | N/A |
| 2 | `tab:stability` | `results/{dataset}/summary.json` | `run_experiments.py` |
| 3 | `tab:detection` | `results/{dataset}/summary.json` → `mean_recall`, `mean_false_positive_rate` | `run_experiments.py` |
| 4 | `tab:peer_dist` | `results/obscare/tables/peer_assignments.csv` | `run_experiments.py` |
| 5 | `tab:benchmark_comparison` | `results/obscare/tables/benchmark_results.csv` | `run_experiments.py` |
| 6 | `tab:privacy` | `results/privacy_summary.csv` | `run_privacy_experiments.py` |
| A.1 | `tab:baseline_comparison_appendix` | `results/tables/baseline_comparison.csv` | `baseline_comparison.py` |
| A.2 | `tab:ground_truth_appendix` | `results/synthetic/ground_truth_results.csv` | `ground_truth_validation.py` |

### Figures

| Figure | Label | Image File | Data Source | Generating Script |
|--------|-------|------------|-------------|-------------------|
| 1 | `fig:tradeoff` | `figures/stability_vs_k.png` | `results/obscare/k_sweep.csv` | `scripts/figures/generate_figures.py` |
| 2 | `fig:coassignment` | `figures/coassignment_heatmap.png` | `results/obscare/coassignment_matrix.npy` | `scripts/figures/generate_figures.py` |
| 3 | `fig:benchmark_percentiles` | `figures/benchmark_percentiles_target_rate.png` | `results/obscare/tables/benchmark_results.csv` | `scripts/figures/generate_figures.py` |

### Sensitivity Analysis Figures (Not in main paper)

| File | Data Source | Script |
|------|-------------|--------|
| `jitter_sensitivity.png` | Sensitivity sweep: σ ∈ {0, 0.01, 0.05, 0.10, 0.15} | `sensitivity_analysis.py` |
| `k_threshold_sensitivity.png` | Sensitivity sweep: δ ∈ {0.03, 0.05, 0.10} | `sensitivity_analysis.py` |
| `alpha_sensitivity.png` | Sensitivity sweep: α ∈ {0.2, 0.3, 0.5, 0.7, 0.8} | `sensitivity_analysis.py` |
| `multiseed_variance.png` | Multi-seed validation (10 seeds) | `liability_analysis.py` |
| `privacy_sensitivity.png` | Privacy analysis: ε ∈ {2.0, 5.0, ∞} | `run_privacy_experiments.py` |

### Regenerating All Data

```bash
# Run all experiments (generates tables & base data)
./run_all_experiments.sh

# Regenerate figures from existing data
python scripts/figures/generate_figures.py
```

**Note**: Figures were generated from specific experiment runs. Re-running experiments may produce slightly different values due to random seed variations. Always verify figure-data consistency after regeneration.
