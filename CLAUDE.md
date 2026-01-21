# CLAUDE.md

## Project Summary
**Privacy-preserving institutional benchmarking via consensus clustering.**

This pipeline constructs peer groups from institutional snapshot data and evaluates benchmarking utility through stability metrics, percentile rankings, and semi-synthetic perturbation testing. Clustering is a means to peer construction—not the endpoint.

---

## Research Context

### Problem Statement
Traditional institutional benchmarking uses ad-hoc peer groupings (e.g., size tiers, geography). This project investigates data-driven peer construction via consensus clustering, measuring whether statistically stable peer groups produce more sensitive and reliable benchmarks than global or rule-based alternatives.

### Key Research Questions
1. Does consensus clustering produce stable peer assignments under data perturbation?
2. Do within-peer benchmarks detect KPI deviations more reliably than global benchmarks?
3. How does the number of clusters (K) affect stability–utility trade-offs?
4. How do mixed-type representations (numeric + categorical) impact peer quality?

---

## Methodological Requirements

### Reproducibility Standards
| Requirement | Implementation |
|-------------|----------------|
| Deterministic runs | Single global seed via `np.random.default_rng(seed)` |
| Version pinning | `requirements.txt` with exact versions |
| Config-driven | All hyperparameters in `configs/*.yml` |
| Output hashing | SHA256 of outputs logged in `summary.json` |

### Statistical Rigor
- **Bootstrap confidence intervals** for all stability metrics (ARI, cluster confidence)
- **Effect sizes** (Cohen's d) for benchmark sensitivity comparisons
- **Multiple comparison correction** (Bonferroni or FDR) when testing across K values
- **Monte Carlo repetitions** ≥100 for perturbation evaluation

### Data Handling
- No imputation without explicit missing indicator columns
- Scaling applied per-representation (StandardScaler for numeric, ordinal encoding for categorical)
- Train/test leakage impossible (snapshot-only, no temporal component)

---

## Architecture

```
src/
├── run_experiments.py   # CLI orchestration, K sweep, output aggregation
├── data.py              # CSV loaders, schema validation, type inference
├── preprocess.py        # Imputation, scaling, encoding, missing indicators
├── consensus.py         # Bootstrap perturbations, co-assignment matrix, consensus
├── clustering.py        # KMeans (numeric), KModes (categorical), mixed handlers
├── benchmark.py         # Within-peer percentiles, z-scores, outlier flags
├── perturbation.py      # Semi-synthetic KPI shifts, Monte Carlo detection rates
├── eval.py              # ARI, silhouette, rank-shift, sensitivity metrics
├── plots.py             # Matplotlib figures (no seaborn)
└── utils.py             # Config parsing, logging, seed management, I/O

configs/
└── default.yml          # All hyperparameters and paths

tests/
├── test_consensus.py    # Co-assignment symmetry, ARI bounds
├── test_benchmark.py    # Percentile range [0,100], z-score sanity
├── test_perturbation.py # Detection rate monotonicity with effect size
└── fixtures/            # Toy datasets for CI
```

---

## Experimental Protocol

### Phase 1: Data Preparation
1. Load snapshot CSV with schema validation
2. Generate four representations:
   - `numeric`: continuous features only (KMeans)
   - `categorical`: discrete features only (KModes)
   - `mixed_encoded`: one-hot or ordinal encoding → KMeans
   - `mixed_separated`: Gower-like handling or ensemble

### Phase 2: Consensus Clustering (per K ∈ {2, 3, 4, 5, 6})
1. Generate B=100 perturbed datasets (bootstrap + jitter + feature subsets)
2. Cluster each perturbation
3. Build co-assignment matrix C[i,j] = proportion of runs where i,j share a cluster
4. Apply hierarchical clustering to C to derive consensus assignments

### Phase 3: Stability Evaluation
- **ARI**: Adjusted Rand Index between each perturbation and consensus
- **Cluster confidence**: Mean co-assignment probability within consensus clusters
- **Silhouette**: Separation quality on original features

### Phase 4: Benchmarking Evaluation
For each KPI target:
1. Compute **within-peer percentile** for each institution
2. Compute **within-peer z-score**
3. Compare against baselines:
   - Global percentile (all institutions)
   - kNN percentile (k=5,10,20 nearest neighbors)
   - Rule-based percentile (if descriptor columns exist)

### Phase 5: Perturbation Utility Testing
1. Inject semi-synthetic KPI shifts (+0.5σ, +1σ, +2σ) into random institutions
2. Re-compute benchmarks
3. Measure **detection rate**: fraction of perturbed institutions flagged as outliers
4. Compare sensitivity across peer-group methods

### Phase 6: K Selection
Select optimal K by:
1. Stability plateau (ARI ≥ 0.8 threshold)
2. Minimum cluster size constraint (≥5% of N)
3. Utility tie-breaker (highest detection rate at 1σ shift)

---

## Output Artifacts

### Tables (`reports/tables/`)
| File | Description |
|------|-------------|
| `peer_assignments.csv` | Institution → cluster mapping for each K |
| `coassignment_matrix.csv` | N×N co-assignment probabilities |
| `stability_summary.csv` | ARI, confidence, silhouette per K |
| `benchmark_results.csv` | Percentiles/z-scores for all KPIs and methods |
| `perturbation_eval.csv` | Detection rates by method, shift size, K |
| `k_sweep_summary.csv` | Aggregated metrics for K selection |

### Figures (`reports/figures/`)

All figures are publication-quality (300 DPI, serif fonts, colorblind-friendly palette).

| File | Purpose | Key Interpretation |
|------|---------|-------------------|
| `fig_coassignment_heatmap.png` | Peer cohesion structure | Block-diagonal structure = stable peer groups; yellow = high co-assignment probability |
| `fig_stability_vs_k.png` | K selection via ARI plateau | Select K where ARI stabilizes ≥0.8; dual axis shows confidence |
| `fig_benchmark_<KPI>.png` | Method comparison per institution | Bar heights show percentile; divergence = peer context matters |
| `fig_detection_<KPI>.png` | Sensitivity analysis | Higher recall at lower σ = better sensitivity; error bars show Monte Carlo variance |
| `fig_method_comparison.png` | Aggregate method summary | Median + IQR across all institutions and KPIs |
| `fig_cluster_profiles.png` | Peer group interpretation | Parallel coordinates of cluster centroids (normalized) |

#### How to Read Each Figure

**Co-assignment Heatmap**:
- Institutions are reordered by cluster assignment
- Bright yellow squares indicate institution pairs that always cluster together (stable peers)
- Darker regions between blocks indicate clear cluster boundaries
- Scattered yellow outside blocks suggests assignment instability

**Stability vs K**:
- X-axis: number of clusters tested
- Left Y-axis (blue): Mean Adjusted Rand Index (stability)
- Right Y-axis (magenta): Mean cluster confidence
- Dashed line: stability threshold (ARI = 0.8)
- Error bars: standard deviation across bootstrap runs
- Look for: plateau where both metrics stabilize

**Benchmark Percentiles**:
- Each bar group = one institution
- Colors = different benchmarking methods (peer, global, kNN, rule-based)
- Reference lines at 25th, 50th, 75th percentiles
- Key insight: where bars diverge, peer context changes the ranking

**Detection Curves**:
- X-axis: perturbation magnitude (in standard deviations)
- Blue line: Recall (sensitivity) - ability to detect true anomalies
- Red line: False positive rate - incorrectly flagged normal institutions
- Shaded regions: undesirable areas (low recall, high FPR)
- Ideal: high recall, low FPR at small magnitudes

### Summary (`reports/summary.json`)
```json
{
  "selected_k": 4,
  "selection_reason": "stability_plateau",
  "mean_ari": 0.847,
  "mean_confidence": 0.912,
  "detection_rate_1sigma": 0.73,
  "output_hash": "sha256:...",
  "config_hash": "sha256:...",
  "seed": 42
}
```

---

## Testing Requirements

### Unit Tests (pytest)
```bash
pytest tests/ -v --tb=short
```

**Invariants to verify:**
- Co-assignment matrix is symmetric and in [0,1]
- ARI ∈ [-1, 1], with ARI=1 for identical clusterings
- Percentiles ∈ [0, 100]
- Z-scores have mean≈0, std≈1 within each peer group
- Detection rate increases monotonically with shift magnitude
- Identical seed → identical outputs

### Integration Tests
```bash
python -m src.run_experiments --config configs/default.yml --dry-run
python -m src.run_experiments --config configs/default.yml
diff reports/summary.json reports/summary.json.bak  # reproducibility check
```

### Statistical Validation
- Bootstrap CI coverage: verify 95% CIs contain true value ≥93% of the time
- Permutation tests: null distribution of ARI under random relabeling
- Sensitivity analysis: vary B (bootstrap count) to assess convergence

---

## Coding Standards

### Type Safety
- All functions have type hints
- `mypy --strict src/` passes without errors

### Error Handling
- Explicit `ValueError` for schema mismatches
- Explicit `KeyError` messages naming missing columns
- No silent NaN propagation

### Dependencies
Core only: `numpy`, `scipy`, `pandas`, `scikit-learn`, `kmodes`, `matplotlib`, `pyyaml`

---

## Quality Gates

Before any commit:
1. `pytest tests/` passes
2. `mypy --strict src/` passes
3. CLI runs end-to-end on toy fixture
4. Re-run with same seed produces byte-identical outputs
5. No hardcoded paths or magic numbers outside config

---

## Research Paper Checklist

When writing results:
- [ ] Report exact seed used
- [ ] Report bootstrap count B and Monte Carlo repetitions
- [ ] Include 95% CIs for all point estimates
- [ ] State null hypothesis and correction method for multi-K comparisons
- [ ] Cite consensus clustering methodology (Monti et al., 2003)
- [ ] Cite stability metrics (ARI: Hubert & Arabie, 1985)
- [ ] Provide supplementary materials with full config YAML
- [ ] Archive code version (git SHA) in paper appendix
