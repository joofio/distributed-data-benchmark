# Experimental Results Analysis

> **Template for analyzing consensus clustering benchmarking results**
>
> Fill in this template after running experiments on each dataset.
> This structured analysis supports thesis writing and paper preparation.

---

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `[DATASET_NAME]` |
| N institutions | `[N]` |
| N features (numeric) | `[N_NUMERIC]` |
| N features (categorical) | `[N_CATEGORICAL]` |
| Target KPI | `[TARGET_RATE]` |
| Bootstrap iterations (B) | `[N_BOOTSTRAPS]` |
| Monte Carlo runs | `[N_RUNS]` |
| K values tested | `[2, 3, 4, 5, 6]` |
| Random seed | `42` |
| Run date | `[DATE]` |

---

## 2. Key Findings Summary

### 2.1 Selected K Value

**Optimal K = `[K]`**

Selection rationale:
- [ ] Stability plateau reached (ARI ≥ 0.8)
- [ ] Minimum cluster size constraint satisfied (≥ 5% of N)
- [ ] Utility tie-breaker (detection rate)

### 2.2 Main Results (One-Sentence Summaries)

1. **Stability**: _[e.g., "Consensus clustering achieved mean ARI = 0.85 ± 0.04, indicating stable peer assignments across perturbations."]_

2. **Benchmark Sensitivity**: _[e.g., "Within-peer benchmarking detected 73% of 1σ deviations vs. 45% for global benchmarking."]_

3. **Method Comparison**: _[e.g., "Peer-based ranking diverged from global ranking by >20 percentile points for 12% of institutions."]_

---

## 3. Detailed Analysis

### 3.1 Clustering Stability (Figure: `fig_stability_vs_k.png`)

| K | Mean ARI | Std ARI | Mean Confidence | Min Cluster Size |
|---|----------|---------|-----------------|------------------|
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |

**Interpretation**:
- At what K does stability plateau? _[Answer]_
- Is there a stability-granularity trade-off? _[Answer]_
- How does confidence correlate with ARI? _[Answer]_

**For thesis**: This figure demonstrates that consensus clustering produces stable peer assignments, with ARI stabilizing at K = _[X]_. The plateau suggests that peer structure is robust to the exact choice of K beyond this threshold.

---

### 3.2 Co-assignment Structure (Figure: `fig_coassignment_heatmap.png`)

**Visual observations**:
- [ ] Clear block-diagonal structure visible
- [ ] Some institutions have ambiguous assignments (mid-range co-assignment values)
- [ ] Certain institution pairs always cluster together (C[i,j] ≈ 1.0)
- [ ] Outlier institutions that don't consistently cluster with any group

**Interpretation**:
- What does the heatmap reveal about peer group structure? _[Answer]_
- Are there natural boundaries between peer groups? _[Answer]_
- Which institutions have unstable assignments? _[Answer]_

**For thesis**: The co-assignment matrix reveals _[N]_ distinct peer clusters, with _[X]_% of institution pairs having co-assignment probability > 0.8, indicating strong peer cohesion.

---

### 3.3 Benchmarking Method Comparison (Figure: `fig_benchmark_*.png`)

**Percentile rank differences**:

| Method | Median Percentile | IQR | Correlation with Peer |
|--------|-------------------|-----|----------------------|
| Peer | 50.0 (by definition) | | 1.00 |
| Global | | | |
| kNN | | | |
| Rule-based | | | |

**Key observations**:
- How many institutions change rank by >10 percentile points between methods? _[Answer]_
- Which method shows highest variance in rankings? _[Answer]_
- Do any institutions appear as outliers in one method but not another? _[Answer]_

**For thesis**: Peer-based benchmarking produced rankings that diverged from global benchmarking by more than 15 percentile points for _[X]_% of institutions, demonstrating that peer context meaningfully affects performance assessment.

---

### 3.4 Detection Sensitivity (Figure: `fig_detection_*.png`)

**Perturbation detection rates**:

| Magnitude (σ) | Recall (Peer) | Recall (Global) | FPR (Peer) | FPR (Global) |
|---------------|---------------|-----------------|------------|--------------|
| 0.1 | | | | |
| 0.2 | | | | |
| 0.3 | | | | |

**Key observations**:
- At what magnitude does peer-based detection exceed 80%? _[Answer]_
- How does false positive rate scale with magnitude? _[Answer]_
- Is peer-based detection more sensitive than global at all magnitudes? _[Answer]_

**For thesis**: Peer-based benchmarking detected _[X]_% of 1σ deviations compared to _[Y]_% for global benchmarking, representing a _[Z]_% improvement in sensitivity to performance anomalies.

---

## 4. Research Question Answers

### RQ1: Does consensus clustering produce stable peer assignments?

**Answer**: _[Yes/No/Partially]_

**Evidence**:
- Mean ARI = _[X]_ (threshold: 0.8)
- Stability across K values: _[Describe pattern]_
- Bootstrap variance: _[Low/Moderate/High]_

**Confidence**: _[High/Medium/Low]_

---

### RQ2: Do within-peer benchmarks detect KPI deviations more reliably?

**Answer**: _[Yes/No/Partially]_

**Evidence**:
- Detection rate improvement: _[X]_% vs global
- Effect size (Cohen's d): _[d]_
- Statistical significance: _[p-value if computed]_

**Confidence**: _[High/Medium/Low]_

---

### RQ3: How does K affect stability-utility trade-offs?

**Answer**: _[Describe relationship]_

**Evidence**:
- Stability peaks at K = _[X]_
- Detection rate peaks at K = _[Y]_
- Optimal balance at K = _[Z]_

**Confidence**: _[High/Medium/Low]_

---

### RQ4: How do mixed-type representations impact peer quality?

**Answer**: _[Answer based on representation comparison if run]_

**Evidence**:
- Numeric-only ARI: _[X]_
- Mixed-encoded ARI: _[Y]_
- Difference in peer assignments: _[Describe]_

**Confidence**: _[High/Medium/Low]_

---

## 5. Limitations and Caveats

### 5.1 Data Limitations
- [ ] Synthetic institution construction (patients → institutions)
- [ ] Limited N (N = 30 institutions)
- [ ] Missing values in original UCI data
- [ ] Other: _[Specify]_

### 5.2 Methodological Limitations
- [ ] Single dataset (generalization unclear)
- [ ] Bootstrap count B may be insufficient for CI coverage
- [ ] Perturbation model assumes additive shifts
- [ ] Other: _[Specify]_

### 5.3 Threats to Validity
- **Internal**: _[List any confounds]_
- **External**: _[List generalization concerns]_
- **Construct**: _[List measurement concerns]_

---

## 6. Figures for Paper

### Recommended figures to include:

| Figure | Filename | Purpose | Include? |
|--------|----------|---------|----------|
| Stability curve | `fig_stability_vs_k.png` | K selection justification | [ ] |
| Co-assignment heatmap | `fig_coassignment_heatmap.png` | Peer structure visualization | [ ] |
| Benchmark comparison | `fig_benchmark_target_rate.png` | Method comparison | [ ] |
| Detection curve | `fig_detection_target_rate.png` | Sensitivity analysis | [ ] |
| Method summary | `fig_method_comparison.png` | Aggregate comparison | [ ] |

### Figure captions (draft):

**Figure X**: _[Draft caption for stability curve]_

**Figure Y**: _[Draft caption for co-assignment heatmap]_

**Figure Z**: _[Draft caption for benchmark comparison]_

---

## 7. Tables for Paper

### Table 1: Dataset Characteristics

| Property | Value |
|----------|-------|
| Source | UCI ML Repository |
| Original purpose | _[Disease classification]_ |
| N patients (original) | _[N]_ |
| N institutions (synthetic) | 30 |
| Patients per institution | _[mean ± std]_ |
| Feature types | Numeric: _[N]_, Categorical: _[N]_ |
| Target variable | Disease prevalence rate |
| Target range | _[min - max]_ |

### Table 2: Stability Metrics by K

_[Copy from Section 3.1]_

### Table 3: Benchmark Method Comparison

_[Copy from Section 3.3]_

### Table 4: Detection Sensitivity

_[Copy from Section 3.4]_

---

## 8. Statistical Tests (if applicable)

### Test 1: ARI significance
- Null hypothesis: Consensus ARI = random clustering ARI
- Test: Permutation test
- p-value: _[p]_
- Conclusion: _[Reject/Fail to reject]_

### Test 2: Detection rate comparison
- Null hypothesis: Peer detection rate = Global detection rate
- Test: McNemar's test / paired t-test
- p-value: _[p]_
- Effect size: _[d]_
- Conclusion: _[Reject/Fail to reject]_

---

## 9. Next Steps

- [ ] Run on additional datasets for generalization
- [ ] Increase bootstrap iterations for tighter CIs
- [ ] Compare with alternative clustering methods
- [ ] Investigate outlier institutions
- [ ] _[Other]_

---

## 10. Raw Output References

| Output | Path |
|--------|------|
| Summary JSON | `reports/[dataset]/summary.json` |
| K sweep table | `reports/[dataset]/tables/k_sweep_summary.csv` |
| Peer assignments | `reports/[dataset]/tables/peer_assignments.csv` |
| Benchmark results | `reports/[dataset]/tables/benchmark_results.csv` |
| Perturbation eval | `reports/[dataset]/tables/perturbation_eval.csv` |
| All figures | `reports/[dataset]/figures/` |

---

*Last updated: [DATE]*
