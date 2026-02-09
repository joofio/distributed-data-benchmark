#!/bin/bash
# Run All Experiments Script
# Regenerates all experimental data for the consensus clustering paper

set -e  # Exit on first error

# Navigate to project root
#cd "$(dirname "$0")/../.."
#PROJECT_ROOT=$(pwd)
#echo "Project root: $PROJECT_ROOT"

# Activate conda environment if needed (uncomment if required)
#conda activate py3

echo "=============================================="
echo "PHASE 1: Main experiments (7 datasets)"
echo "=============================================="

DATASETS=("obscare" "heart_disease" "breast_cancer" "pima_diabetes" "hcv" "liver_disorders" "early_diabetes")

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Running: $dataset"
    echo "----------------------------------------------"
    python scripts/experiments/run_experiments.py --config "configs/${dataset}.yml"
done

echo ""
echo "=============================================="
echo "PHASE 2: Baseline comparison (Gower, K-prototypes)"
echo "=============================================="
python scripts/experiments/baseline_comparison.py

echo ""
echo "=============================================="
echo "PHASE 3: Ground-truth validation (synthetic data)"
echo "=============================================="
python scripts/experiments/ground_truth_validation.py

echo ""
echo "=============================================="
echo "PHASE 4: Sensitivity analysis"
echo "=============================================="
python scripts/experiments/sensitivity_analysis.py

echo ""
echo "=============================================="
echo "PHASE 5: Privacy experiments (ε ∈ {2, 5, ∞})"
echo "=============================================="
python scripts/experiments/run_privacy_experiments.py

echo ""
echo "=============================================="
echo "PHASE 6: Multi-KPI analysis"
echo "=============================================="
python scripts/experiments/multi_kpi_analysis.py --config configs/obscare.yml

echo ""
echo "=============================================="
echo "PHASE 7: Re-identification risk analysis"
echo "=============================================="
python scripts/experiments/reidentification_analysis.py --config configs/obscare.yml

echo ""
echo "=============================================="
echo "PHASE 8: Liability analysis (multi-seed, baseline)"
echo "=============================================="
python scripts/experiments/liability_analysis.py

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/{dataset}/summary.json     (per-dataset summaries)"
echo "  - results/tables/                    (CSV tables)"
echo "  - results/sensitivity/               (sensitivity figures)"
echo "  - results/privacy_summary.csv        (privacy analysis)"
echo ""
