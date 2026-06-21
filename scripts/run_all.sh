#!/usr/bin/env bash
# Full end-to-end pipeline.  Run from the project root:
#   bash scripts/run_all.sh
#
# Steps
# -----
#  1. Generate 80/20 train / test split
#  2. Extract spectral index features (32-d)
#  3. Fine-tune RGB CNN backbone
#  4. Fine-tune MS CNN backbone (10 land bands)
#  5. Extract fine-tuned CNN features (512-d per modality)
#  6. Train LightGBM arms (indices_only / rgb_cnn / ms_cnn / ms_cnn_indices)
#  7. Run late-fusion pipeline
#  8. Evaluate CNN models and generate all visualizations

set -euo pipefail

PYTHON="${PYTHON:-python3}"    # override with: PYTHON=/path/to/python bash scripts/run_all.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "================================================"
echo " EuroSAT MS Project — full pipeline"
echo " Python: $PYTHON"
echo " Root:   $ROOT"
echo "================================================"

echo ""
echo "--- Step 1: Generate train/test splits ---"
$PYTHON data/splits.py

echo ""
echo "--- Step 2: Extract spectral index features ---"
$PYTHON features/spectral_indices.py

echo ""
echo "--- Step 3: Fine-tune RGB backbone ---"
$PYTHON train.py --modality rgb

echo ""
echo "--- Step 4: Fine-tune MS backbone (10 land bands) ---"
$PYTHON train.py --modality ms

echo ""
echo "--- Step 5: Extract fine-tuned CNN features ---"
$PYTHON features/cnn_extractor.py --modality rgb --finetuned
$PYTHON features/cnn_extractor.py --modality ms  --finetuned

echo ""
echo "--- Step 6: Train LightGBM arms ---"
$PYTHON train_ml.py

echo ""
echo "--- Step 7: Late-fusion pipeline ---"
$PYTHON fusion_ft/run_fusion.py

echo ""
echo "--- Step 8: Evaluate CNN models (test set + visualizations) ---"
$PYTHON test.py --modality rgb --checkpoint results/rgb/best.pt --visualize
$PYTHON test.py --modality ms  --checkpoint results/ms/best.pt  --visualize

echo ""
echo "================================================"
echo " Done!  Results:"
echo "  results/rgb/     — RGB CNN metrics + confusion matrix + gallery"
echo "  results/ms/      — MS  CNN metrics + confusion matrix + gallery"
echo "  results/ml/      — LightGBM arms + SHAP"
echo "  results_fusion_ft/ — Late-fusion results"
echo "================================================"
