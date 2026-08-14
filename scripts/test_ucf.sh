#!/usr/bin/env bash
# Evaluate a UCF-Crime checkpoint (dsanet_v3 lineage).
# Scoring recipe: --gaussian-sigma 2, debiased causal graph (no snippet gating / category refine).
# Usage:  bash scripts/test_ucf.sh [model_path]
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
MODEL_PATH=${1:-model/ckpt_ucf_best_map.pth}
mkdir -p logs

python scripts/profile_gpu.py python src/ucf_test.py \
  --gaussian-sigma 2 \
  --model-path "$MODEL_PATH" 2>&1 | tee logs/ucf_test.log
