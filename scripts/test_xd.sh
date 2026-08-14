#!/usr/bin/env bash
# Evaluate the XD-Violence student checkpoint on the audio-available test subset.
# Usage:  bash scripts/test_xd.sh [model_path]
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
MODEL_PATH=${1:-model/model_xd_ts_kdfix61_bestMAP.pth}
mkdir -p logs

python scripts/profile_gpu.py python src/xd_test.py \
  --use-dcsa --use-dnp --use-audio-aux --use-debiased-causal-graph \
  --temp 1.5 --logits3-alpha 0.7 \
  --model-path "$MODEL_PATH" 2>&1 | tee logs/xd_test.log
