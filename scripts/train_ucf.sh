#!/usr/bin/env bash
# Train the UCF-Crime teacher-student model (dsanet_v3 lineage).
#   teacher: 1 epoch -> student: 15 epochs (debiased causal graph, gaussian-sigma 2).
# Usage:  bash scripts/train_ucf.sh
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
mkdir -p logs model

python scripts/profile_gpu.py python src/ucf_train_teacher_student.py \
  --train-teacher \
  --teacher-epochs 1 --teacher-lr 2e-5 \
  --max-epoch 15 --lr 2e-5 \
  --loss1-weight 0.8 --loss2-weight 1.1 \
  --seed 234 \
  --use-debiased-causal-graph --debiased-graph-threshold 0.2 \
  --gaussian-sigma 2 \
  --pretrained-path model/model_ucf.pth \
  --model-path model/ucf_dsanet_v3.pth \
  --checkpoint-path model/ckpt_ucf.pth \
  --teacher-model-path model/ucf_dsanet_v3_teacher.pth \
  --log-path model/ucf_train.log 2>&1 | tee logs/ucf_train_profile.log
