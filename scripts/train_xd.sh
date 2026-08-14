#!/usr/bin/env bash
# Train the XD-Violence teacher-student model from the pretrained checkpoint.
#   teacher: 3 epochs -> student: 12 epochs (joint distillation, audio+DCSA+DNP).
# Usage:  bash scripts/train_xd.sh
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
mkdir -p logs model

python scripts/profile_gpu.py python src/xd_train_teacher_student.py \
  --train-stage joint \
  --use-dcsa --use-dnp --use-audio-aux --use-debiased-causal-graph \
  --train-teacher --teacher-epochs 3 --teacher-scheduler-milestones 2 4 \
  --max-epoch 12 --scheduler-milestones 5 8 11 \
  --kd-temp 4.0 \
  --distill-kd-bin-weight 0.4 \
  --distill-kd-multi-weight 0.30 \
  --distill-kd-feat-weight 0.30 \
  --dcsa-loss-weight 0.20 \
  --dnp-loss-weight 0.10 \
  --loss1-weight 1.0 \
  --temporal-consistency-weight 0.10 \
  --temp 1.5 \
  --logits3-alpha 0.7 \
  --teacher-model-path model/model_xd_teacher_v61.pth \
  --model-path model/model_xd_ts_kdfix61.pth \
  --checkpoint-path model/ckpt_xd_ts_kdfix61.pth \
  --log-path model/xd_ts_kdfix61.log \
  --pretrained-path model/model_xd.pth
