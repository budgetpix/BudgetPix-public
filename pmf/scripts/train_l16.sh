#!/usr/bin/env bash
# Fine-tune the released pMF-L/16 into the BudgetPix pMF model.
#
#   cd pmf
#   PMF_ROOT=/path/to/pMF DATA=/data/imagenet-1k/data WEIGHTS=../checkpoints/pmf_released_L16_full_uv.pt NGPU=7 bash scripts/train_l16.sh
#
# Resumes from $OUT/checkpoint-last.pt when re-run.
set -eo pipefail
: "${PMF_ROOT:?set PMF_ROOT to a checkout of github.com/Lyy-iiis/pMF (branch torch)}"
: "${DATA:?set DATA to the imagenet-1k parquet directory}"
WEIGHTS=${WEIGHTS:?set WEIGHTS to the FULL (u+v) released pMF-L/16 weights, pmf_released_L16_full_uv.pt}
NGPU=${NGPU:-7}; BATCH=${BATCH:-13}; STEPS=${STEPS:-64000}; OUT=${OUT:-runs/pmf_l16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
torchrun --nproc_per_node=$NGPU --master_port ${MASTER_PORT:-29530} train_pmf_apt.py \
  --arch L16 --ckpt "$WEIGHTS" --data "$DATA" --out "$OUT" --steps $STEPS --batch $BATCH \
  --lr 1e-5 --new_lr_mult 8 --warmup 500 --clip 20 --ema 0.9995 --save_every 2000 \
  --anchor_p 0.30 --scramble_p 0.10 --offset_lo 0.0 --offset_hi 2.0 --perceptual 1 --cfg_beta 2.0 \
  --lr_decay_start $((STEPS - 8000)) --lr_decay_steps 8000 --workers 4 --log_every 20 "$@"
