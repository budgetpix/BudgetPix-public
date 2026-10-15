#!/usr/bin/env bash
# Fine-tune one JiT line (global batch 896 on 4 GPUs).
#   LINE=jit_b16_256 RELEASED=/weights/jit-b-16/checkpoint-last.pth DATA=/data/imagenet-1k/data bash train.sh
set -eo pipefail
LINE=${LINE:?jit_b16_256 | jit_b32_512 | jit_l16_256 | jit_l32_512}
case "$LINE" in
  jit_b16_256|jit_b32_512) EPOCHS=100; BS=112; ACCUM=2 ;;
  jit_l16_256)             EPOCHS=50;  BS=112; ACCUM=2 ;;
  jit_l32_512)             EPOCHS=50;  BS=56;  ACCUM=4 ;;
  *) echo "unknown LINE=$LINE" >&2; exit 1 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
torchrun --nproc_per_node=${NGPU:-4} "$HERE/train.py" --released "${RELEASED:?released JiT checkpoint of this line}" \
  --data "${DATA:?ImageNet-1k parquet directory}" --epochs $EPOCHS --batch_size $BS --grad_accum $ACCUM \
  --out "${OUT:-$HERE/runs/$LINE}" --resume
