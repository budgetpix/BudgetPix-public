# BudgetPix on JiT (class-conditional ImageNet, 256 and 512)

The four ImageNet lines of the paper: the released JiT-B/16 and JiT-L/16 (256x256, 16-px patches) and JiT-B/32 and
JiT-L/32 (512x512, 32-px patches), each fine-tuned with the BudgetPix layout and the pixel-DiT refiner. Every line has a
dense grid of 256 tokens; a budget below 256 merges low-entropy 2x2 and 4x4 groups of base patches.

| file | content |
|---|---|
| `model.py` | the JiT transformer (vendored from [LTH14/JiT](https://github.com/LTH14/JiT)) with the BudgetPix embedding, positions and refine head; checkpoint loading |
| `layout.py` | the entropy quadtree, budget calibration and packing |
| `diffusion.py` | the training loss and the Heun sampler |
| `data.py` | the ImageNet-1k parquet stream |
| `sample.py`, `train.py` (+ `train.sh`), `eval.py` | the three entry points |

## Sample

```bash
cd jit
python sample.py --ckpt ../checkpoints/budgetpix_jit_l32_512.pth --budget 128 --classes 207,88,417,933 --overlay
python sample.py --ckpt ../checkpoints/budgetpix_jit_b16_256.pth --sweep --classes 207,88     # 100 / 75 / 50 / 25 %
cd ..
```

`--budget` is the token count per image (256 = dense, 128 = 50 %, 64 = 25 %); `--overlay` also writes the token layout.
The architecture and resolution are read from the checkpoint. The sampler runs the first steps dense (10 of 50 at 256,
5 of 50 at 512), then re-derives the layout from the model's own x0 estimate every 5 steps. Guidance 3.0 is the paper's
setting; the best value rises as the budget falls (about 6.0 at 25-50 % on the 512 lines). `--released
/weights/jit-l-32/checkpoint-last.pth` runs the released JiT at the same budget instead (the untrained control).

## Train

Data: ImageNet-1k as the Hugging Face parquet shards (`train-*.parquet`, columns `image.bytes` and `label`). Base
weights: the released JiT checkpoints from https://github.com/LTH14/JiT.

```bash
cd jit
LINE=jit_b16_256 RELEASED=/weights/jit-b-16/checkpoint-last.pth DATA=/data/imagenet-1k/data NGPU=4 bash train.sh
cd ..
```

`train.sh` holds the recipe of each line: 100 epochs for the B lines and 50 for the L lines, a global batch of 896 on
4 GPUs (the learning rate is `1e-5 x batch / 256`, 8x on the added modules). Re-running the same command resumes from
`runs/<line>/checkpoint-last.pth`. Checkpoints hold `{model, ema, optimizer, step, args}`; `sample.py` and `eval.py`
read them directly and use the EMA weights. All details are in [`docs/TRAINING_DETAILS.md`](../docs/TRAINING_DETAILS.md).

## Evaluate

FID / IS at a budget with the protocol of the paper (50 images per class, 50 Heun steps, guidance 3.0):

```bash
cd jit
torchrun --nproc_per_node=4 eval.py --ckpt ../checkpoints/budgetpix_jit_l32_512.pth --budget 128
torchrun --nproc_per_node=4 eval.py --released /weights/jit-l-32/checkpoint-last.pth --budget 128   # untrained control
cd ..
```

The paper's budgets are 256, 230, 205, 179, 154, 128 and 64 tokens (100 / 90 / 80 / 70 / 60 / 50 / 25 %). Image k
always gets the same noise, so every budget sees the same noise. The run prints the mean tokens per image, FID and IS,
computed by the LTH14 fork of `torch-fidelity` (see `requirements.txt`) against the reference statistics in
`fid_stats/`.
