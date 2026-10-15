# BudgetPix on pixel MeanFlow (pMF-L/16, one-step ImageNet 256)

The released pMF-L/16 (pixel MeanFlow: one-step, class-conditional, pixel-space ImageNet 256; a JiT descendant with
16-px patches and 256 tokens on the dense grid) fine-tuned with the BudgetPix layout and the token-local refiner, with
pMF's own MeanFlow objective. One forward pass makes an image; the token budget sets how much of that pass is spent.
The released model code is imported at run time from the official PyTorch branch:

```bash
git clone -b torch https://github.com/Lyy-iiis/pMF upstream/pMF     # from the repository root
export PMF_ROOT=$PWD/upstream/pMF
```

That checkout has a plain `models` package, so no other `models` package may be on `PYTHONPATH` at the same time.

## Sample (1 NFE)

A one-step sampler has no trajectory to measure entropy on, so the layout is a condition chosen up front:

```bash
cd pmf
python ../scripts/download_checkpoints.py pmf_l16_256
python sample_pmf_apt.py --ckpt ../checkpoints/budgetpix_pmf_l16_256.pt --classes 207,88,417,933 --budget 128 --overlay --out samples.png
cd ..
```

`--budget` is the token count (256 = dense, 128 = 50 %, 64 = 25 %); `--overlay` draws the layout. By default
(`--layout self`) the layout is the entropy quadtree of the model's own dense one-step sample for the same class and
noise, bisected to the budget, which costs one extra dense forward. `--layout image --layout_from photo.jpg` takes the
entropy layout of any image instead, and `--layout random` is a content-free random map (clearly worse: the layout
should follow the content). `--nfe 2` adds a second step whose layout comes from the first step's own clean-image
estimate. Guidance defaults are the released L/16 settings (omega 7.0 on the CFG interval [0.2, 0.7]).

The released model driven at a reduced budget without any training (the no-training control; it collapses below 80 %):

```bash
cd pmf
python sample_pmf_apt.py --ckpt released --released /weights/pMF-L-16.pt --budget 128 --classes 207
cd ..
```

## Train

The fine-tuning uses pMF's full objective (MeanFlow with the compound target, in-target guidance, the auxiliary v-loss,
LPIPS and ConvNeXt perceptual terms), so it needs the released weights WITH the auxiliary v-branch. The Hub's inference
files (`pMF-L-16.pt`) carry the u-path only. The converted B/16 and L/16 training states are in the Hugging Face repo
next to the checkpoints (`pmf_released_B16_full_uv.pt`, `pmf_released_L16_full_uv.pt`).

```bash
cd pmf
DATA=/data/imagenet-1k/data WEIGHTS=../checkpoints/pmf_released_L16_full_uv.pt NGPU=7 bash scripts/train_l16.sh
cd ..
```

The recipe: 13 images per process (91 per step with 7 processes), lr 1e-5 (x8 on the added modules) constant after a
500-step warm-up, EMA 0.9995, grad clip 20, anchor 0.30, scramble 0.10, offset window U(0, 2), 64000 steps with a
linear lr cooldown over the last 8000. The EMA weights at step 64000 are the released checkpoint. Data is ImageNet-1k
as HuggingFace parquet shards, centre-cropped to 256 without flips (pMF's preprocessing). See
[`docs/TRAINING_DETAILS.md`](../docs/TRAINING_DETAILS.md).

## Evaluate

FID / IS at a budget with pMF's protocol (50 per class, noise seeded by the image index, pMF's torch-fidelity wrapper
against the ImageNet-256 reference statistics in `jit/fid_stats/`):

```bash
cd pmf
torchrun --nproc_per_node=4 eval_pmf_apt.py --arch L16 --model ../checkpoints/budgetpix_pmf_l16_256.pt --budget 128 --nfe 1 --first self --num 50000
PMF_WEIGHTS=/weights/pMF-L-16.pt torchrun --nproc_per_node=4 eval_pmf_apt.py --arch L16 --model released --budget 128 --num 50000   # no-training control
PMF_WEIGHTS=/weights/pMF-L-16.pt torchrun --nproc_per_node=4 eval_pmf_apt.py --arch L16 --model released --policy tome --reduce_budget 128 --num 50000   # ToMe / FeatSim baselines
cd ..
```

`--first` picks the layout of the one-step sample (`self`, `random` or `dense`); rows land in `evals/eval_pmf_apt.tsv`,
and the layout-adherence numbers (how closely the generated image's own entropy layout follows the given one) in
`evals/adherence_pmf_apt.tsv`.

