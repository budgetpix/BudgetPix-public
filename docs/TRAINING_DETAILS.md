# Training details

The seven fine-tuning schemes of the paper, with every value read out of the config files, the trainer arguments
stored inside the checkpoints and the trainer source. Four class-conditional
(ImageNet) and three text-to-image schemes, plus the one-step pMF line described at the end:

| scheme | base model | resolution | code | config |
|---|---|---|---|---|
| JiT-B/16 256 | JiT-B/16 | 256 | `jit/` | `LINE=jit_b16_256 RELEASED=... DATA=... bash jit/train.sh` |
| JiT-B/32 512 | JiT-B/32 | 512 | `jit/` | `LINE=jit_b32_512 RELEASED=... DATA=... bash jit/train.sh` |
| JiT-L/16 256 | JiT-L/16 | 256 | `jit/` | `LINE=jit_l16_256 RELEASED=... DATA=... bash jit/train.sh` |
| JiT-L/32 512 | JiT-L/32 | 512 | `jit/` | `LINE=jit_l32_512 RELEASED=... DATA=... bash jit/train.sh` |
| MiniT2I-B/16 512 | MiniT2I-B/16 | 512 | `minit2i/` | `minit2i/configs/minit2i_b_512.yml` |
| MiniT2I-L/16 512 | MiniT2I-L/16 | 512 | `minit2i/` | `minit2i/configs/minit2i_l_512.yml` |
| PixelDiT-T2I 1.3B 1024 | PixelDiT-1300M-1024px | 1024 | `pixeldit/` | `pixeldit/configs/pixeldit_1024.yml` |

All added modules are zero-initialised, so at the start of fine-tuning the network reproduces the released model
exactly. The trainers of these seven schemes verify this contract before step 0 and refuse to train if it does not hold.

## Settings shared by all seven schemes

**Layout rule and its augmentation** (identical everywhere):
- Merge-only entropy quadtree over a luma histogram with `entropy_bins = 512` (so entropy is in [0, 9] bits).
- Per-scale merge thresholds `6.0` and `4.0` bits, finest-first.
- Per-image threshold offset `δ ~ U(0.0, 2.0)`, drawn per image per step, which spreads the training token count over
  roughly 25–100 % of the dense grid.
- `anchor_p = 0.15`: that fraction of images trains on the dense grid, which keeps the pretrained behaviour from drifting.
- `scramble_p = 0.10`: that fraction gets a content-independent random layout at a comparable token count, so the model
  handles any layout rather than memorising the entropy heuristic.
- Draw order per image: anchor, then scramble, then offset.
- The layout is drawn independently of the noise level `t`.

**Optimisation shape** (identical everywhere): AdamW with `weight_decay = 0`; linear warmup then a **constant** learning
rate (no cosine schedule, no decay); the added modules get `8×` the backbone learning rate;
EMA decay `0.9999` over all trainable parameters in fp32, and **evaluation always uses the EMA weights**; bf16 autocast
with fp32 master weights and fp32 EMA; `seed = 0`; every scheme fine-tunes **all** weights; the loss is the plain unweighted velocity MSE, with no auxiliary
term.

**Augmentation**: the JiT lines use JiT's own preprocessing, an ADM centre crop and a random horizontal flip
(p = 0.5). The text-to-image and pMF lines use no image augmentation: no flip, no crop jitter, no colour; their geometry
is fixed when the data is prepared (centre crop plus resize). Beyond that, the only augmentation is the layout
augmentation above.

---

## Per-scheme tables

### JiT-B/16, 256 px

| item | value |
|---|---|
| base checkpoint | released JiT-B/16 (https://github.com/LTH14/JiT), EMA weights (`model_ema1`) |
| model | `BudgetPixJiT` (`jit/model.py`), JiT-B |
| resolution / base patch / scales | 256 / 16 / 16, 32, 64 |
| thresholds (bits) | 6.0, 4.0 |
| parameters | 133.72 M total, of which 2.39 M added |
| steps / epochs | 142,987 / 100.0 |
| images seen | 128.1 M |
| global batch | 896 = 112 per GPU × 4 GPUs × 2 accumulation |
| learning rate | base `blr` 1e-5, scaled by batch/256 → backbone 3.50e-05; added modules ×8 → 2.80e-04 |
| warmup | 1000 steps, linear, then constant |
| optimizer | AdamW, betas (0.9, 0.95), weight decay 0, grad clip 3.0 (global norm) |
| EMA | 0.9999, fp32, on GPU |
| precision | bf16 autocast, fp32 master/EMA |
| noise scale | 1.0 |
| noise level law | `t ~ sigmoid(N(-0.8, 0.8))`, `t_eps = 5e-2` |
| conditioning dropout | label dropout 0.1 → null class index 1000 |
| data | ImageNet-1k parquet, ADM centre-crop + bicubic resize to 256, uint8 → [-1, 1]; random horizontal flip |
| checkpoint cadence | save every 1000, snapshot every 14,290 (every 10 epochs) |

### JiT-B/32, 512 px

| item | value |
|---|---|
| base checkpoint | released JiT-B/32, EMA weights |
| model | `BudgetPixJiT` (`jit/model.py`), JiT-B |
| resolution / base patch / scales | 512 / 32 / 32, 64, 128 |
| thresholds (bits) | 6.0, 4.0 — here they threshold the 64 px and 128 px merge levels |
| parameters | 135.79 M total, of which 2.40 M added |
| steps / epochs | 142,987 / 100.0 |
| images seen | 128.1 M |
| global batch | 896 = 112 × 4 GPUs × 2 |
| learning rate | backbone 3.50e-05; added modules 2.80e-04 |
| warmup | 1000, then constant |
| optimizer | AdamW (0.9, 0.95), wd 0, clip 3.0 |
| EMA / precision | 0.9999 fp32 on GPU / bf16 autocast |
| noise scale | 2.0 (set from the resolution: img / 256) |
| conditioning dropout | label dropout 0.1 |
| data | ImageNet-1k, centre crop + resize 512, random horizontal flip |
| checkpoint cadence | save 1000, snapshot 14,290 |

### JiT-L/16, 256 px

| item | value |
|---|---|
| base checkpoint | released JiT-L/16, EMA weights |
| model | `BudgetPixJiT` (`jit/model.py`), JiT-L |
| resolution / base patch / scales | 256 / 16 / 16, 32, 64 |
| thresholds (bits) | 6.0, 4.0 |
| parameters | 462.20 M total, of which 3.06 M added |
| steps / epochs | 71,494 / 50.0 |
| images seen | 64.06 M |
| global batch | 896 = 112 × 4 GPUs × 2 |
| learning rate | backbone 3.50e-05; added modules 2.80e-04 |
| warmup | 1000, then constant |
| optimizer | AdamW (0.9, 0.95), wd 0, clip 3.0 |
| EMA / precision | 0.9999 fp32 on GPU / bf16 autocast |
| noise scale | 1.0 |
| conditioning dropout | label dropout 0.1 |
| data | ImageNet-1k, centre crop + resize 256, random horizontal flip |
| checkpoint cadence | save 1000, snapshot 7,145 (every 5 epochs) |

### JiT-L/32, 512 px

| item | value |
|---|---|
| base checkpoint | released JiT-L/32, EMA weights |
| model | `BudgetPixJiT` (`jit/model.py`), JiT-L |
| resolution / base patch / scales | 512 / 32 / 32, 64, 128 |
| thresholds (bits) | 6.0, 4.0 (64 px and 128 px merge levels) |
| parameters | 464.86 M total, of which 3.06 M added |
| steps / epochs | 71,494 / 50.0 |
| images seen | 64.06 M |
| global batch | 896 = 56 per GPU × 4 GPUs × 4 accumulation |
| learning rate | backbone 3.50e-05; added modules 2.80e-04 |
| warmup | 1000, then constant |
| optimizer | AdamW (0.9, 0.95), wd 0, clip 3.0 |
| EMA / precision | 0.9999 fp32 on GPU / bf16 autocast |
| noise scale | 2.0 |
| conditioning dropout | label dropout 0.1 |
| data | ImageNet-1k, centre crop + resize 512, random horizontal flip |
| checkpoint cadence | save 1000, snapshot 7,145 |

### MiniT2I-B/16, 512 px

| item | value |
|---|---|
| base checkpoint | released `minit2i-b-16` (HF `MiniT2I/MiniT2I`), 375/375 tensors loaded |
| backbone | hidden 768, 17 double blocks, 2 text-preamble blocks, 12 heads × 64, MLP ratio 2.667, x-prediction |
| resolution / patch / scales | 512 / 16 / 16, 32, 64 (dense 1024 tokens) |
| thresholds (bits) | 6.0, 4.0 |
| added modules | multi-scale embedding (`ScaleMixer`, bottleneck 256) + token-local pixel-DiT refiner (width 64, depth 2, sub-patch 4, 4 heads) |
| parameters | 256.55 M pretrained + 2.26 M added |
| frozen inside backbone | `net.t_embedder.*`, `net.pooled_embedder.*` (MiniT2I never reads `vec`) |
| steps | 100,000 (single stage) |
| images seen | 25.6 M (~25.6 passes of the 1 M set) |
| global batch | 256 = micro 16 × 4 GPUs × 4 accumulation |
| learning rate | `blr` 1e-5 × batch/256 → 1.0e-05; added modules ×8 → 8.0e-05 |
| warmup | 1000, linear, both groups, then constant |
| optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, wd 0, clip 3.0 |
| EMA / precision | 0.9999 fp32 on GPU / bf16 autocast |
| noise scale / law | 2.0; `t ~ sigmoid(N(-0.8, 0.8))`, `eps ~ N(0, noise_scale²)`, x-prediction MSE in v-space |
| caption dropout | 0.10, applied by zeroing the attention mask |
| text encoder | `google/flan-t5-large`, frozen, bf16, 256 tokens, encoded on the fly |
| data | CC12M with LLaVA recaptions at 512 px, 1 M train + 5 k held out; no train-time augmentation |
| checkpoint cadence | save 2000, snapshot 10,000, log 50 |

### MiniT2I-L/16, 512 px — the only two-stage scheme

| item | value |
|---|---|
| base checkpoint | released `minit2i-l-16`, 495/495 tensors loaded |
| backbone | hidden 1248, 23 double blocks, 2 text-preamble blocks, 24 heads × 52, MLP ratio 2.705, x-prediction |
| resolution / patch / scales | 512 / 16 / 16, 32, 64 (dense 1024 tokens) |
| thresholds (bits) | 6.0, 4.0 |
| added modules | as MiniT2I-B: `ScaleMixer` (bottleneck 256) + pixel-DiT refiner (64 / 2 / 4 / 4 heads) |
| parameters | 908.61 M pretrained + 3.50 M added |
| **stage 1** | steps 1 → 24,000 on CC12M (1 M images, 512 px) |
| **stage 2** | steps 24,001 → 40,000 (16,000 steps) on the released 120 K alignment mix; resumed with model + EMA + optimizer state, so it runs past warmup at full LR |
| steps total / images seen | 40,000 / 10.24 M (stage 1 ≈ 6.1 CC12M passes; stage 2 ≈ 34.6 passes of the 118 k mix) |
| global batch | 256 = micro 4 × 4 GPUs × 16 accumulation (both stages) |
| learning rate | 1.0e-05 backbone; added modules 8.0e-05 |
| warmup | 1000, then constant |
| optimizer | AdamW (0.9, 0.95), eps 1e-8, wd 0, clip 3.0 |
| EMA / precision | 0.9999 fp32 on GPU / bf16 autocast |
| noise scale / law | 2.0; `t ~ sigmoid(N(-0.8, 0.8))`, x-prediction |
| caption dropout | 0.10 by zeroing the attention mask |
| text encoder | flan-t5-large, frozen, 256 tokens |
| stage-2 mix composition | 118,206 train + 1,024 held out = BLIP3o-60k 58,859 + dalle-3-dataset 19,016 + ShareGPT-4o-Image 41,355; seed 0; resize 512 bicubic + centre crop 512 |
| checkpoint cadence | save 2000; snapshot 4000 (stage 1) / 2000 (stage 2); log 50 |
| shipped weights | `checkpoint-final.pt` at step 40,000 |

### PixelDiT-T2I 1.3B, 1024 px

| item | value |
|---|---|
| base checkpoint | `pixeldit_t2i_v1.pth` = `nvidia/PixelDiT-1300M-1024px`, 385 tensors loaded |
| backbone | hidden 1536, 24 groups, patch depth 14, pixel depth 2, pixel hidden 16, pixel attention hidden 1152, 4 text blocks, text embed dim 2304, text max length 300, text RoPE (theta 10,000), REPA encoder index 6, pixel absolute positions on |
| resolution / patch / scales | 1024 / 16 / 16, 32, 64 (dense 4096 tokens) |
| thresholds (bits) | 6.0, 4.0 |
| added modules | multi-scale embedding (`ScaleMixer`, bottleneck 256) + `CellRefine` token-local refiner (width 64, depth 2, 4 heads) |
| parameters | 1302.47 M pretrained + 4.37 M added |
| steps | 30,000 (single stage) |
| images seen | 0.96 M = 30,000 × 32 (~4.8 passes of the 200 k precomputed set) |
| global batch | 32 = micro 4 × 4 GPUs × 2 accumulation |
| learning rate | 2.0e-05 absolute and constant (upstream's stage-3 value, **not** batch-scaled); added modules ×8 → 1.6e-04 |
| warmup | 2000 steps, then constant |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-15, wd 0, **grad clip 0.1** (upstream's value, not 3.0) |
| EMA | 0.9999, fp32, **kept on CPU** (1.3 B fp32 ≈ 5.2 GB off the card) |
| precision / memory | bf16 autocast, fp32 master/EMA; **gradient checkpointing on** (required at 1024 px) |
| diffusion law | flow matching, velocity target (noise − image), shifted linear-flow sigmas, `flow_shift 4.0`, 1000 diffusion steps, `sigma ~ logit_normal(0, 1)`; no `noise_scale` |
| caption dropout | 0.1 onto a precomputed **null embedding** (empty string encoded by Gemma), shape (1, 300, 2304) |
| text encoder | `gemma-2-2b-it`, frozen and never loaded during training — embeddings precomputed offline; 300 tokens, dim 2304 |
| data | PD12M at 1024 px, precomputed chunks (uint8 pixels + Gemma embeddings); 200,000 images = 2 × 100,000 from a 1 M-caption subset; short-side resize + centre crop |
| checkpoint cadence | save 1000, snapshot 3000, log 20 |

---

## What differs across the schemes

- **Backbone and initialisation**: each scheme starts from its own released checkpoint, and for JiT from that
  checkpoint's EMA weights (`model_ema1`).
- **Resolution and dense grid**: 256/256 tokens (JiT-B/16, JiT-L/16), 512/256 tokens (JiT-B/32, JiT-L/32 — patch 32),
  512/1024 tokens (MiniT2I), 1024/4096 tokens (PixelDiT).
- **Scale chain**: 16, 32, 64 everywhere except the JiT 512 lines, which use 32, 64, 128. Thresholds are always 6.0 and
  4.0 bits, finest-first, so on the JiT 512 lines they apply to the 64 px and 128 px merge levels.
- **Diffusion law**: JiT and MiniT2I use x-prediction with `t ~ sigmoid(N(-0.8, 0.8))` and `noise_scale` 1.0 (256 px) or
  2.0 (512 px); PixelDiT uses flow matching with a velocity target and no `noise_scale`.
- **Conditioning dropout**: label dropout 0.1 (JiT, class-conditional); caption dropout 0.1 by zeroing the attention
  mask (MiniT2I); caption dropout 0.1 onto a null embedding (PixelDiT).
- **Learning rate**: JiT 3.50e-05 backbone, MiniT2I 1.0e-05, PixelDiT 2.0e-05; the added modules always ×8. JiT and
  MiniT2I scale `blr` by batch/256; PixelDiT uses an absolute LR.
- **Optimizer constants**: betas (0.9, 0.95) and eps 1e-8 for JiT and MiniT2I; (0.9, 0.999) and eps 1e-15 for PixelDiT.
  Weight decay 0 everywhere.
- **Warmup**: 1000 steps, except PixelDiT with 2000. Constant afterwards in all schemes.
- **Grad clip**: 3.0, except PixelDiT with 0.1.
- **Global batch**: 896 (all JiT), 256 (both MiniT2I), 32 (PixelDiT). The per-GPU × accumulation split differs:
  112 × 2 (JiT except L/32), 56 × 4 (JiT-L/32), 16 × 4 (MiniT2I-B), 4 × 16 (MiniT2I-L), 4 × 2 (PixelDiT).
- **Length**: 142,987 steps / 100 epochs / 128.1 M images (JiT-B lines); 71,494 / 50 epochs / 64.06 M (JiT-L lines);
  100,000 steps / 25.6 M (MiniT2I-B); 40,000 / 10.24 M in two stages (MiniT2I-L); 30,000 / 0.96 M (PixelDiT).
- **Parameters added**: 2.39 M, 2.40 M, 3.06 M, 3.06 M (JiT B/16, B/32, L/16, L/32); 2.26 M, 3.50 M (MiniT2I B, L);
  4.37 M (PixelDiT) — i.e. 0.3–1.8 % of the base model in every case.
- **Memory tricks**: gradient checkpointing and EMA on CPU only for PixelDiT.
- **Staging**: only MiniT2I-L is two-stage (CC12M, then the 120 K alignment mix).
- **Data**: ImageNet-1k (JiT); CC12M + LLaVA recaptions (MiniT2I-B); CC12M then the 118 k mix (MiniT2I-L); PD12M
  (PixelDiT).

---

## The one-step line: pMF-L/16, 256 px

Base model: the released pMF-L/16 (pixel MeanFlow, https://github.com/Lyy-iiis/pMF), its EMA-1000 weights with the
auxiliary v-branch (the full training state converted to torch; the Hub's inference file carries the u-path only).
Same layout rule, thresholds, window and zero-init contract as the seven schemes above; the whole model is fine-tuned
with pMF's own objective (MeanFlow with the compound target, in-target guidance with sampled omega and CFG interval,
the auxiliary v-loss, LPIPS-VGG 0.4 and ConvNeXt-V2-B 0.1 perceptual terms for t < 0.8). The layout of an image is
drawn from the clean image and held constant in every network call of the step (the u forward and its JVP, both
guidance forwards, the auxiliary v and the perceptual x-hat), which keeps the MeanFlow identity exact.

| item | value |
|---|---|
| model variant | `AdaptivePMF` (`pmf/model_pmf_apt.py`), merged-token base = mean of the cell embeddings, pixel-DiT refine heads (width 64, depth 2, sub-patch 4, 4 heads) on both branches |
| resolution / base patch / scales | 256 / 16 / 16, 32, 64 (dense 256 tokens) |
| thresholds (bits) | 6.0, 4.0 |
| parameters | 511 M in the full training state (410 M u-path + 101 M v-branch), of which 4.77 M added |
| steps | 64,000: 0-56,000 at constant lr, 56,000-64,000 linear lr cooldown to 0 |
| images seen | ~5.1 M (~4 ImageNet epochs) |
| batch | 13 per process; 91 per step with 7 processes (52 for steps 40,000-58,000) |
| learning rate | 1e-5 on the pretrained weights, x8 on the added modules; 500-step linear warm-up, then constant |
| optimizer | AdamW (0.9, 0.95), weight decay 0, grad clip 20 (pMF itself trained with Muon) |
| EMA | 0.9995, every saved step is usable |
| layout window | anchor_p 0.30 (0.15 on the other lines), scramble_p 0.10, offset U(0, 2) |
| guidance law | pMF's: omega ~ [1, 8] with density proportional to omega^-2 (cfg_beta 2.0, the released L/16 config), t_min ~ U(0, 0.5), t_max ~ U(0.5, 1) |
| (t, r) law | logit-normal(0.8, 0.8), r = t for 50 % of the batch |
| data | ImageNet-1k parquet, ADM centre crop to 256, no flip |
| checkpoint cadence | every 2000 steps |
| sampling | 1 NFE, omega 7.0 on [0.2, 0.7] at every budget; the layout is chosen up front (see `pmf/README.md`) |

## Model architecture

### The base models

| | JiT-B/16 256 | JiT-B/32 512 | JiT-L/16 256 | JiT-L/32 512 | MiniT2I-B/16 512 | MiniT2I-L/16 512 | PixelDiT 1024 |
|---|---|---|---|---|---|---|---|
| depth | 12 | 12 | 24 | 24 | 17 double + 2 text | 23 double + 2 text | 14 patch + 2 pixel |
| width | 768 | 768 | 1024 | 1024 | 768 | 1248 | 1536 |
| heads × head dim | 12 × 64 | 12 × 64 | 16 × 64 | 16 × 64 | 12 × 64 | 24 × 52 | 24 × 64 |
| MLP ratio | 4.0 (SwiGLU) | 4.0 | 4.0 | 4.0 | 2.667 | 2.705 | 4.0 |
| patch | 16 | 32 | 16 | 32 | 16 | 16 | 16 |
| dense tokens | 256 | 256 | 256 | 256 | 1024 | 1024 | 4096 |
| conditioning | 32 in-context class tokens from block 4 | same | from block 8 | from block 8 | text, no adaLN | text, no adaLN | text + `silu(t)` modulation |
| parameters | 131.32 M | 133.39 M | 459.14 M | 461.80 M | 258.12 M | 911.77 M | 1302.47 M |

All three families share a bottleneck patch embedding (`Conv2d(3, 128, patch, patch)` then `Conv2d(128, width, 1)`),
RMSNorm with qk-norm, and SwiGLU feed-forwards.

- **JiT** is class-conditional, has no text encoder, and uses per-block adaLN.
- **MiniT2I** has **no adaLN at all** — the released double-stream blocks and final layer ignore the conditioning
  vector, so its timestep and pooled-text embedders reach no layer (this port does not compute them). Text is
  `flan-t5-large`, frozen, 256 tokens of dim 1024, projected to the trunk width; it is encoded on the fly from the
  stored token ids.
- **PixelDiT** has a patch stage (14 MM-DiT blocks at width 1536) plus a **pixel stage** (2 blocks, pixel hidden 16,
  attention hidden 1152) that always decodes all 1024² pixels — 4096 cells × 256 pixels — at every budget. Text is
  `gemma-2-2b-it`, frozen, 300 tokens of dim 2304, precomputed offline. Its config lists 4 text-refine blocks, but the
  released model sets them to `None`, so they are unused.

Parameter counts were summed over the checkpoints themselves. Counting only *trainable* pretrained parameters gives
slightly less for MiniT2I (256.55 M and 908.61 M), because `t_embedder` and `pooled_embedder` are frozen: the trunk
never reads their output.

### The added modules

Three additions, in every port, each gated so it contributes nothing at initialisation:

**(a) Multi-scale patch embedding** — how a merged token of size s = k·(base patch) gets one embedding.
- **JiT (`ScaleEncoder`, one per merged scale):** two paths. The pretrained patch conv is applied to the bilinearly
  downsampled s×s patch, and a learned tap pools the k×k pretrained sub-patch embeddings:
  `Conv2d(width, 256, 1)` → GELU → depthwise `Conv2d(256, 256, k, stride k)` → pointwise `Conv2d(256, 256, 1)` → GELU
  → `Linear(256, width)`, the last **zero-initialised**, plus a learned sub-patch position `(1, width, k, k)`.
  So at initialisation a merged token is exactly the pretrained embedding of its downsampled patch.
- **MiniT2I and PixelDiT (`ScaleMixer`):** one path and simpler — the pretrained embedding is run **once over the whole
  image**, and a merged token is the **mean of its k×k pretrained sub-patch embeddings** plus
  `Conv2d(width, 256, 1)` → depthwise `Conv2d(256, 256, k)` → `Linear(256, width)` (zero-init), with no nonlinearity.

**(b) Token-local refinement** — how one token paints the s×s pixels it owns, instead of a single flat prediction.
- **JiT and MiniT2I (`PixelDiTRefineHead`, one per merged scale):** the pretrained final layer predicts the token's
  base-sized patch; that prediction is bilinearly upsampled to s×s and corrected by a small DiT that runs on the
  token's own pixels: 4×4 sub-patches → `Linear(48, 64)` + learned positions → 2 blocks (RMSNorm, qkv `Linear(64, 192)`,
  qk-RMSNorm, 4-head SDPA, `Linear(64, 64)`, MLP 64→256→64, and adaLN `SiLU → Linear(width, 6·64)` conditioned on the
  token's own hidden state) → `Linear(64, 4·4·r²·3)` (zero-init) → pixel shuffle. Pixels are scattered into the image
  and accumulated, not unpatchified as squares.
- **PixelDiT (`CellRefine`):** no pixel-space head, because the released pixel stage already decodes every pixel. The
  problem there is that a merged token broadcasts one conditioning vector to all k² cells, so the addition is a
  **hidden-space** per-cell correction: learned slots `(1, k², 64)` → 2 blocks (LayerNorm without affine, 4-head SDPA,
  MLP 64→256→64, adaLN `Linear(1536, 6·64)` zero-init) → `Linear(64, 1536)` (zero-init), added to the per-cell
  conditioning before the released pixel blocks run unchanged.

**(c) Position and scale information.**
- **JiT** adds a zero-initialised scale embedding `nn.Embedding(#scales, width)` to every token. Additive sincos is
  computed analytically at the token centre, and base-scale tokens are overwritten with the **exact pretrained table
  row**, so a dense layout is bit-exact. 2D RoPE is evaluated at the same centres.
- **MiniT2I** has **no scale embedding**; scale enters only through which per-scale module is used and through the
  token centre. Its additive sincos is the released grid table re-expressed analytically, computed in fp32.
- **PixelDiT** has no additive image sincos (the released model has none); its RoPE table is interpolated from
  upstream's own `linspace`, so integer centres stay bit-exact.
- In all three, a merged token's centre is `(top-left + s/2)/base − 0.5`, i.e. the centre of the cells it covers, which
  is fractional for k > 1.
- Attention plumbing differs: JiT packs variable-length sequences (`flash_attn_varlen_func`, no padding); MiniT2I pads
  image tokens and masks with an additive −inf bias; PixelDiT passes a boolean mask, built only when padding exists.

No model receives its token budget as an input.

### The zero-initialisation contract, in code

| port | zero-initialised tensors | file |
|---|---|---|
| JiT | scale embedding; `ScaleEncoder.up` weight+bias; refine-head adaLN weight+bias; refine-head `out` weight+bias | `jit/model.py` |
| MiniT2I | `ScaleMixer.up` weight+bias; `PixDiTBlock` adaLN weight+bias; refine-head `out` weight+bias | `minit2i/model.py` |
| PixelDiT | `ScaleMixer.up` weight+bias; `RefineBlock.ada` weight+bias; `CellRefine.out` weight+bias | `pixeldit/apt/model_apt.py` |

Each block's two residual additions are multiplied by a gate that comes from the zero adaLN projection
(`x = x + g₁·attn(...)`, `x = x + g₂·mlp(...)`), so both are exactly zero at step 0. The learned positions and slots
(`pos`, `mini_pos`, `slots`) are *not* zero — they are `trunc_normal_(0.02)` — but each sits upstream of a zero output
projection, so the contract holds. The JiT trainer asserts that these tensors are zero before step 0; the MiniT2I and
PixelDiT trainers check that the dense forward equals the released model's.

### Added parameters

| model | base | added | added % |
|---|---|---|---|
| JiT-B/16 256 | 131,320,320 | 2,394,944 | 1.82 % |
| JiT-B/32 512 | 133,387,008 | 2,401,088 | 1.80 % |
| JiT-L/16 256 | 459,139,808 | 3,056,704 | 0.67 % |
| JiT-L/32 512 | 461,796,320 | 3,062,848 | 0.66 % |
| MiniT2I-B/16 512 | 258,122,880 | 2,261,056 | 0.88 % |
| MiniT2I-L/16 512 | 911,766,696 | 3,500,416 | 0.38 % |
| PixelDiT 1024 | 1,302,470,787 | 4,373,504 | 0.34 % |

Counted from the checkpoints by key prefix (`scale_emb`, `x_embedder.enc`, `final_layer.refine` for JiT; `mixers.`,
`refines.` for the T2I ports); the base column equals the parameter count of each released base model.

**The adaLN projection `Linear(width, 6·64)` is about half of the added parameters (49–55 %)** in every model, because it is the only place where the full trunk width meets the small head. The heads themselves are
tiny (width 64, depth 2).

### Architectural differences between the ports

1. JiT has a scale embedding; the two T2I ports do not.
2. A merged token is embedded from the downsampled patch plus a pooled tap (JiT), or as the mean of the pretrained
   sub-patch embeddings plus a depthwise mix (MiniT2I, PixelDiT).
3. Refinement is in pixel space on top of a bilinear upsample (JiT, MiniT2I) or in hidden space on the per-cell
   conditioning (PixelDiT).
4. PixelDiT's pixel stage always runs at full resolution, so adaptivity acts on its patch stage only — that sets its
   cost floor and is the reason its speed-ups are the smallest of the three families.
5. The JiT 512 lines change the **base patch** (32, with scales 32/64/128), not just the image size; every other line
   uses base 16 with scales 16/32/64.
6. Merging is strictly coarse-only: no layout splits below the base patch.

