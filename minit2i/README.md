# BudgetPix on MiniT2I (text-to-image, 512x512)

The released MiniT2I-B/16 and MiniT2I-L/16 (pixel-space MM-DiT, frozen flan-t5-large text encoder, 1024 tokens dense
at 512x512) fine-tuned with the BudgetPix layout and the pixel-DiT refiner. The released model code is imported at run
time from a checkout of the official PyTorch port:

```bash
git clone https://github.com/Hope7Happiness/minit2i-torch upstream/minit2i-torch     # from the repository root
export MINIT2I_ROOT=$PWD/upstream/minit2i-torch
```

| file | content |
|---|---|
| `model.py` | the released model wrapped with the BudgetPix embedding, positions, masking and refine head; loading |
| `layout.py` | the entropy quadtree, budget calibration and packing |
| `diffusion.py` | the training loss and the Euler sampler |
| `data.py` | the training data stream and the text encoder |
| `sample.py`, `train.py` (+ `configs/`), `eval.py` | the three entry points |
| `mix120k_build.py`, `cc12m_build.py` | build the training data from public sources |

## Sample

```bash
cd minit2i
python sample.py --ckpt ../checkpoints/budgetpix_minit2i_l_512.pt \
    --prompts "a corgi in a spacesuit" "a watercolor of a lighthouse at dusk" --budget 512 --cfg 6.0 --steps 50
cd ..
```

`--budget` is the token count per image (1024 = dense, 512 = 50 %, 256 = 25 %); omit it for the dense grid. The text
encoder (`google/flan-t5-large`) is downloaded from the Hub on first use. The sampler runs 5 steps dense, then
re-derives the layout from the model's own x0 estimate every 5 steps. Guidance 6.0 for the L model and 2.5-5.0 for the
B model are the released defaults at the dense grid (`sample.py` uses 5.0 unless `--cfg` is given); the best value
rises as the budget falls. `--released <dir>`
runs the released MiniT2I at the same budget instead (the untrained control; the geometry comes from `--config`).

## Train

Data: MiniT2I tensor chunks, `chunk_*.pt` files holding `pixel_values` (uint8, 3x512x512), `input_ids` and
`attention_mask` (flan-t5-large tokens, length 256). Two builders write that format from public sources:

* `mix120k_build.py`: the 120K alignment mix MiniT2I itself was fine-tuned on (BLIP3o-60k + DALL-E 3 +
  ShareGPT-4o-Image, all public on the Hub). The L line trained on it, and it is the recommended corpus.
* `cc12m_build.py`: CC12M with LLaVA-NeXT recaptions (the B line's data; images are fetched from their URLs).

Base weights: the released HF checkpoints (`MiniT2I/MiniT2I`, a directory with `transformer/{config.json,*.safetensors}`).
Point `checkpoint` and `data.local_dataset_dir` in the config at your copies, then:

```bash
cd minit2i
torchrun --nproc_per_node=4 train.py --config configs/minit2i_l_512.yml --resume
cd ..
```

The global batch is 256 on both lines (`micro_batch_size x GPUs x grad_accum`, checked at the first step; adjust
`grad_accum` to your GPU count). The L line of the paper ran 24k steps on CC12M and then 16k on the mix; the B line
100k steps on CC12M. Training starts only if the adapted model reproduces the released one on the dense grid.
Checkpoints hold `{model, ema, optimizer, step, cfg}`; `sample.py` and `eval.py` read them directly and use the EMA
weights. All details are in [`docs/TRAINING_DETAILS.md`](../docs/TRAINING_DETAILS.md).

## Evaluate

FID / IS at a budget on held-out captions. The samples of a held-out chunk folder (1,024 for the 120K mix) give both
the captions and the reference images, which are written once to `evals/`; `--n` takes only the first n:

```bash
cd minit2i
torchrun --nproc_per_node=4 eval.py --ckpt ../checkpoints/budgetpix_minit2i_l_512.pt \
    --data /data/mix120k/chunks_heldout --budget 512 --cfg 5.0
cd ..
```

The paper's budgets are 1024, 911, 819, 717, 614, 512, 410, 307 and 256 tokens, with guidance 5.0 down to 512 tokens
and 7.0 below. 50 Euler steps; image k always gets the same noise, so every budget sees the same noise. GenEval and
DPG-Bench are scored with their official toolkits on images from `sample.py` at 100 steps and guidance 5.0.
