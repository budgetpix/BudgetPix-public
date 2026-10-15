# BudgetPix on PixelDiT (text-to-image, 1024x1024)

The released PixelDiT-T2I 1.3B (NVIDIA, `nvidia/PixelDiT-1300M-1024px`; 4096 tokens dense at 1024x1024, frozen
gemma-2-2b-it text encoder, flow matching) fine-tuned with the BudgetPix layout and the token-local refiner. `apt/` is
the port; the released model classes are imported at run time from a checkout of the official repository:

```bash
git clone https://github.com/NVlabs/PixelDiT upstream/PixelDiT     # from the repository root
export PIXELDIT_ROOT=$PWD/upstream/PixelDiT
```

PixelDiT and its released weights are distributed under the NVIDIA License (NSCLv1, non-commercial research use);
the BudgetPix PixelDiT checkpoint is a derivative of those weights and carries the same terms.

## Sample

```bash
cd pixeldit
python sample_apt.py --ckpt ../checkpoints/budgetpix_pixeldit_1024.pt \
    --prompts "a red fox in a snowy forest, golden hour" --budget 2048 --cfg 2.75 --steps 50 --out samples.png
cd ..
```

`--budget` is the token count per image (4096 = dense, 2048 = 50 %, 1024 = 25 %). Captions are encoded with the
released recipe (gemma-2-2b-it with the CHI prompt prefix, `apt/text.py`); the released negative prompt drives
classifier-free guidance. The sampler is the released one (DPM-Solver++ 2M, 50 steps, guidance 2.75) with the token
layout re-estimated every 5 steps from the running clean-image estimate. Note that PixelDiT's pixel stage always runs
over every pixel, so the compute saving at a budget comes from the patch stage only; `eval_apt.py` reports both parts.

## Train

Data: precomputed chunks (uint8 pixels + Gemma caption embeddings), written once by `precompute_data.py` from a folder
of images with sidecar captions or from webdataset shards:

```bash
cd pixeldit
python precompute_data.py --images /data/images_1024 --out /data/chunks_1024 --size 1024 --per_chunk 64
python encode_prompts.py --null_only --out eval/null_emb.pt        # the caption-dropout embedding
cd ..
```

The paper's line trained on a 1M-caption subset of PD12M (200k precomputed images). Base weights:
`pixeldit_t2i_v1.pth` and `config.json` from the Hub repo above. Point `checkpoint`, `llm` and `data.chunks` in the
config at your copies, then:

```bash
cd pixeldit
torchrun --nproc_per_node=4 train_apt.py --config configs/pixeldit_1024.yml --resume auto
cd ..
```

Global batch 32 (micro 4 x 4 GPUs x accumulation 2), 30k steps, gradient checkpointing on, EMA kept on the CPU; the
optimiser constants are upstream's stage-3 values (constant lr 2e-5, clip 0.1). See
[`docs/TRAINING_DETAILS.md`](../docs/TRAINING_DETAILS.md).

## Evaluate

```bash
cd pixeldit
python precompute_data.py --tars /data/shards_heldout --out /data/chunks_heldout --size 1024    # held-out images + captions, once
python encode_prompts.py --chunks /data/chunks_heldout --n 2000 --out eval/prompts_2k.pt
python make_fid_ref.py --chunks /data/chunks_heldout --n 2000 --out eval/real_2k
torchrun --nproc_per_node=4 eval_apt.py --line budgetpix --ckpt ../checkpoints/budgetpix_pixeldit_1024.pt \
    --prompts eval/prompts_2k.pt --real eval/real_2k --budgets 4096,3645,3277,2867,2458,2048,1638,1229,1024 \
    --cfgs "4096:2.75,3645:2.75,3277:2.75,2867:2.75,2458:4.5,2048:4.5,1638:4.5,1229:4.5,1024:4.5" --tsv evals/ladder_1024.tsv
cd ..
```

