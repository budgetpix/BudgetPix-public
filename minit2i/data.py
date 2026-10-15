"""Training data and text encoding.

Data format (tensor chunks, written by mix120k_build.py or cc12m_build.py):
    <dir>/chunk_*.pt = {"pixel_values": uint8 (n,3,512,512), "input_ids": (n,256), "attention_mask": (n,256),
                        "caption": list[str] (optional)}
Chunks are split across ranks x workers, shuffled per pass, samples shuffled inside each chunk; infinite.
"""
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset


class ChunkStream(IterableDataset):
    def __init__(self, root, seed=0, rank=0, world_size=1):
        self.files = sorted(Path(root).glob("chunk_*.pt"))
        if not self.files:
            raise SystemExit(f"no chunk_*.pt under {root}")
        self.seed, self.rank, self.world_size = int(seed), rank, world_size

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        nw, wid = (w.num_workers, w.id) if w is not None else (1, 0)
        files = self.files[self.rank * nw + wid::self.world_size * nw]
        if not files:
            raise RuntimeError(f"{len(self.files)} chunks for {self.world_size * nw} ranks x workers; lower data.num_workers")
        rng = random.Random(self.seed + self.rank * 1000 + wid)
        while True:
            order = list(files)
            rng.shuffle(order)
            for path in order:
                chunk = torch.load(path, map_location="cpu", weights_only=False)
                idx = list(range(int(chunk["pixel_values"].shape[0])))
                rng.shuffle(idx)
                for i in idx:
                    yield {"pixel_values": chunk["pixel_values"][i], "input_ids": chunk["input_ids"][i].long(),
                           "attention_mask": chunk["attention_mask"][i].long()}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in ("pixel_values", "input_ids", "attention_mask")}


def make_loader(root, batch_size, num_workers=8, seed=0, rank=0, world_size=1):
    return DataLoader(ChunkStream(root, seed, rank, world_size), batch_size=batch_size, num_workers=num_workers,
                      collate_fn=collate, persistent_workers=num_workers > 0,
                      prefetch_factor=2 if num_workers > 0 else None, drop_last=True)


def to_unit(pixel_values, device):
    """uint8 -> [-1, 1] (float input is taken as already in [-1, 1])."""
    x = pixel_values.to(device, non_blocking=True)
    return x.to(torch.float32).mul_(1.0 / 127.5).add_(-1.0) if x.dtype == torch.uint8 else x


class TextEncoder:
    """Frozen flan-t5-large encoder: captions or token ids -> (states (B,P,1024), mask (B,P))."""

    def __init__(self, llm="google/flan-t5-large", device="cpu", max_length=256):
        from transformers import AutoTokenizer, T5EncoderModel
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.encoder = T5EncoderModel.from_pretrained(llm).to(device).eval().requires_grad_(False)
        self.device, self.max_length = device, max_length

    @torch.no_grad()
    def encode_ids(self, ids, mask):
        ids, mask = ids.to(self.device), mask.to(self.device)
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask.float()

    def __call__(self, captions):
        b = self.tokenizer(captions, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        return self.encode_ids(b.input_ids, b.attention_mask)
