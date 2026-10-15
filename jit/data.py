"""ImageNet-1k training stream over the Hugging Face parquet shards (train-*.parquet: image bytes + label).

Shards are split across (rank, worker), rows are shuffled with a buffer, the stream loops forever.
Preprocessing: ADM centre crop + random horizontal flip. Layouts are drawn in the collate.
"""
import glob
import io
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from layout import pack, training_layouts


def center_crop_arr(pil_image, image_size):
    """ADM centre crop: halve while too large, resize the short side, crop the centre."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


class ImageNetParquet(IterableDataset):
    def __init__(self, data_dir, img_size=256, seed=0, rank=0, world_size=1, shuffle_buffer=2048):
        super().__init__()
        self.files = sorted(glob.glob(f"{data_dir}/train-*.parquet"))
        if not self.files:
            raise SystemExit(f"no train-*.parquet in {data_dir!r} (the ImageNet-1k parquet directory)")
        self.img_size, self.seed, self.rank, self.world_size = img_size, seed, rank, world_size
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        import pyarrow.parquet as pq
        wi = get_worker_info()
        nworkers = wi.num_workers if wi else 1
        cid = self.rank * nworkers + (wi.id if wi else 0)
        files = self.files[cid::self.world_size * nworkers]
        rng = random.Random(self.seed + cid)
        buf = []
        while True:
            order = files[:]
            rng.shuffle(order)
            for path in order:
                pf = pq.ParquetFile(path)
                for rg in range(pf.metadata.num_row_groups):
                    d = pf.read_row_group(rg, columns=["image", "label"]).to_pydict()
                    rows = list(range(len(d["label"])))
                    rng.shuffle(rows)
                    for i in rows:
                        try:
                            pil = Image.open(io.BytesIO(d["image"][i]["bytes"])).convert("RGB")
                        except Exception:
                            continue
                        pil = center_crop_arr(pil, self.img_size)
                        if random.random() < 0.5:
                            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
                        buf.append((torch.from_numpy(np.array(pil, copy=True)).permute(2, 0, 1).contiguous(),
                                    int(d["label"][i])))
                        if len(buf) >= self.shuffle_buffer:
                            j = rng.randrange(len(buf))
                            yield buf[j]
                            buf[j] = buf[-1]
                            buf.pop()


def collate(batch, geo, **layout_kw):
    """-> (images uint8 (B,3,H,W), labels (B,), packed training layout)."""
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, labels, pack(training_layouts(imgs, geo, **layout_kw), geo)
