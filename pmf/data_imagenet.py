"""Streaming ImageNet-1k train reader over the local parquet shards: ADM centre crop to 256, no random flip.
Yields (uint8 (3,H,W), label). Labels in these shards use the torchvision ImageFolder class order."""
import glob, io, random
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


def center_crop_arr(pil_image, image_size):            # ADM centre crop
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


class ImageNetParquet(IterableDataset):
    def __init__(self, data_dir, img_size=256, shuffle_buffer=2048, seed=0, rank=0, world_size=1, split="train"):
        """data_dir: ImageNet-1k as Hugging Face parquet shards (train-*.parquet)."""
        super().__init__()
        self.files = sorted(glob.glob(f"{data_dir}/{split}-*.parquet"))
        assert self.files, f"no {split} parquet in {data_dir}"
        self.img_size, self.shuffle_buffer, self.seed, self.rank, self.world_size = img_size, shuffle_buffer, seed, rank, world_size

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        cid, ncons = self.rank * nw + wid, self.world_size * nw
        mine = self.files[cid::ncons] or self.files          # few shards (val): every consumer reads all, own shuffle
        rng = random.Random(self.seed + cid)
        buf = []
        while True:
            order = mine[:]; rng.shuffle(order)
            for f in order:
                pf = pq.ParquetFile(f)
                for g in range(pf.metadata.num_row_groups):
                    d = pf.read_row_group(g, columns=["image", "label"]).to_pydict()
                    idx = list(range(len(d["label"]))); rng.shuffle(idx)
                    for i in idx:
                        try:
                            pil = Image.open(io.BytesIO(d["image"][i]["bytes"])).convert("RGB")
                        except Exception:
                            continue
                        img = torch.from_numpy(np.array(center_crop_arr(pil, self.img_size), copy=True)).permute(2, 0, 1).contiguous()
                        buf.append((img, int(d["label"][i])))
                        if len(buf) >= self.shuffle_buffer:
                            j = rng.randrange(len(buf)); yield buf[j]; buf[j] = buf[-1]; buf.pop()


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch], dtype=torch.long)
