"""
Streaming WebDataset IterableDataset for the Indian Road Dataset.

Features:
- Progressive WebDataset TAR shard streaming directly from Hugging Face Hub.
- Strict isolation of all 25 benchmark validation clips (zero data leakage).
- Worker-disjoint shard partitioning preventing duplicate samples across workers.
- In-memory sample shuffle buffer to decouple consecutive video frames.
- Image decoding and letterbox/resize to (640, 640).
- Robust JSON annotation parsing with class mapping to the 12 target classes.
- Full collation returning (images [B, 3, 640, 640], targets [M, 6]).
"""

import io
import json
import math
import os
from datetime import datetime
from pathlib import Path
import random
import tarfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from src.data.benchmark_leakage import is_benchmark_validation_sample
from src.data.shard_cache import BoundedShardCache

# The 12 official benchmark classes in exact order
CLASS_NAMES: List[str] = [
    "person",           # 0
    "rider",            # 1
    "car",              # 2
    "truck",            # 3
    "bus",              # 4
    "motorcycle",       # 5
    "bicycle",          # 6
    "autorickshaw",     # 7
    "animal",           # 8
    "vehicle fallback", # 9
    "traffic light",    # 10
    "traffic sign",     # 11
]

CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_bdd_labels(
    json_bytes: bytes,
    orig_w: int,
    orig_h: int,
) -> torch.Tensor:
    """
    Parses BDD100K JSON annotations and converts bounding boxes to normalized
    YOLO coordinates: [class_id, cx, cy, w, h].
    Returns tensor of shape [N, 5].
    """
    try:
        data = json.loads(json_bytes.decode("utf-8"))
    except Exception:
        return torch.zeros((0, 5), dtype=torch.float32)

    labels = data.get("labels", [])
    if not labels:
        return torch.zeros((0, 5), dtype=torch.float32)

    boxes: List[List[float]] = []
    for item in labels:
        category = item.get("category", "")
        if category not in CLASS_TO_ID:
            continue
        cls_id = float(CLASS_TO_ID[category])

        box2d = item.get("box2d")
        if not box2d:
            continue

        try:
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
        except (KeyError, TypeError, ValueError):
            continue

        # Clamp to image boundaries
        x_min = max(0.0, min(float(orig_w), min(x1, x2)))
        x_max = max(0.0, min(float(orig_w), max(x1, x2)))
        y_min = max(0.0, min(float(orig_h), min(y1, y2)))
        y_max = max(0.0, min(float(orig_h), max(y1, y2)))

        w = x_max - x_min
        h = y_max - y_min

        # Filter degenerate/zero-area boxes (< 2 pixels in either dimension)
        if w < 2.0 or h < 2.0:
            continue

        cx = (x_min + x_max) / 2.0 / float(orig_w)
        cy = (y_min + y_max) / 2.0 / float(orig_h)
        norm_w = w / float(orig_w)
        norm_h = h / float(orig_h)

        # Coordinate safeguard clamp
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        norm_w = max(0.001, min(0.999, norm_w))
        norm_h = max(0.001, min(0.999, norm_h))

        boxes.append([cls_id, cx, cy, norm_w, norm_h])

    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


class StreamingIndianRoadDataset(IterableDataset):
    """
    High-throughput IterableDataset streaming WebDataset shards from Hugging Face
    with bounded caching, worker partitioning, and benchmark leakage filtering.
    """

    def __init__(
        self,
        shard_ids: Optional[List[int]] = None,
        total_shards: int = 646,
        cache: Optional[BoundedShardCache] = None,
        cache_dir: str = "data/cache_shards",
        max_cached_shards: int = 10,
        prefetch_ahead: int = 2,
        img_size: Tuple[int, int] = (640, 640),
        shuffle: bool = True,
        shuffle_buffer_size: int = 256,
        seed: int = 42,
        epoch: int = 0,
        exclude_benchmark_val: bool = True,
    ):
        super().__init__()
        self.total_shards = total_shards
        self.shard_ids = list(range(total_shards)) if shard_ids is None else shard_ids
        self.cache_dir = cache_dir
        self.max_cached_shards = max_cached_shards
        self.prefetch_ahead = prefetch_ahead
        self._cache = cache
        self.img_size = img_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = epoch
        self.exclude_benchmark_val = exclude_benchmark_val

        # Diagnostic counters
        self.samples_streamed = 0
        self.samples_excluded_val = 0
        self.samples_corrupted = 0

    @property
    def cache(self) -> BoundedShardCache:
        if self._cache is None:
            self._cache = BoundedShardCache(
                cache_dir=self.cache_dir,
                max_shards=self.max_cached_shards,
                prefetch_ahead=self.prefetch_ahead,
            )
        return self._cache

    def __getstate__(self) -> Dict[str, Any]:
        """Ensures the dataset can be pickled by PyTorch spawn workers on Windows."""
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores state in child worker process."""
        self.__dict__.update(state)
        self._cache = None

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for deterministic shard shuffle."""
        self.epoch = epoch

    def _get_worker_shards(self) -> List[int]:
        """Partitions the shard list deterministically across DataLoader workers."""
        worker_info = get_worker_info()
        shards = list(self.shard_ids)

        if self.shuffle:
            # Deterministic pseudo-random shuffle per epoch
            rng = random.Random(self.seed + self.epoch * 10007)
            rng.shuffle(shards)

        if worker_info is None:
            return shards

        # Disjoint slice per worker
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return shards[worker_id::num_workers]

    def _process_shard(self, shard_id: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Extracts and yields valid (image_tensor, target_tensor) from a single shard."""
        shard_path = self.cache.get_shard(shard_id, total_shards=self.total_shards, auto_prefetch=False)

        # In-memory dictionary grouping by sample stem
        # stem -> {"jpg": bytes, "json": bytes}
        pending_samples: Dict[str, Dict[str, bytes]] = {}

        try:
            with tarfile.open(shard_path, "r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    name = member.name
                    # Only process .jpg and .json (ignore .png segmentation masks)
                    if not (name.endswith(".jpg") or name.endswith(".json")):
                        continue

                    # Extract stem
                    stem = Path(name).stem
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    data_bytes = f.read()

                    if stem not in pending_samples:
                        pending_samples[stem] = {}

                    if name.endswith(".jpg"):
                        pending_samples[stem]["jpg"] = data_bytes
                    elif name.endswith(".json"):
                        pending_samples[stem]["json"] = data_bytes

                    # Check if sample is complete (both jpg and json available)
                    if "jpg" in pending_samples[stem] and "json" in pending_samples[stem]:
                        sample = pending_samples.pop(stem)
                        result = self._decode_sample(stem, sample["jpg"], sample["json"])
                        if result is not None:
                            yield result

        except Exception as e:
            self.samples_corrupted += 1
            # Continue to next shard gracefully on tar read errors

    def _decode_sample(
        self, stem: str, jpg_bytes: bytes, json_bytes: bytes
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Decodes image bytes and parses JSON annotation with leakage check."""
        # 1. Benchmark Data Leakage Check
        if self.exclude_benchmark_val and is_benchmark_validation_sample(stem):
            self.samples_excluded_val += 1
            return None

        # 2. Decode Image
        try:
            pil_img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
            orig_w, orig_h = pil_img.size
        except Exception:
            self.samples_corrupted += 1
            return None

        # 3. Parse Bounding Boxes
        targets = parse_bdd_labels(json_bytes, orig_w, orig_h)

        # 4. Fast Resize to Target Image Size
        target_h, target_w = self.img_size
        if (orig_w, orig_h) != (target_w, target_h):
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)

        # 5. Convert to Tensor [3, H, W] in [0, 1]
        img_np = np.array(pil_img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        self.samples_streamed += 1
        return img_tensor, targets

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        w_id = worker_info.id if worker_info else 0
        w_total = worker_info.num_workers if worker_info else 1
        worker_shards = self._get_worker_shards()
        buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        rng = random.Random(self.seed + self.epoch * 10007 + w_id)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Worker {w_id}/{w_total}] Initialized with {len(worker_shards)} shards assigned for epoch {self.epoch}", flush=True)

        for idx, shard_id in enumerate(worker_shards):
            # Prefetch upcoming shards assigned to this worker
            next_shards = worker_shards[idx + 1 : idx + 1 + self.prefetch_ahead]
            if next_shards:
                self.cache.prefetch(next_shards)

            shard_samples = 0
            for sample in self._process_shard(shard_id):
                shard_samples += 1
                if not self.shuffle or self.shuffle_buffer_size <= 1:
                    yield sample
                else:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer_size:
                        # Pop random element from buffer
                        pick_idx = rng.randint(0, len(buffer) - 1)
                        yield buffer.pop(pick_idx)

            # Evict consumed shard to maintain bounded disk footprint
            self.cache.evict(shard_id)

        # Drain remaining buffer
        if buffer:
            if self.shuffle:
                rng.shuffle(buffer)
            for sample in buffer:
                yield sample


def collate_streaming_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for streaming DataLoader.
    Batches images into [B, 3, H, W].
    Formats bounding boxes into [M, 6] -> [batch_idx, class_id, cx, cy, w, h].
    """
    # Filter out empty/None samples if any
    valid = [item for item in batch if item is not None and item[0] is not None]
    if not valid:
        return torch.zeros((0, 3, 640, 640)), torch.zeros((0, 6))

    images = torch.stack([item[0] for item in valid], dim=0)

    # Attach batch index to targets
    all_targets: List[torch.Tensor] = []
    for b_idx, (_, boxes) in enumerate(valid):
        if boxes.numel() > 0:
            num_boxes = boxes.size(0)
            b_tensor = torch.full((num_boxes, 1), b_idx, dtype=torch.float32)
            # targets: [batch_idx, class_id, cx, cy, w, h]
            all_targets.append(torch.cat([b_tensor, boxes], dim=1))

    if all_targets:
        targets_batch = torch.cat(all_targets, dim=0)
    else:
        targets_batch = torch.zeros((0, 6), dtype=torch.float32)

    return images, targets_batch
