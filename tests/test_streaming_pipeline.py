"""
Unit tests for the IRD V2 streaming dataset, bounded cache, and leakage protection.
"""

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.benchmark_leakage import (
    BENCHMARK_VAL_CLIPS,
    extract_clip_id,
    is_benchmark_validation_sample,
)
from src.data.shard_cache import BoundedShardCache
from src.data.streaming_dataset import (
    StreamingIndianRoadDataset,
    collate_streaming_batch,
    parse_bdd_labels,
)


class TestBenchmarkLeakage(unittest.TestCase):
    def test_val_clip_count(self):
        self.assertEqual(len(BENCHMARK_VAL_CLIPS), 25)

    def test_extract_clip_id(self):
        cid = "001c0d67-590e-479b-86b9-c521fe884139"
        self.assertEqual(extract_clip_id(f"{cid}__0000.jpg"), cid)
        self.assertEqual(extract_clip_id(f"{cid}_0000.jpg"), cid)
        self.assertEqual(extract_clip_id(f"{cid}/0000.jpg"), cid)

    def test_validation_exclusion(self):
        # Sample in validation clips
        self.assertTrue(is_benchmark_validation_sample("001c0d67-590e-479b-86b9-c521fe884139_0000.jpg"))
        # Sample in training clips (fake uuid)
        self.assertFalse(is_benchmark_validation_sample("ffffffff-ffff-ffff-ffff-ffffffffffff_0000.jpg"))


class TestBDDLabelParsing(unittest.TestCase):
    def test_parse_valid_boxes(self):
        bdd_json = {
            "labels": [
                {"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}},
                {"category": "motorcycle", "box2d": {"x1": 50, "y1": 50, "x2": 90, "y2": 150}},
                {"category": "unknown_class", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}},
            ]
        }
        json_bytes = json.dumps(bdd_json).encode("utf-8")
        boxes = parse_bdd_labels(json_bytes, orig_w=1000, orig_h=1000)
        self.assertEqual(boxes.shape[0], 2)
        # Class 2 is car
        self.assertEqual(boxes[0, 0].item(), 2.0)
        # cx, cy, w, h
        self.assertAlmostEqual(boxes[0, 1].item(), 0.15, places=2)
        self.assertAlmostEqual(boxes[0, 3].item(), 0.10, places=2)

    def test_parse_empty_or_corrupt(self):
        self.assertEqual(parse_bdd_labels(b"", 100, 100).shape, (0, 5))
        self.assertEqual(parse_bdd_labels(b"invalid json", 100, 100).shape, (0, 5))
        self.assertEqual(parse_bdd_labels(b'{"labels": []}', 100, 100).shape, (0, 5))


class TestBoundedShardCache(unittest.TestCase):
    def test_cache_bounds_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BoundedShardCache(cache_dir=tmpdir, max_shards=3)
            # Create 5 dummy shard files
            for i in range(5):
                p = Path(tmpdir) / f"train-{i:05d}-of-00646.tar"
                with open(p, "wb") as f:
                    f.write(b"x" * (1024 * 1024 + 10))  # >1MB
                cache.cached_shards[i] = p
                cache._enforce_bound()

            # Cache size must be <= 3
            self.assertLessEqual(len(cache.cached_shards), 3)
            cache.cleanup()


class TestStreamingDatasetMock(unittest.TestCase):
    def test_mock_tar_streaming_and_collate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "train-00000-of-00646.tar"
            with tarfile.open(tar_path, "w") as tar:
                # Create 3 synthetic samples: 2 training, 1 benchmark val
                samples = [
                    ("test_clip_train1", "0000", "car", False),
                    ("test_clip_train2", "0001", "person", False),
                    ("001c0d67-590e-479b-86b9-c521fe884139", "0000", "bus", True),  # In val clips
                ]
                for clip_id, frame_id, cat, _ in samples:
                    stem = f"{clip_id}_{frame_id}"
                    # Create dummy JPEG
                    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
                    img_bytes_io = io.BytesIO()
                    img.save(img_bytes_io, format="JPEG")
                    img_bytes = img_bytes_io.getvalue()

                    ti = tarfile.TarInfo(name=f"{stem}.jpg")
                    ti.size = len(img_bytes)
                    tar.addfile(ti, io.BytesIO(img_bytes))

                    # Create dummy JSON
                    ann = {"labels": [{"category": cat, "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}]}
                    ann_bytes = json.dumps(ann).encode("utf-8")
                    tj = tarfile.TarInfo(name=f"{stem}.json")
                    tj.size = len(ann_bytes)
                    tar.addfile(tj, io.BytesIO(ann_bytes))

            # Instantiate Streaming Dataset with mock cache
            cache = BoundedShardCache(cache_dir=tmpdir, max_shards=2)
            cache.cached_shards[0] = tar_path

            dataset = StreamingIndianRoadDataset(
                shard_ids=[0],
                total_shards=1,
                cache=cache,
                exclude_benchmark_val=True,
                shuffle=False,
            )

            results = list(dataset)
            # The benchmark validation clip must be excluded, leaving 2 samples
            self.assertEqual(len(results), 2)
            self.assertEqual(dataset.samples_excluded_val, 1)

            # Test Collation
            images, targets = collate_streaming_batch(results)
            self.assertEqual(images.shape, (2, 3, 640, 640))
            self.assertEqual(targets.shape[1], 6)  # [batch_idx, class_id, cx, cy, w, h]
            self.assertEqual(targets.shape[0], 2)

            cache.cleanup()


if __name__ == "__main__":
    unittest.main()
