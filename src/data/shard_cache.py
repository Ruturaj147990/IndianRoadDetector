"""
Bounded Local Shard Cache and Asynchronous Prefetcher for WebDataset TAR Shards.

Features:
- Bounded on-disk footprint (default max 10 shards ~2.8 GB, configurable).
- Background threaded prefetching of upcoming shards to hide network latency.
- Automatic LRU/FIFO eviction of consumed shards to prevent disk overflow.
- Atomic partial-download writes (.tmp -> .tar) with integrity checks.
- Robust retry with exponential backoff on HTTP/network errors.
"""

from collections import OrderedDict
import concurrent.futures
import os
from pathlib import Path
import shutil
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Union
import urllib.request
import requests


class BoundedShardCache:
    """
    Manages a bounded on-disk cache of WebDataset TAR shards with
    background prefetching and automatic eviction.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache_shards",
        max_shards: int = 10,
        prefetch_ahead: int = 2,
        base_url_template: str = "https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset/resolve/main/data/train-{shard_id:05d}-of-00646.tar",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_shards = max(2, max_shards)
        self.prefetch_ahead = max(1, prefetch_ahead)
        self.base_url_template = base_url_template

        # Thread synchronization
        self.lock = threading.RLock()
        # Track completed shards and their access order: shard_id -> local_path
        self.cached_shards: OrderedDict[int, Path] = OrderedDict()
        # Track in-flight downloads: shard_id -> Future
        self.downloading: Dict[int, concurrent.futures.Future] = {}
        # Thread pool for background downloads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="ShardPrefetcher")

        # Scan existing valid shards in cache_dir
        self._scan_existing_cache()

    def _scan_existing_cache(self) -> None:
        with self.lock:
            tar_files = sorted(self.cache_dir.glob("*.tar"))
            for tf in tar_files:
                # Expecting format: train-00000-of-00646.tar
                stem = tf.stem
                parts = stem.split("-")
                if len(parts) >= 2 and parts[1].isdigit():
                    shard_id = int(parts[1])
                    if tf.stat().st_size > 1024 * 1024:  # At least 1MB to be considered non-empty
                        self.cached_shards[shard_id] = tf
            self._enforce_bound()

    def _shard_filename(self, shard_id: int) -> str:
        return f"train-{shard_id:05d}-of-00646.tar"

    def _shard_url(self, shard_id: int) -> str:
        return self.base_url_template.format(shard_id=shard_id)

    def _download_shard(self, shard_id: int) -> Path:
        """Synchronously downloads a single shard to cache atomically."""
        filename = self._shard_filename(shard_id)
        target_path = self.cache_dir / filename
        temp_path = self.cache_dir / f"{filename}.tmp_{os.getpid()}_{threading.get_ident()}"
        url = self._shard_url(shard_id)

        # Check if already present and valid
        if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
            with self.lock:
                self.cached_shards[shard_id] = target_path
                self.cached_shards.move_to_end(shard_id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache HIT] Shard {shard_id:05d} already present ({target_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)
            return target_path

        max_retries = 15
        backoff = 3.0
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ShardCache] Downloading shard {shard_id:05d}/00645 from Hugging Face Hub (attempt {attempt}/{max_retries})...", flush=True)
                
                # Stream download with requests
                with requests.get(url, stream=True, timeout=(15, 90)) as resp:
                    resp.raise_for_status()
                    total_bytes = 0
                    with open(temp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 512):
                            if chunk:
                                f.write(chunk)
                                total_bytes += len(chunk)

                # Rename atomically
                if temp_path.exists():
                    try:
                        if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
                            temp_path.unlink()
                        else:
                            if target_path.exists():
                                target_path.unlink()
                            temp_path.rename(target_path)
                    except OSError:
                        pass

                elapsed = max(time.perf_counter() - t0, 0.001)
                size_mb = total_bytes / (1024 * 1024)
                mb_per_sec = size_mb / elapsed
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ShardCache] Shard {shard_id:05d} cached: {size_mb:.1f} MB in {elapsed:.1f}s ({mb_per_sec:.1f} MB/s)", flush=True)

                with self.lock:
                    self.cached_shards[shard_id] = target_path
                    self.cached_shards.move_to_end(shard_id)
                    self._enforce_bound()

                return target_path

            except Exception as e:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                wait_sec = min(backoff, 30.0)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ShardCache WARNING] Shard {shard_id:05d} download failed ({e}). Retrying in {wait_sec:.1f}s (attempt {attempt}/{max_retries})...", flush=True)
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download shard {shard_id} after {max_retries} attempts: {e}")
                time.sleep(wait_sec)
                backoff = min(backoff * 1.5, 30.0)

        raise RuntimeError(f"Unexpected download exit for shard {shard_id}")

    def _enforce_bound(self) -> None:
        """Evict least-recently-used shards until cache size <= max_shards."""
        # Must be called while holding self.lock
        while len(self.cached_shards) > self.max_shards:
            oldest_id, oldest_path = self.cached_shards.popitem(last=False)
            try:
                if oldest_path.exists():
                    oldest_path.unlink()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ShardCache LRU] Evicted shard {oldest_id:05d} (maintaining limit <= {self.max_shards} shards)", flush=True)
            except OSError:
                pass

    def prefetch(self, shard_ids: List[int]) -> None:
        """Triggers asynchronous download of upcoming shards in background."""
        with self.lock:
            for s_id in shard_ids:
                if s_id in self.cached_shards or s_id in self.downloading:
                    continue
                fut = self.executor.submit(self._download_shard, s_id)
                self.downloading[s_id] = fut

    def get_shard(self, shard_id: int, total_shards: int = 646, auto_prefetch: bool = True) -> Path:
        """
        Retrieves a local path for shard_id, blocking until available.
        Also optionally automatically pre-fetches the next prefetch_ahead shards.
        """
        # Prefetch upcoming shards if requested
        if auto_prefetch:
            upcoming = [
                (shard_id + i) % total_shards
                for i in range(1, self.prefetch_ahead + 1)
            ]
            self.prefetch(upcoming)

        # Check if already cached
        with self.lock:
            if shard_id in self.cached_shards and self.cached_shards[shard_id].exists():
                self.cached_shards.move_to_end(shard_id)
                return self.cached_shards[shard_id]
            fut = self.downloading.get(shard_id)

        # If already downloading, wait for it
        if fut is not None:
            res = fut.result()
            with self.lock:
                self.downloading.pop(shard_id, None)
            return res

        # Otherwise download now
        return self._download_shard(shard_id)

    def evict(self, shard_id: int) -> None:
        """Explicitly evict a completed shard from cache to free space immediately."""
        with self.lock:
            path = self.cached_shards.pop(shard_id, None)
            if path and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def cleanup(self) -> None:
        """Shuts down executor and removes temporary files."""
        self.executor.shutdown(wait=False)
        with self.lock:
            for tmp in self.cache_dir.glob("*.tmp*"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
