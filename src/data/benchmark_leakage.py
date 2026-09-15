"""
Benchmark Data Leakage Protection for Indian Road Dataset.

Guarantees 100% isolation of the authoritative 1,719-image benchmark validation set
(25 distinct video clips) from the full 646k dataset training stream.
"""

from pathlib import Path
from typing import Set, Tuple, Union

# Exact 25 validation clip IDs from the corrected benchmark split
# (data/indian_road_yolo/images/val) totaling 1,719 images.
BENCHMARK_VAL_CLIPS: Set[str] = {
    "001c0d67-590e-479b-86b9-c521fe884139",
    "00263864-d94b-44c5-8eb9-4e51093e59f4",
    "007e2cdb-abe0-4fe7-89ea-79569159ad14",
    "008062ff-e7f8-458b-80bd-1f909551011c",
    "0088ab0e-9098-4ec5-8f36-a47aef7ae54e",
    "00ca74e0-be2b-4b3e-ab4b-46da5a9f4279",
    "00cca8c2-9bcf-4c11-a245-fdc3d5ee2769",
    "01143ada-8e82-4812-bfc6-63842c9f4dc6",
    "01977365-1007-4f84-88fd-eb6ca0326ac4",
    "01d46ccf-1cc7-4dc9-9cd7-f8ebc85a8a0c",
    "020e23ed-7195-47c3-85e7-123772977eb0",
    "026aa3bc-d1e9-483c-afa4-d4a3205b483d",
    "026f4929-e252-4991-b366-b747276680cc",
    "028a91c7-17a1-4d98-b896-82d333bd12f5",
    "02b3d13a-28b3-4f79-84d2-9b92ff674a3e",
    "02bf8b67-bb97-4b95-b74f-83ccf880f34d",
    "02e2f022-96ab-4542-ae69-10068aae2e0c",
    "03102699-82f7-4cb7-af89-7df02e156413",
    "0332b7aa-2d97-42ac-8326-56f860abd2a6",
    "03407111-1fd9-4df1-ac6b-333a1e0c74ca",
    "03aca383-acd4-4381-b617-67b47cbe9ebb",
    "03dd2923-8537-465d-8bf9-066ff76a5d65",
    "03e1f4be-a8fe-4d93-b7e8-6d762b6ceeb6",
    "0419ccf8-54ad-48be-8baa-0899ab41a505",
    "041c376d-482b-4a71-8ea9-3dc1bd3a393b",
}


def extract_clip_id(identifier: str) -> str:
    """
    Extracts the clip UUID from a filename, sample name, or path.
    Supports formats:
      - '001c0d67-590e-479b-86b9-c521fe884139__0000.jpg'
      - '001c0d67-590e-479b-86b9-c521fe884139_0000.jpg'
      - '001c0d67-590e-479b-86b9-c521fe884139/0000.jpg'
      - '001c0d67-590e-479b-86b9-c521fe884139'
    """
    # Strip path separators
    name = Path(identifier).name
    # Strip extension
    stem = Path(name).stem

    if "/" in identifier or "\\" in identifier:
        parts = identifier.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[-2]:
            return parts[-2]

    if "__" in stem:
        return stem.split("__")[0]
    elif "_" in stem:
        # UUID is 36 characters with 4 hyphens
        parts = stem.rsplit("_", 1)
        if len(parts[0]) == 36 and parts[0].count("-") == 4:
            return parts[0]
        return stem.split("_")[0]
    return stem


def is_benchmark_validation_sample(identifier: str) -> bool:
    """
    Returns True if the given sample belongs to any of the 25 benchmark
    validation clips and must be excluded from training.
    """
    clip_id = extract_clip_id(identifier)
    return clip_id in BENCHMARK_VAL_CLIPS
