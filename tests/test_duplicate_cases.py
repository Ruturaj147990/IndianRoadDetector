"""
Unit Test Suite for IRD V1.5 Duplicate Suppression and Class-Aware NMS.

Explicitly validates:
1. 2 boxes, same class, IoU 0.8 -> keep only highest confidence
2. 3 boxes, same class, overlapping in a chain -> greedy sequential suppression
3. Different classes with high IoU -> do NOT suppress automatically
4. Rider + motorcycle overlapping -> preserve both
5. Person + car overlapping -> preserve both
6. Non-overlapping same-class objects -> preserve both
"""

import sys
from pathlib import Path
import torch

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.box_coder import class_aware_nms, pure_pytorch_nms


def test_case_1_two_boxes_same_class_high_iou():
    """2 boxes, same class, IoU 0.8 -> keep only highest confidence."""
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [10.0, 10.0, 50.0, 42.0],
    ])
    scores = torch.tensor([0.92, 0.78])
    classes = torch.tensor([2, 2])  # class 'car'

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50)
    assert keep.tolist() == [0], f"Expected [0], got {keep.tolist()}"

    # Also verify if order is inverted (lower score listed first in tensor)
    scores_inv = torch.tensor([0.78, 0.92])
    keep_inv = class_aware_nms(boxes, scores_inv, classes, iou_threshold=0.50)
    assert keep_inv.tolist() == [1], f"Expected [1], got {keep_inv.tolist()}"


def test_case_2_three_boxes_chain_overlapping_greedy():
    """
    3 boxes, same class, overlapping -> greedy suppression:
    Box A (score 0.95): [0, 0, 100, 100]
    Box B (score 0.80): [25, 0, 125, 100] -> IoU(A, B) = 75/125 = 0.60 > 0.50
    Box C (score 0.60): [50, 0, 150, 100] -> IoU(B, C) = 75/125 = 0.60 > 0.50, but IoU(A, C) = 50/150 = 0.333 <= 0.50
    
    Greedy NMS MUST:
    1. Select Box A (highest score).
    2. Suppress Box B (IoU with A is 0.60 > 0.50).
    3. Retain Box C (IoU with A is 0.333 <= 0.50; B was suppressed, so cannot suppress C).
    Final kept: [A, C] -> indices [0, 2].
    """
    boxes = torch.tensor([
        [0.0, 0.0, 100.0, 100.0],    # A
        [25.0, 0.0, 125.0, 100.0],   # B
        [50.0, 0.0, 150.0, 100.0],   # C
    ])
    scores = torch.tensor([0.95, 0.80, 0.60])
    classes = torch.tensor([0, 0, 0])

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50)
    assert set(keep.tolist()) == {0, 2}, f"Greedy NMS failed: expected [0, 2], got {keep.tolist()}"


def test_case_3_different_classes_high_iou():
    """Different classes with high IoU -> do NOT suppress automatically."""
    boxes = torch.tensor([
        [50.0, 50.0, 150.0, 150.0],
        [50.0, 50.0, 150.0, 150.0],
    ])
    scores = torch.tensor([0.95, 0.90])
    classes = torch.tensor([0, 2])  # class 0 and class 2

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50)
    assert len(keep) == 2, f"Different classes suppressed! Expected 2 kept, got {len(keep)}"
    assert set(keep.tolist()) == {0, 1}


def test_case_4_rider_plus_motorcycle_overlapping():
    """Rider (class 1) + motorcycle (class 5) overlapping -> preserve both."""
    boxes = torch.tensor([
        [100.0, 80.0, 160.0, 220.0],   # rider
        [90.0, 120.0, 170.0, 240.0],   # motorcycle
    ])
    scores = torch.tensor([0.88, 0.91])
    classes = torch.tensor([1, 5])  # 1 = rider, 5 = motorcycle

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50)
    assert len(keep) == 2, f"Rider and motorcycle falsely suppressed! Got {keep.tolist()}"
    assert set(keep.tolist()) == {0, 1}


def test_case_5_person_plus_car_overlapping():
    """Person (class 0) + car (class 2) overlapping -> preserve both."""
    boxes = torch.tensor([
        [200.0, 150.0, 260.0, 320.0],  # person
        [180.0, 120.0, 380.0, 340.0],  # car
    ])
    scores = torch.tensor([0.82, 0.94])
    classes = torch.tensor([0, 2])  # 0 = person, 2 = car

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50)
    assert len(keep) == 2, f"Person and car falsely suppressed! Got {keep.tolist()}"
    assert set(keep.tolist()) == {0, 1}


def test_case_6_non_overlapping_same_class():
    """Non-overlapping same-class objects -> preserve both."""
    boxes = torch.tensor([
        [10.0, 10.0, 60.0, 60.0],     # car 1 (top-left)
        [400.0, 300.0, 500.0, 420.0],  # car 2 (bottom-right)
    ])
    scores = torch.tensor([0.89, 0.85])
    classes = torch.tensor([2, 2])  # both cars

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50)
    assert len(keep) == 2, f"Non-overlapping same-class objects suppressed! Got {keep.tolist()}"
    assert set(keep.tolist()) == {0, 1}


if __name__ == "__main__":
    test_case_1_two_boxes_same_class_high_iou()
    print("Test Case 1 (2 boxes same class high IoU): PASSED")
    test_case_2_three_boxes_chain_overlapping_greedy()
    print("Test Case 2 (3 boxes chain greedy suppression): PASSED")
    test_case_3_different_classes_high_iou()
    print("Test Case 3 (different classes high IoU): PASSED")
    test_case_4_rider_plus_motorcycle_overlapping()
    print("Test Case 4 (rider + motorcycle co-occurrence): PASSED")
    test_case_5_person_plus_car_overlapping()
    print("Test Case 5 (person + car co-occurrence): PASSED")
    test_case_6_non_overlapping_same_class()
    print("Test Case 6 (non-overlapping same-class objects): PASSED")
    print("\nALL 6 CANONICAL DUPLICATE SUPPRESSION TESTS PASSED SUCCESSFULLY!")
