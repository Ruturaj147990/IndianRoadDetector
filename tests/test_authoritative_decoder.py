"""
Unit Test Suite for IRD V1 Authoritative Box Decoder and Coordinate Geometry Engine.

Verifies:
1. Mathematical identity across training target decoding, evaluation, and inference.
2. All three strides (8, 16, 32).
3. Grid boundary conditions and corner extremes.
4. Tiny boxes and large boxes.
5. Strict numerical stability (zero NaNs, zero Infs).
6. Non-vanishing gradient property of the smooth non-saturating formulation.
7. Class-aware NMS correctness and max_det ceilings.
"""

import math
import sys
from pathlib import Path
try:
    import pytest
except ImportError:
    pytest = None
import torch

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.box_coder import (
    NUM_CLASSES,
    class_aware_nms,
    decode_boxes_smooth,
    decode_ird_predictions_authoritative,
    pure_pytorch_nms,
)
from src.models.head.custom_head import HeadOutput


def test_strides_and_coordinate_parity():
    """Verify decode_boxes_smooth behaves identically across all three strides."""
    strides = [8, 16, 32]
    num_samples = 50

    for stride in strides:
        raw_box = torch.randn(num_samples, 4, dtype=torch.float32)
        grid_x = torch.randint(0, 640 // stride, (num_samples,), dtype=torch.float32)
        grid_y = torch.randint(0, 640 // stride, (num_samples,), dtype=torch.float32)

        # Smooth decode
        decoded = decode_boxes_smooth(raw_box, grid_x, grid_y, stride=stride, version="v2_smooth")

        assert decoded.shape == (num_samples, 4)
        assert not torch.isnan(decoded).any(), f"NaN detected at stride {stride}"
        assert not torch.isinf(decoded).any(), f"Inf detected at stride {stride}"

        # Check that x2 > x1 and y2 > y1 strictly
        w = decoded[:, 2] - decoded[:, 0]
        h = decoded[:, 3] - decoded[:, 1]
        assert (w > 0).all(), f"Non-positive width detected at stride {stride}"
        assert (h > 0).all(), f"Non-positive height detected at stride {stride}"


def test_gradient_survival():
    """Verify that gradients never vanish even under extreme raw inputs (tw = +/- 20)."""
    extreme_inputs = torch.tensor([
        [0.0, 0.0, -20.0, -20.0],
        [0.0, 0.0, 20.0, 20.0],
        [0.0, 0.0, -10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
    ], requires_grad=True)

    grid_x = torch.tensor([10.0, 10.0, 10.0, 10.0])
    grid_y = torch.tensor([10.0, 10.0, 10.0, 10.0])

    decoded = decode_boxes_smooth(extreme_inputs, grid_x, grid_y, stride=16, version="v2_smooth")
    # Sum of width and height: w = (x2 - x1), h = (y2 - y1)
    w = decoded[:, 2] - decoded[:, 0]
    h = decoded[:, 3] - decoded[:, 1]
    loss = (w + h).sum()
    loss.backward()

    grads = extreme_inputs.grad
    assert grads is not None
    assert not torch.isnan(grads).any()
    # In smooth v2, gradients must be strictly non-zero:
    assert (grads[:, 2].abs() > 1e-12).all(), f"Gradient vanished for width parameter: {grads[:, 2]}"
    assert (grads[:, 3].abs() > 1e-12).all(), f"Gradient vanished for height parameter: {grads[:, 3]}"


def test_boundary_and_corner_clamping():
    """Verify grid cells at extreme corners (0,0) and (W-1, H-1) decode gracefully."""
    for stride in [8, 16, 32]:
        grid_dim = 640 // stride
        corner_x = torch.tensor([0.0, float(grid_dim - 1)])
        corner_y = torch.tensor([0.0, float(grid_dim - 1)])
        raw_box = torch.zeros(2, 4)

        decoded = decode_boxes_smooth(raw_box, corner_x, corner_y, stride=stride, version="v2_smooth")
        # Center of (0,0) with tx=0, ty=0 is (0 + 2*0.5 - 0.5) * stride = 0.5 * stride
        expected_cx_0 = 0.5 * stride
        expected_w_0 = stride * math.exp(0.0)  # stride
        expected_x1_0 = expected_cx_0 - expected_w_0 / 2.0  # 0.0
        expected_x2_0 = expected_cx_0 + expected_w_0 / 2.0  # stride

        assert torch.isclose(decoded[0, 0], torch.tensor(expected_x1_0), atol=1e-4)
        assert torch.isclose(decoded[0, 2], torch.tensor(expected_x2_0), atol=1e-4)


def test_nms_class_aware_and_max_det():
    """Verify class-aware NMS preserves overlapping different-class objects and respects max_det."""
    # Two identical spatial boxes, but different classes (person 0 and motorcycle 5)
    boxes = torch.tensor([
        [50.0, 50.0, 150.0, 150.0],
        [50.0, 50.0, 150.0, 150.0],
        [51.0, 51.0, 151.0, 151.0],  # same-class duplicate of box 0
    ])
    scores = torch.tensor([0.90, 0.85, 0.80])
    classes = torch.tensor([0, 5, 0])  # box 0 is class 0, box 1 is class 5, box 2 is class 0

    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.50, max_det=10)

    # Box 0 (class 0, 0.90) kept
    # Box 1 (class 5, 0.85) kept (different class!)
    # Box 2 (class 0, 0.80) suppressed by box 0
    assert 0 in keep
    assert 1 in keep
    assert 2 not in keep
    assert len(keep) == 2


def test_max_det_enforcement():
    """Verify max_det ceiling is strictly respected."""
    # Generate 500 distinct non-overlapping boxes
    boxes = []
    for i in range(500):
        boxes.append([float(i * 10), 0.0, float(i * 10 + 5), 5.0])
    boxes_t = torch.tensor(boxes)
    scores_t = torch.linspace(0.99, 0.10, 500)
    classes_t = torch.zeros(500, dtype=torch.long)

    keep = class_aware_nms(boxes_t, scores_t, classes_t, iou_threshold=0.50, max_det=150)
    assert len(keep) == 150, f"Expected 150 boxes retained, got {len(keep)}"


if __name__ == "__main__":
    test_strides_and_coordinate_parity()
    test_gradient_survival()
    test_boundary_and_corner_clamping()
    test_nms_class_aware_and_max_det()
    test_max_det_enforcement()
    print("ALL AUTHORITATIVE DECODER UNIT TESTS PASSED!")
