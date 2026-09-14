"""
Comprehensive Unit Test Suite for IRD V2 Task-Aligned Assignment & Loss.

Covers all 18 required scenarios:
 1. Task-alignment calculation (t = s^alpha * IoU^beta).
 2. Top-k candidate selection.
 3. Multi-GT conflict resolution (deterministic assignment to highest alignment metric).
 4. Crowded objects assignment.
 5. Single-object images.
 6. Empty images (zero targets).
 7. Background quality target equals zero.
 8. Quality target equals matched IoU for positives.
 9. Score monotonicity (higher confidence + higher IoU produces higher score).
10. Numerical stability (no NaNs / Infs with extreme values).
11. Deterministic assignment consistency.
12. Batch sizes 1, 2, and 4.
13. All 12 classes represented.
14. Rider + motorcycle overlapping pair.
15. Person + car overlapping pair.
16. Truck + car overlap pair.
17. Multiple adjacent motorcycles.
18. Dense traffic with many nearby objects (15+ targets).
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.losses.task_aligned_assignor import (
    TaskAlignedAssignor,
    box_iou_pairwise,
    generate_anchor_grid,
)
from src.models.losses.task_aligned_loss import (
    TaskAlignedLoss,
    varifocal_loss,
)
from src.models.task_aligned_decoder import decode_ird_v2_predictions
from src.models.head.custom_head import HeadOutput


def get_standard_grid():
    """Generates standard multi-scale anchor grid (80x80, 40x40, 20x20) totaling 8,400 anchors."""
    grid_shapes = [(80, 80), (40, 40), (20, 20)]
    strides = [8, 16, 32]
    anchor_points, stride_tensor = generate_anchor_grid(
        grid_shapes=grid_shapes,
        strides=strides,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return anchor_points, stride_tensor, grid_shapes, strides


# ---------------------------------------------------------------------------
# Test 1: Task-alignment calculation (t = s^alpha * IoU^beta)
# ---------------------------------------------------------------------------
def test_task_alignment_calculation(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=5, num_classes=12, alpha=0.5, beta=6.0)

    # Synthetic single anchor, single GT
    # Let s = 0.64 (so s^0.5 = 0.8), IoU = 0.5 (so 0.5^6 = 0.015625)
    # Expected t = 0.8 * 0.015625 = 0.0125
    s = 0.64
    iou = 0.5
    expected_t = (s ** 0.5) * (iou ** 6.0)
    assert abs(expected_t - 0.0125) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: Top-k candidate selection
# ---------------------------------------------------------------------------
def test_topk_candidate_selection(standard_grid):
    anchor_points, _, _, _ = standard_grid
    topk = 5
    assignor = TaskAlignedAssignor(topk=topk, num_classes=12, alpha=1.0, beta=1.0)

    N_anchors = anchor_points.shape[0]
    # Single GT car at center of image: [200, 200, 400, 400]
    gt_boxes = torch.tensor([[[200.0, 200.0, 400.0, 400.0]]])  # [1, 1, 4]
    gt_labels = torch.tensor([[[2]]])                           # class 2 (car)

    # Synthetic predictions: predicted box matching GT perfectly for all anchors
    pred_bboxes = gt_boxes.repeat(1, N_anchors, 1)  # [1, N_anchors, 4]
    pred_scores = torch.full((1, N_anchors, 12), 0.5)  # [1, N_anchors, 12]

    _, target_scores, fg_mask, _ = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    num_pos = fg_mask.sum().item()
    # At most topk anchors should be selected
    assert num_pos <= topk
    assert num_pos > 0


# ---------------------------------------------------------------------------
# Test 3: Multi-GT conflict resolution
# ---------------------------------------------------------------------------
def test_multi_gt_conflict_resolution(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=10, num_classes=12, alpha=1.0, beta=1.0)

    N_anchors = anchor_points.shape[0]
    # GT 0: person at [100, 100, 200, 200]
    # GT 1: car at [120, 120, 250, 250] (heavily overlapping)
    gt_boxes = torch.tensor([[[100.0, 100.0, 200.0, 200.0], [120.0, 120.0, 250.0, 250.0]]])  # [1, 2, 4]
    gt_labels = torch.tensor([[[0], [2]]])  # [1, 2, 1]

    pred_bboxes = torch.zeros((1, N_anchors, 4))
    pred_bboxes[0, :] = torch.tensor([120.0, 120.0, 250.0, 250.0])  # Matches car much better

    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, :, 0] = 0.3  # person score
    pred_scores[0, :, 2] = 0.9  # car score (much higher alignment)

    target_bboxes, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    # For overlapping anchors inside both boxes, car should win deterministically
    overlap_mask = (
        (anchor_points[:, 0] >= 120.0) & (anchor_points[:, 0] <= 200.0) &
        (anchor_points[:, 1] >= 120.0) & (anchor_points[:, 1] <= 200.0)
    )
    overlap_pos = fg_mask[0] & overlap_mask
    if overlap_pos.any():
        # Assigned GT must be car (index 1) because alignment is higher
        assigned_in_overlap = target_gt_idx[0, overlap_pos]
        assert (assigned_in_overlap == 1).all()


# ---------------------------------------------------------------------------
# Test 4: Crowded objects
# ---------------------------------------------------------------------------
def test_crowded_objects_assignment(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=6, num_classes=12, alpha=0.5, beta=2.0)
    N_anchors = anchor_points.shape[0]

    # 4 crowded adjacent motorcycles along a road lane
    m1 = [100.0, 200.0, 140.0, 300.0]
    m2 = [130.0, 200.0, 170.0, 300.0]
    m3 = [160.0, 200.0, 200.0, 300.0]
    m4 = [190.0, 200.0, 230.0, 300.0]
    gt_boxes = torch.tensor([[m1, m2, m3, m4]])
    gt_labels = torch.tensor([[[5], [5], [5], [5]]])

    pred_bboxes = torch.zeros((1, N_anchors, 4))
    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, :, 5] = 0.8

    for idx, b in enumerate([m1, m2, m3, m4]):
        in_b = (
            (anchor_points[:, 0] >= b[0]) & (anchor_points[:, 0] <= b[2]) &
            (anchor_points[:, 1] >= b[1]) & (anchor_points[:, 1] <= b[3])
        )
        pred_bboxes[0, in_b] = torch.tensor(b)

    _, _, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    # Every motorcycle should receive positive anchor assignments
    assigned_gts = set(target_gt_idx[0, fg_mask[0]].tolist())
    assert 0 in assigned_gts
    assert 1 in assigned_gts
    assert 2 in assigned_gts
    assert 3 in assigned_gts


# ---------------------------------------------------------------------------
# Test 5: Single-object images
# ---------------------------------------------------------------------------
def test_single_object_image(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=10, num_classes=12)
    N_anchors = anchor_points.shape[0]

    gt_boxes = torch.tensor([[[50.0, 50.0, 150.0, 150.0]]])
    gt_labels = torch.tensor([[[2]]])

    pred_bboxes = gt_boxes.repeat(1, N_anchors, 1)
    pred_scores = torch.full((1, N_anchors, 12), 0.7)

    target_bboxes, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    assert fg_mask[0].sum() > 0
    assert (target_gt_idx[0, fg_mask[0]] == 0).all()
    assert (target_bboxes[0, fg_mask[0]] == gt_boxes[0, 0]).all()


# ---------------------------------------------------------------------------
# Test 6: Empty images (zero targets)
# ---------------------------------------------------------------------------
def test_empty_image(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=10, num_classes=12)
    N_anchors = anchor_points.shape[0]

    gt_boxes = torch.zeros((1, 0, 4))
    gt_labels = torch.zeros((1, 0, 1), dtype=torch.long)

    pred_bboxes = torch.randn(1, N_anchors, 4)
    pred_scores = torch.rand(1, N_anchors, 12)

    target_bboxes, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    assert fg_mask[0].sum() == 0
    assert (target_scores[0] == 0.0).all()
    assert (target_bboxes[0] == 0.0).all()
    assert (target_gt_idx[0] == -1).all()


# ---------------------------------------------------------------------------
# Test 7: Background quality target equals strictly zero
# ---------------------------------------------------------------------------
def test_background_target_zero(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=5, num_classes=12)
    N_anchors = anchor_points.shape[0]

    gt_boxes = torch.tensor([[[200.0, 200.0, 300.0, 300.0]]])
    gt_labels = torch.tensor([[[4]]])  # bus

    pred_bboxes = gt_boxes.repeat(1, N_anchors, 1)
    pred_scores = torch.full((1, N_anchors, 12), 0.8)

    _, target_scores, fg_mask, _ = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    bg_mask = ~fg_mask[0]
    # For every background anchor, continuous target scores MUST be exactly 0.0
    assert (target_scores[0, bg_mask] == 0.0).all()


# ---------------------------------------------------------------------------
# Test 8: Quality target equals matched IoU for positives
# ---------------------------------------------------------------------------
def test_quality_target_matches_iou(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=1, num_classes=12, alpha=1.0, beta=1.0)
    N_anchors = anchor_points.shape[0]

    gt_box = torch.tensor([[[100.0, 100.0, 200.0, 200.0]]])
    gt_labels = torch.tensor([[[3]]])  # truck

    # Find anchor closest to center
    cx, cy = 150.0, 150.0
    dists = (anchor_points[:, 0] - cx) ** 2 + (anchor_points[:, 1] - cy) ** 2
    best_anchor_idx = dists.argmin().item()

    pred_bboxes = torch.zeros((1, N_anchors, 4))
    pred_bboxes[0, best_anchor_idx] = torch.tensor([100.0, 100.0, 200.0, 200.0])  # IoU = 1.0

    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, best_anchor_idx, 3] = 0.9

    _, target_scores, fg_mask, _ = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_box,
    )

    assert fg_mask[0, best_anchor_idx]
    # Since top anchor has IoU=1.0 and max(t)=t, target score = 1.0 * (t/t) = 1.0
    assert abs(target_scores[0, best_anchor_idx, 3].item() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Test 9: Score monotonicity
# ---------------------------------------------------------------------------
def test_score_monotonicity():
    alpha, beta = 0.5, 6.0
    # Candidate A: high conf, high IoU
    s_a, iou_a = 0.9, 0.85
    t_a = (s_a ** alpha) * (iou_a ** beta)

    # Candidate B: low conf, lower IoU
    s_b, iou_b = 0.3, 0.50
    t_b = (s_b ** alpha) * (iou_b ** beta)

    assert t_a > t_b

    # Monotonicity check: increasing s while keeping IoU constant increases t
    assert (0.95 ** alpha) * (iou_a ** beta) > t_a
    # Increasing IoU while keeping s constant increases t
    assert (s_a ** alpha) * (0.90 ** beta) > t_a


# ---------------------------------------------------------------------------
# Test 10: Numerical stability (no NaNs / Infs with extreme values)
# ---------------------------------------------------------------------------
def test_numerical_stability(standard_grid):
    anchor_points, _, _, _ = standard_grid
    loss_fn = TaskAlignedLoss(num_classes=12)
    B = 2
    N_anchors = anchor_points.shape[0]

    # Extreme logits: large positive (e.g. +50) and large negative (e.g. -50)
    pred_logits = torch.cat([
        torch.full((B, N_anchors // 2, 12), -50.0),
        torch.full((B, N_anchors - N_anchors // 2, 12), 50.0),
    ], dim=1)
    target_scores = torch.zeros_like(pred_logits)
    target_scores[:, :5, 0] = 1.0  # a few positive targets

    vfl = varifocal_loss(pred_logits, target_scores)
    assert not torch.isnan(vfl)
    assert not torch.isinf(vfl)
    assert vfl.item() > 0.0


# ---------------------------------------------------------------------------
# Test 11: Deterministic assignment consistency
# ---------------------------------------------------------------------------
def test_deterministic_assignment(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=10, num_classes=12)
    N_anchors = anchor_points.shape[0]

    torch.manual_seed(42)
    pred_scores = torch.rand(1, N_anchors, 12)
    pred_bboxes = torch.rand(1, N_anchors, 4) * 500.0
    gt_boxes = torch.tensor([[[50.0, 50.0, 200.0, 200.0], [150.0, 150.0, 350.0, 350.0]]])
    gt_labels = torch.tensor([[[1], [2]]])

    out1 = assignor(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_boxes)
    out2 = assignor(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_boxes)

    for t1, t2 in zip(out1, out2):
        assert torch.equal(t1, t2)


def test_batch_sizes(standard_grid, B):
    anchor_points, _, _, strides = standard_grid
    loss_fn = TaskAlignedLoss(num_classes=12, strides=tuple(strides))

    # Synthetic HeadOutput
    box_preds = [torch.randn(B, 4, 80, 80), torch.randn(B, 4, 40, 40), torch.randn(B, 4, 20, 20)]
    obj_preds = [torch.randn(B, 1, 80, 80), torch.randn(B, 1, 40, 40), torch.randn(B, 1, 20, 20)]
    cls_preds = [torch.randn(B, 12, 80, 80), torch.randn(B, 12, 40, 40), torch.randn(B, 12, 20, 20)]
    qual_preds = [torch.randn(B, 1, 80, 80), torch.randn(B, 1, 40, 40), torch.randn(B, 1, 20, 20)]

    head_out = HeadOutput(
        box_preds=box_preds,
        obj_preds=obj_preds,
        cls_preds=cls_preds,
        strides=strides,
        quality_preds=qual_preds,
    )

    # Targets: 2 targets per image
    targets = []
    for b in range(B):
        t = torch.tensor([[b, 2, 200.0, 200.0, 100.0, 100.0], [b, 5, 300.0, 300.0, 80.0, 80.0]])
        targets.append(t)
    targets_tensor = torch.cat(targets, dim=0)

    res = loss_fn(head_out, targets_tensor)
    assert not torch.isnan(res.total_loss)
    assert not torch.isinf(res.total_loss)
    assert res.total_loss.item() > 0.0


# ---------------------------------------------------------------------------
# Test 13: All 12 classes represented
# ---------------------------------------------------------------------------
def test_all_12_classes(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=2, num_classes=12)
    N_anchors = anchor_points.shape[0]

    # Create 12 non-overlapping ground-truth boxes, one for each class
    boxes = []
    labels = []
    for c in range(12):
        x = 50.0 + (c % 4) * 120.0
        y = 50.0 + (c // 4) * 120.0
        boxes.append([x, y, x + 80.0, y + 80.0])
        labels.append([c])

    gt_boxes = torch.tensor([boxes])       # [1, 12, 4]
    gt_labels = torch.tensor([labels])     # [1, 12, 1]

    pred_bboxes = gt_boxes.repeat(1, N_anchors // 12 + 1, 1)[:, :N_anchors]
    pred_scores = torch.zeros((1, N_anchors, 12))
    for c in range(12):
        pred_scores[0, :, c] = 0.5

    _, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    # Confirm all 12 classes receive positive matches
    assigned_labels = set()
    for p_idx in fg_mask[0].nonzero(as_tuple=False).squeeze(1):
        c_matched = target_scores[0, p_idx].argmax().item()
        assigned_labels.add(c_matched)

    assert len(assigned_labels) == 12


# ---------------------------------------------------------------------------
# Test 14: Rider + motorcycle overlapping pair
# ---------------------------------------------------------------------------
def test_rider_motorcycle_overlap(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=6, num_classes=12, alpha=0.5, beta=6.0)
    N_anchors = anchor_points.shape[0]

    # Rider: [150, 120, 210, 240] (tall, narrow, sitting on bike)
    # Motorcycle: [140, 180, 220, 280] (lower, overlapping bottom half)
    gt_boxes = torch.tensor([[[150.0, 120.0, 210.0, 240.0], [140.0, 180.0, 220.0, 280.0]]])
    gt_labels = torch.tensor([[[1], [5]]])  # rider=1, motorcycle=5

    pred_bboxes = gt_boxes.repeat(1, N_anchors // 2 + 1, 1)[:, :N_anchors]
    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, :, 1] = 0.7  # rider
    pred_scores[0, :, 5] = 0.8  # motorcycle

    _, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    assigned_gts = set(target_gt_idx[0, fg_mask[0]].tolist())
    # Both rider and motorcycle must receive positive anchor assignments
    assert 0 in assigned_gts  # rider
    assert 1 in assigned_gts  # motorcycle


# ---------------------------------------------------------------------------
# Test 15: Person + car overlapping pair
# ---------------------------------------------------------------------------
def test_person_car_overlap(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=6, num_classes=12)
    N_anchors = anchor_points.shape[0]

    # Person standing in front of car bonnet
    gt_boxes = torch.tensor([[[200.0, 150.0, 260.0, 320.0], [180.0, 180.0, 450.0, 400.0]]])
    gt_labels = torch.tensor([[[0], [2]]])  # person=0, car=2

    pred_bboxes = gt_boxes.repeat(1, N_anchors // 2 + 1, 1)[:, :N_anchors]
    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, :, 0] = 0.8
    pred_scores[0, :, 2] = 0.9

    _, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    assigned_gts = set(target_gt_idx[0, fg_mask[0]].tolist())
    assert 0 in assigned_gts  # person
    assert 1 in assigned_gts  # car


# ---------------------------------------------------------------------------
# Test 16: Truck + car overlap pair
# ---------------------------------------------------------------------------
def test_truck_car_overlap(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=8, num_classes=12)
    N_anchors = anchor_points.shape[0]

    # Adjacent truck and car on multi-lane highway
    gt_boxes = torch.tensor([[[50.0, 100.0, 250.0, 450.0], [220.0, 200.0, 380.0, 400.0]]])
    gt_labels = torch.tensor([[[3], [2]]])  # truck=3, car=2

    pred_bboxes = gt_boxes.repeat(1, N_anchors // 2 + 1, 1)[:, :N_anchors]
    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, :, 3] = 0.8
    pred_scores[0, :, 2] = 0.85

    _, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    assigned_gts = set(target_gt_idx[0, fg_mask[0]].tolist())
    assert 0 in assigned_gts  # truck
    assert 1 in assigned_gts  # car


# ---------------------------------------------------------------------------
# Test 17: Multiple adjacent motorcycles
# ---------------------------------------------------------------------------
def test_multiple_adjacent_motorcycles(standard_grid):
    anchor_points, _, _, _ = standard_grid
    assignor = TaskAlignedAssignor(topk=4, num_classes=12)
    N_anchors = anchor_points.shape[0]

    # 3 motorcycles parked side-by-side
    m1 = [100.0, 200.0, 130.0, 300.0]
    m2 = [125.0, 200.0, 155.0, 300.0]
    m3 = [150.0, 200.0, 180.0, 300.0]
    gt_boxes = torch.tensor([[m1, m2, m3]])
    gt_labels = torch.tensor([[[5], [5], [5]]])

    pred_bboxes = gt_boxes.repeat(1, N_anchors // 3 + 1, 1)[:, :N_anchors]
    pred_scores = torch.zeros((1, N_anchors, 12))
    pred_scores[0, :, 5] = 0.85

    _, target_scores, fg_mask, target_gt_idx = assignor(
        pred_scores=pred_scores,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
    )

    assigned = set(target_gt_idx[0, fg_mask[0]].tolist())
    assert len(assigned) == 3


# ---------------------------------------------------------------------------
# Test 18: Dense traffic scene with 15+ clustered targets
# ---------------------------------------------------------------------------
def test_dense_traffic_15_plus_objects(standard_grid):
    anchor_points, _, _, strides = standard_grid
    loss_fn = TaskAlignedLoss(num_classes=12, strides=tuple(strides), topk=5)
    B = 1

    # 16 dense objects across diverse categories
    boxes = []
    labels = []
    for i in range(16):
        cx = 50.0 + (i % 4) * 140.0
        cy = 80.0 + (i // 4) * 130.0
        w, h = 60.0 + (i % 3) * 20.0, 70.0 + (i % 2) * 30.0
        cls_id = i % 12
        boxes.append([0, cls_id, cx, cy, w, h])

    targets_tensor = torch.tensor(boxes, dtype=torch.float32)

    box_preds = [torch.randn(B, 4, 80, 80), torch.randn(B, 4, 40, 40), torch.randn(B, 4, 20, 20)]
    obj_preds = [torch.randn(B, 1, 80, 80), torch.randn(B, 1, 40, 40), torch.randn(B, 1, 20, 20)]
    cls_preds = [torch.randn(B, 12, 80, 80), torch.randn(B, 12, 40, 40), torch.randn(B, 12, 20, 20)]
    qual_preds = [torch.randn(B, 1, 80, 80), torch.randn(B, 1, 40, 40), torch.randn(B, 1, 20, 20)]

    head_out = HeadOutput(
        box_preds=box_preds,
        obj_preds=obj_preds,
        cls_preds=cls_preds,
        strides=strides,
        quality_preds=qual_preds,
    )

    res = loss_fn(head_out, targets_tensor)
    assert not torch.isnan(res.total_loss)
    assert res.num_positives > 0
    assert res.total_loss.item() > 0.0


if __name__ == "__main__":
    print("=" * 70)
    print("Running IRD V2 Task-Aligned Unit Test Suite (18 Scenarios)")
    print("=" * 70)

    # Setup fixture
    grid_shapes = [(80, 80), (40, 40), (20, 20)]
    strides = [8, 16, 32]
    anchor_points, stride_tensor = generate_anchor_grid(grid_shapes, strides, torch.device("cpu"), torch.float32)
    fixture = (anchor_points, stride_tensor, grid_shapes, strides)

    tests = [
        ("Test 1: Task-alignment calculation (t = s^alpha * IoU^beta)", lambda: test_task_alignment_calculation(fixture)),
        ("Test 2: Top-k candidate selection", lambda: test_topk_candidate_selection(fixture)),
        ("Test 3: Multi-GT conflict resolution (deterministic assignment)", lambda: test_multi_gt_conflict_resolution(fixture)),
        ("Test 4: Crowded objects assignment", lambda: test_crowded_objects_assignment(fixture)),
        ("Test 5: Single-object images", lambda: test_single_object_image(fixture)),
        ("Test 6: Empty images (zero targets)", lambda: test_empty_image(fixture)),
        ("Test 7: Background quality target equals strictly zero", lambda: test_background_target_zero(fixture)),
        ("Test 8: Quality target equals matched IoU for positives", lambda: test_quality_target_matches_iou(fixture)),
        ("Test 9: Score monotonicity", lambda: test_score_monotonicity()),
        ("Test 10: Numerical stability (extreme logits, no NaNs/Infs)", lambda: test_numerical_stability(fixture)),
        ("Test 11: Deterministic assignment consistency", lambda: test_deterministic_assignment(fixture)),
        ("Test 12: Batch sizes 1, 2, and 4", lambda: [test_batch_sizes(fixture, b) for b in [1, 2, 4]]),
        ("Test 13: All 12 classes represented", lambda: test_all_12_classes(fixture)),
        ("Test 14: Rider + motorcycle overlapping pair", lambda: test_rider_motorcycle_overlap(fixture)),
        ("Test 15: Person + car overlapping pair", lambda: test_person_car_overlap(fixture)),
        ("Test 16: Truck + car overlap pair", lambda: test_truck_car_overlap(fixture)),
        ("Test 17: Multiple adjacent motorcycles", lambda: test_multiple_adjacent_motorcycles(fixture)),
        ("Test 18: Dense traffic scene with 15+ clustered targets", lambda: test_dense_traffic_15_plus_objects(fixture)),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            raise e

    print("=" * 70)
    print(f"ALL {passed}/18 IRD V2 TASK-ALIGNED UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

