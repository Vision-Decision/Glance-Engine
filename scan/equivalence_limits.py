"""Frozen, reference-only BF16 spreads across the five batch/padding layouts.

Reproduce on engopt with scan.test_scan_equivalence --real --measure-reference FILE
and the v1 checkpoint. Each acceptance limit is twice its reference maximum.
Source: outputs/engine_opt/09_numerics/raw/item9_strict_replay.json.
"""

# Raw public outputs include the discontinuity from changing the winning patch.
BF16_REFERENCE_SPREAD = {
    "presence_prob": 0.01585182547569275,
    "iou_prob": 0.20370902121067047,
    "box_pred": 0.21247240900993347,
    "typed_prob": 0.01167655736207962,
}
BF16_LIMIT = {k: 2 * v for k, v in BF16_REFERENCE_SPREAD.items()}

# These additional gates apply to comparisons exposing center indices. The patch
# gates evaluate the independent reference edge/IoU heads at the engine's chosen
# index, including on a switch. Regret measures its score deficit on the reference.
CENTER_REFERENCE_SPREAD = {
    "same_center_box": 0.018160194158554077,
    "same_center_iou": 0.024312853813171387,
    "patch_box": 0.046973466873168945,
    "patch_iou": 0.03391812741756439,
    "center_regret": 0.4870161392572623,
}
CENTER_LIMIT = {k: 2 * v for k, v in CENTER_REFERENCE_SPREAD.items()}

# All 60 mixed-prefix pairwise centers matched, so this fixture also checks exact selection.
MIXED_BOX_LIMIT = {
    "box_pred": 2 * 0.0033432841300964355,
    "iou_prob": 2 * 0.006695806980133057,
    "center_regret": 0.0,
}
