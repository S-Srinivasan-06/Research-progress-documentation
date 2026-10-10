"""Threshold selection helpers for clip-level sensitivity analysis."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import math


def _target_value(target_sensitivity: float) -> float:
    if isinstance(target_sensitivity, (bool, np.bool_)):
        raise ValueError("target sensitivity must be a number in (0, 1]")
    try:
        target = float(target_sensitivity)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target sensitivity must be a number in (0, 1]") from exc
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("target sensitivity must be in (0, 1]")
    return target


def make_threshold_policy(objective: str, target_sensitivity: float = 0.95) -> dict:
    if objective not in ("f1", "sensitivity"):
        raise ValueError("threshold objective must be 'f1' or 'sensitivity'")
    target = _target_value(target_sensitivity)
    return {
        "objective": objective,
        "target_sensitivity": target if objective == "sensitivity" else None,
    }


def _validated_arrays(y, p, target_sensitivity):
    _target_value(target_sensitivity)
    try:
        labels = np.asarray(y, dtype=np.float64)
        scores = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("labels and scores must be numeric arrays") from exc
    if labels.ndim != 1 or scores.ndim != 1 or labels.size == 0 or labels.size != scores.size:
        raise ValueError("labels and scores must be non-empty 1D arrays of equal length")
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only finite binary values 0 and 1")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return labels.astype(np.int8), scores


def select_threshold(y, p, objective="f1", target_sensitivity=0.95) -> float:
    """Select a threshold with p >= threshold prediction semantics.

    The F1 path intentionally retains the original sklearn PR-curve calculation,
    including its first-maximum tie behavior.
    """
    if objective not in ("f1", "sensitivity"):
        raise ValueError("threshold objective must be 'f1' or 'sensitivity'")
    labels, scores = _validated_arrays(y, p, target_sensitivity)
    target = _target_value(target_sensitivity)
    if objective == "f1":
        from sklearn.metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        return float(thresholds[int(np.nanargmax(f1))]) if thresholds.size else 0.5

    positive_scores = np.sort(scores[labels == 1])[::-1]
    positive_count = int(positive_scores.size)
    if positive_count == 0:
        raise ValueError("sensitivity threshold selection requires at least one positive label")
    required_true_positives = math.ceil(target * positive_count)
    # Match the empirical-recall comparison exactly if float multiplication
    # rounded a mathematically integral count just above or below an integer.
    while (required_true_positives > 1 and
           (required_true_positives - 1) / positive_count >= target):
        required_true_positives -= 1
    while required_true_positives <= positive_count and \
            required_true_positives / positive_count < target:
        required_true_positives += 1
    return float(positive_scores[required_true_positives - 1])


def ensure_resume_compatible(saved_policy, requested_policy) -> dict:
    """Treat pre-policy checkpoints as historical F1 runs and reject policy changes."""
    requested = _normalize_policy(requested_policy)
    if saved_policy is None:
        saved = make_threshold_policy("f1", 0.95)
    else:
        saved = _normalize_policy(saved_policy)
    if saved != requested:
        raise ValueError(
            "Resume threshold policy differs from the saved run; start a new output run "
            "to change threshold objective or sensitivity target"
        )
    return saved


def _normalize_policy(policy) -> dict:
    if not isinstance(policy, Mapping) or "objective" not in policy:
        raise ValueError("saved threshold policy metadata is invalid")
    target = policy.get("target_sensitivity")
    if policy["objective"] == "f1" and target is None:
        target = 0.95
    elif target is None:
        raise ValueError("saved sensitivity policy is missing its target")
    return make_threshold_policy(policy["objective"], target)
