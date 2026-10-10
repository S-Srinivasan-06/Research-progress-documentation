"""Focused checks for threshold selection; no training dependencies required."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from sensitivity_policy import (
    ensure_resume_compatible,
    make_threshold_policy,
    select_threshold,
)


def _reference_precision_recall_curve(y, scores):
    """Small binary PR curve equivalent for checking the runner's old F1 rule."""
    thresholds = np.unique(scores)
    positives = np.count_nonzero(y == 1)
    precisions, recalls = [], []
    for threshold in thresholds:
        pred = scores >= threshold
        tp = np.count_nonzero((y == 1) & pred)
        fp = np.count_nonzero((y == 0) & pred)
        precisions.append(tp / (tp + fp) if tp + fp else 1.0)
        recalls.append(tp / positives if positives else 0.0)
    return (np.asarray(precisions + [1.0]),
            np.asarray(recalls + [0.0]), thresholds)


def _old_f1_threshold(y, scores):
    precision, recall, thresholds = _reference_precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))]) if thresholds.size else 0.5


def _brute_force_sensitivity_threshold(y, scores, target):
    positives = np.count_nonzero(y == 1)
    for threshold in np.unique(scores)[::-1]:
        pred = scores >= threshold
        tp = np.count_nonzero((y == 1) & pred)
        fn = np.count_nonzero((y == 1) & ~pred)
        if tp / (tp + fn) >= target:
            return float(threshold)
    raise AssertionError("the minimum observed score must include every positive")


class ThresholdPolicyTests(unittest.TestCase):
    def test_sensitivity_threshold_and_confusion_counts_include_ties(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.7, 0.2, 0.7, 0.4, 0.1])
        threshold = select_threshold(y, scores, "sensitivity", 2 / 3)
        pred = scores >= threshold
        tn = np.count_nonzero((y == 0) & ~pred)
        fp = np.count_nonzero((y == 0) & pred)
        fn = np.count_nonzero((y == 1) & ~pred)
        tp = np.count_nonzero((y == 1) & pred)
        self.assertEqual(threshold, 0.7)
        self.assertEqual((tn, fp, fn, tp), (2, 1, 1, 2))
        self.assertGreaterEqual(tp / (tp + fn), 2 / 3)

    def test_target_one_uses_highest_threshold_with_full_recall(self):
        threshold = select_threshold([1, 1, 0], [0.8, 0.3, 0.5], "sensitivity", 1.0)
        self.assertEqual(threshold, 0.3)

    def test_sensitivity_matches_brute_force_on_tied_scores(self):
        rng = np.random.default_rng(731)
        targets = (0.01, 0.07, 0.29, 0.5, 2 / 3, 0.95, 1.0)
        for _ in range(80):
            n = int(rng.integers(3, 30))
            y = rng.integers(0, 2, size=n)
            y[0] = 1
            scores = rng.integers(0, 11, size=n).astype(float) / 10
            for target in targets:
                with self.subTest(target=target, y=y.tolist(), scores=scores.tolist()):
                    expected = _brute_force_sensitivity_threshold(y, scores, target)
                    self.assertEqual(select_threshold(y, scores, "sensitivity", target), expected)

    def test_f1_default_matches_historical_pr_curve_selection(self):
        sklearn = types.ModuleType("sklearn")
        metrics = types.ModuleType("sklearn.metrics")
        metrics.precision_recall_curve = _reference_precision_recall_curve
        sklearn.metrics = metrics
        cases = [
            (np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1])),
            (np.array([1, 1, 1, 0, 0]), np.array([0.8, 0.5, 0.5, 0.5, 0.2])),
        ]
        with patch.dict(sys.modules, {"sklearn": sklearn, "sklearn.metrics": metrics}):
            for y, scores in cases:
                expected = _old_f1_threshold(y, scores)
                self.assertEqual(select_threshold(y, scores), expected)
                self.assertEqual(select_threshold(y, scores, "f1", 0.95), expected)

    def test_rejects_invalid_labels_and_scores(self):
        invalid_cases = [
            ([0, 2], [0.1, 0.9]),
            ([0, np.nan], [0.1, 0.9]),
            ([[0, 1]], [0.1, 0.9]),
            ([0, 1], [0.1]),
            ([0, 1], [0.1, np.nan]),
            ([0, 1], [0.1, np.inf]),
            ([0, 1], [-0.1, 0.9]),
            ([0, 1], [0.1, 1.1]),
            ([0, 0], [0.1, 0.9]),
        ]
        for labels, scores in invalid_cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    select_threshold(labels, scores, "sensitivity", 0.95)

    def test_rejects_invalid_targets_and_objectives(self):
        for target in (0, -0.1, 1.01, np.nan, np.inf, None, True):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    make_threshold_policy("sensitivity", target)
                with self.assertRaises(ValueError):
                    select_threshold([0, 1], [0.2, 0.8], "sensitivity", target)
        with self.assertRaises(ValueError):
            make_threshold_policy("precision", 0.95)

    def test_resume_policy_is_explicit_and_legacy_means_f1(self):
        f1 = make_threshold_policy("f1", 0.95)
        sensitivity = make_threshold_policy("sensitivity", 0.95)
        self.assertEqual(ensure_resume_compatible(None, f1), f1)
        with self.assertRaises(ValueError):
            ensure_resume_compatible(None, sensitivity)
        with self.assertRaises(ValueError):
            ensure_resume_compatible(sensitivity, make_threshold_policy("sensitivity", 0.9))


if __name__ == "__main__":
    unittest.main()
