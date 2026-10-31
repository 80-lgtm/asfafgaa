import unittest

import numpy as np


class DatasetTests(unittest.TestCase):
    def test_rejects_manifest_with_missing_columns(self):
        from src.dataset import validate_manifest

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_manifest({"image_path": ["a.png"], "label": ["Good_clean"]})

    def test_deterministic_split_preserves_requested_counts(self):
        from src.dataset import deterministic_split

        rows = [{"id": str(i), "split": ""} for i in range(12)]
        split = deterministic_split(rows, seed=42, train=6, validation=3, test=3)
        self.assertEqual(
            {k: len(v) for k, v in split.items()}, {"train": 6, "validation": 3, "test": 3}
        )
        self.assertEqual(
            [r["id"] for r in split["train"]],
            [r["id"] for r in deterministic_split(list(rows), seed=42, train=6, validation=3, test=3)["train"]],
        )


class MemoryBankTests(unittest.TestCase):
    def test_nearest_memory_distance_and_patch_scores(self):
        from src.memory_bank import MemoryBank

        bank = MemoryBank()
        bank.add(np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(float(bank.nearest_distance(np.array([0.25, 0.0]))), 0.25)
        np.testing.assert_allclose(
            bank.patch_scores(np.array([[1.2, 0.0], [0.0, 2.0]])), [0.2, 2.0], rtol=1e-6
        )


class StripeContextTests(unittest.TestCase):
    def test_stripe_consistency_is_bounded(self):
        from src.stripe_context import StripeContextScorer

        scorer = StripeContextScorer()
        self.assertEqual(float(scorer.consistency(np.zeros((8, 8)))), 0.0)
        self.assertGreaterEqual(float(scorer.consistency(np.tile(np.arange(8), (8, 1)))), 0.0)
        self.assertLessEqual(float(scorer.consistency(np.tile(np.arange(8), (8, 1)))), 1.0)


class ScoringTests(unittest.TestCase):
    def test_stripe_penalty_reduces_positive_scores_only(self):
        from src.scoring import saga_score

        scores = np.array([2.0, 0.0])
        consistency = np.array([0.5, 1.0])
        np.testing.assert_allclose(saga_score(scores, consistency, alpha=1.0), [4 / 3, 0.0])

    def test_rejects_mismatched_score_lengths(self):
        from src.scoring import saga_score

        with self.assertRaises(ValueError):
            saga_score(np.array([1.0]), np.array([1.0, 0.0]), alpha=1.0)


class EvaluationTests(unittest.TestCase):
    def test_binary_metrics_include_stripe_false_positive_rate(self):
        from src.evaluation import binary_metrics

        metrics = binary_metrics(
            labels=np.array([1, 1, 0, 0]),
            scores=np.array([0.9, 0.4, 0.8, 0.2]),
            stripe_labels=np.array([0, 0, 1, 0]),
        )
        self.assertAlmostEqual(metrics["auroc"], 0.75)
        self.assertEqual(metrics["stripe_fpr"], 1.0)

    def test_rejects_mismatched_labels(self):
        from src.evaluation import binary_metrics

        with self.assertRaises(ValueError):
            binary_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), None)


if __name__ == "__main__":
    unittest.main()
