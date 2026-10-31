import io
import unittest

import numpy as np


class PilotSelectionTests(unittest.TestCase):
    def test_stratified_selection_preserves_official_split_and_regions(self):
        from src.pilot import stratified_sample

        rows = []
        for split in ("train", "val", "test"):
            for kind in ("faulty", "good_clean", "good_stripes"):
                for i in range(5):
                    rows.append(
                        {
                            "image": f"{split}-{kind}-{i}.png",
                            "region": f"{i}",
                            "time": "1",
                            "type": kind,
                            "split": split,
                        }
                    )
        selected = stratified_sample(rows, per_group=2, seed=7)
        self.assertEqual(len(selected), 18)
        self.assertEqual(
            {(r["split"], r["type"]) for r in selected},
            {(s, t) for s in ("train", "val", "test") for t in ("faulty", "good_clean", "good_stripes")},
        )
        self.assertEqual(selected, stratified_sample(rows, per_group=2, seed=7))

    def test_selection_rejects_insufficient_group(self):
        from src.pilot import stratified_sample

        with self.assertRaisesRegex(ValueError, "insufficient rows"):
            stratified_sample([{"type": "faulty", "split": "train"}], per_group=2, seed=7)


class StripeMapTests(unittest.TestCase):
    def test_global_stripe_strength_is_bounded(self):
        from src.stripe_context import global_stripe_strength

        uniform = np.zeros((32, 32), dtype=np.float32)
        stripes = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
        self.assertEqual(global_stripe_strength(uniform), 0.0)
        self.assertGreater(global_stripe_strength(stripes), 0.0)
        self.assertLessEqual(global_stripe_strength(stripes), 1.0)

    def test_context_map_suppresses_nonstripe_images(self):
        from src.stripe_context import stripe_consistency_map

        image = np.zeros((32, 32), dtype=np.float32)
        result = stripe_consistency_map(image, grid_shape=(4, 4))
        self.assertEqual(result.shape, (4, 4))
        np.testing.assert_allclose(result, 0.0)


class MetricTests(unittest.TestCase):
    def test_operating_metrics_use_validation_threshold(self):
        from src.evaluation import operating_metrics

        result = operating_metrics(
            labels=np.array([1, 1, 0, 0]),
            scores=np.array([0.9, 0.4, 0.8, 0.2]),
            stripe_labels=np.array([0, 0, 1, 0]),
            threshold=0.5,
        )
        self.assertEqual(result["faulty_recall"], 0.5)
        self.assertEqual(result["stripe_fpr"], 1.0)
        self.assertEqual(result["normal_fpr"], 0.5)

    def test_threshold_uses_youden_on_validation_only(self):
        from src.evaluation import youden_threshold

        threshold = youden_threshold(
            labels=np.array([0, 0, 1, 1]),
            scores=np.array([0.1, 0.2, 0.8, 0.9]),
        )
        predictions = np.array([0.1, 0.2, 0.8, 0.9]) >= threshold
        np.testing.assert_array_equal(predictions, [False, False, True, True])

    def test_threshold_is_midpoint_for_perfectly_separated_scores(self):
        from src.evaluation import youden_threshold

        threshold = youden_threshold(
            labels=np.array([0, 0, 1, 1]),
            scores=np.array([0.1, 0.2, 0.8, 0.9]),
        )
        self.assertAlmostEqual(threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
