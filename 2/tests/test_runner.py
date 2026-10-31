import unittest

import numpy as np


class RunnerTests(unittest.TestCase):
    def test_score_grid_combines_local_and_stripe_maps(self):
        from src.runner import score_grid

        local = np.array([2.0, 4.0], dtype=np.float32)
        consistency = np.array([0.5, 0.0], dtype=np.float32)
        result = score_grid(local, consistency, alpha=1.0)
        np.testing.assert_allclose(result, [4 / 3, 4.0])

    def test_alpha_selection_uses_validation_auroc(self):
        from src.runner import select_alpha

        candidates = [
            {"alpha": 0.0, "val_labels": [0, 1], "val_scores": [0.2, 0.8]},
            {"alpha": 1.0, "val_labels": [0, 1], "val_scores": [0.8, 0.2]},
        ]
        self.assertEqual(select_alpha(candidates), 0.0)

    def test_feature_cache_key_changes_with_image_hash(self):
        from src.runner import feature_cache_key

        first = feature_cache_key("000001.png", b"abc", image_size=224)
        second = feature_cache_key("000001.png", b"abd", image_size=224)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
