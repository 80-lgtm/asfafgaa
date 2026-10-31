import unittest

import numpy as np


class CoresetTests(unittest.TestCase):
    def test_greedy_coreset_picks_k_and_covers(self):
        from src.memory_bank import greedy_coreset

        rng = np.random.RandomState(0)
        features = rng.rand(50, 8).astype(np.float32)
        subset = greedy_coreset(features, k=5, seed=0)
        self.assertEqual(subset.shape, (5, 8))
        self.assertEqual(greedy_coreset(features, k=5, seed=0).tolist(), subset.tolist())

    def test_coreset_rejects_invalid_k(self):
        from src.memory_bank import greedy_coreset

        with self.assertRaises(ValueError):
            greedy_coreset(np.zeros((4, 2)), k=0, seed=0)
        with self.assertRaises(ValueError):
            greedy_coreset(np.zeros((4, 2)), k=5, seed=0)


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_ci_contains_zero_for_identical_scores(self):
        from src.stats import paired_bootstrap_ci

        a = np.array([0.1, 0.5, 0.9, 0.3])
        lo, hi = paired_bootstrap_ci(a, a.copy(), seed=0, n_boot=200)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_paired_bootstrap_detects_positive_shift(self):
        from src.stats import paired_bootstrap_ci

        a = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.5])
        b = a + 0.2
        lo, hi = paired_bootstrap_ci(b, a, seed=0, n_boot=500)
        self.assertGreater(lo, 0.0)

    def test_bootstrap_rejects_mismatched_lengths(self):
        from src.stats import paired_bootstrap_ci

        with self.assertRaises(ValueError):
            paired_bootstrap_ci(np.array([1.0]), np.array([1.0, 2.0]), seed=0, n_boot=10)

    def test_cluster_bootstrap_resamples_clusters_not_rows(self):
        from src.stats import cluster_bootstrap_indices

        clusters = np.array(["a", "a", "b", "b", "c"])
        samples = list(cluster_bootstrap_indices(clusters, seed=0, n_boot=20))
        self.assertEqual(len(samples), 20)
        for idx in samples:
            selected = clusters[idx]
            # A cluster may be sampled 0, 1, 2, ... times, always with all rows.
            for value in ("a", "b", "c"):
                count = int(np.sum(selected == value))
                original = int(np.sum(clusters == value))
                self.assertEqual(count % original, 0)

    def test_cluster_bootstrap_ci_uses_metric_on_resampled_indices(self):
        from src.stats import cluster_bootstrap_ci

        values = np.array([1.0, 3.0, 5.0])
        clusters = np.array(["a", "a", "b"])
        result = cluster_bootstrap_ci(lambda idx: float(values[idx].mean()), clusters,
                                      seed=0, n_boot=200)
        self.assertLessEqual(result["ci95"][0], result["ci95"][1])
        self.assertGreaterEqual(result["ci95"][0], 1.0)
        self.assertLessEqual(result["ci95"][1], 5.0)
        self.assertEqual(result["n_defined"] + result["n_skipped"], 200)

    def test_cluster_bootstrap_reports_undefined_replicates(self):
        from src.stats import cluster_bootstrap_ci

        def metric(idx):
            if len(idx) == 1:
                raise ValueError("undefined")
            return 0.0

        result = cluster_bootstrap_ci(metric, np.array(["a", "b"]), seed=0, n_boot=5)
        self.assertGreaterEqual(result["n_skipped"], 0)

    def test_cluster_bootstrap_rejects_bad_inputs(self):
        from src.stats import cluster_bootstrap_ci, cluster_bootstrap_indices

        with self.assertRaises(ValueError):
            list(cluster_bootstrap_indices(np.array([]), seed=0, n_boot=1))
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(lambda idx: 0.0, np.array([["a"], ["b"]]), seed=0, n_boot=1)
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(lambda idx: 0.0, np.array(["a"]), seed=0, n_boot=0)

    def test_region_any_fpr_counts_one_region_for_dual_states(self):
        from src.stats import region_any_fpr

        labels = np.array([0, 0, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 2.0])
        clusters = np.array(["r1", "r1", "r2", "r3"])
        groups = np.array([False, True, True, False])
        self.assertEqual(region_any_fpr(labels, scores, clusters, threshold=0.5), 0.5)
        self.assertEqual(
            region_any_fpr(labels, scores, clusters, threshold=0.5,
                           group=groups.astype(int)), 0.5)
        self.assertEqual(region_any_fpr(labels, scores, clusters, threshold=0.5,
                                        group=groups), 0.5)

    def test_region_any_fpr_rejects_group_on_faulty_rows(self):
        from src.stats import region_any_fpr

        labels = np.array([0, 1])
        scores = np.array([0.1, 0.2])
        clusters = np.array(["r1", "r2"])
        groups = np.array([False, True])
        with self.assertRaises(ValueError):
            region_any_fpr(labels, scores, clusters, threshold=0.5, group=groups)



class BboxTests(unittest.TestCase):
    def test_bbox_hit_rate_counts_overlap(self):
        from src.localization import bbox_hit

        heat = np.zeros((10, 10), dtype=np.float32)
        heat[2:5, 2:5] = 1.0
        self.assertTrue(bbox_hit(heat, (2, 2, 5, 5), threshold=0.5))
        self.assertFalse(bbox_hit(heat, (6, 6, 9, 9), threshold=0.5))

    def test_parse_faulty_labels(self):
        import io

        from src.localization import parse_labels_csv

        text = "image;region;time;class;xmin;ymin;xmax;ymax;split\n000001.png;1;1;Circ6;10;20;30;40;val\n"
        boxes = parse_labels_csv(io.StringIO(text))
        self.assertEqual(boxes["000001.png"], [(10, 20, 30, 40)])


class FrequencyStripeTests(unittest.TestCase):
    def test_fft_stripe_strength_detects_stripes(self):
        from src.stripe_context import fft_stripe_strength

        rng = np.random.RandomState(0)
        noise = rng.rand(64, 64).astype(np.float32)
        stripes = np.tile(np.sin(np.linspace(0, 12 * np.pi, 64)), (64, 1)).astype(np.float32)
        self.assertGreater(fft_stripe_strength(stripes), fft_stripe_strength(noise))
        self.assertEqual(fft_stripe_strength(np.zeros((32, 32))), 0.0)

    def test_fft_strength_bounded(self):
        from src.stripe_context import fft_stripe_strength

        img = np.random.RandomState(1).rand(64, 64)
        v = fft_stripe_strength(img)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
