import unittest

import numpy as np


class MatchedControlTests(unittest.TestCase):
    def test_matched_banks_each_have_ten_frames(self):
        from scripts.run_region_uncertainty import sample_matched_frames

        clean = [{"image": f"c{i}.png", "type": "good_clean"} for i in range(10)]
        stripes = [{"image": f"s{i}.png", "type": "good_stripes"} for i in range(9)]
        clean_bank, mixed_bank = sample_matched_frames(clean, stripes, seed=17)
        self.assertEqual(len(clean_bank), 10)
        self.assertEqual(len(mixed_bank), 10)
        self.assertEqual(sum(r["type"] == "good_clean" for r in mixed_bank), 5)
        self.assertEqual(sum(r["type"] == "good_stripes" for r in mixed_bank), 5)

    def test_matched_frame_sampling_is_reproducible(self):
        from scripts.run_region_uncertainty import sample_matched_frames

        clean = [{"image": f"c{i}.png", "type": "good_clean"} for i in range(10)]
        stripes = [{"image": f"s{i}.png", "type": "good_stripes"} for i in range(9)]
        first = sample_matched_frames(clean, stripes, seed=17)
        second = sample_matched_frames(clean, stripes, seed=17)
        self.assertEqual(first, second)

    def test_matched_frame_sampling_rejects_empty_inputs(self):
        from scripts.run_region_uncertainty import sample_matched_frames

        with self.assertRaises(ValueError):
            sample_matched_frames([], [{"image": "s.png"}], seed=17)
        with self.assertRaises(ValueError):
            sample_matched_frames([{"image": "c.png"}], [], seed=17)


if __name__ == "__main__":
    unittest.main()
