import io
import unittest


DATASET_CSV = (
    "image;region;time;type;split;subset_a;subset_b\n"
    "000001.png;100;1;faulty;train;1;1\n"
    "000002.png;100;2;faulty;train;1;1\n"
    "000003.png;101;1;good_clean;train;0;1\n"
)


class SaddTests(unittest.TestCase):
    def test_parse_semicolon_manifest(self):
        from src.sadd import parse_dataset_csv

        rows = parse_dataset_csv(io.StringIO(DATASET_CSV))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["image"], "000001.png")
        self.assertEqual(rows[0]["type"], "faulty")

    def test_archive_path_uses_type_and_official_split(self):
        from src.sadd import archive_image_path

        self.assertEqual(
            archive_image_path({"image": "000001.png", "type": "faulty", "split": "train"}),
            "SADD/images/faulty/train/000001.png",
        )

    def test_one_image_per_region_is_deterministic(self):
        from src.sadd import one_per_region, parse_dataset_csv

        rows = parse_dataset_csv(io.StringIO(DATASET_CSV))
        selected = one_per_region(rows, time="1")
        self.assertEqual([r["image"] for r in selected], ["000001.png", "000003.png"])

    def test_rejects_nonpositive_time(self):
        from src.sadd import one_per_region, parse_dataset_csv

        with self.assertRaisesRegex(ValueError, "time must be positive"):
            one_per_region(parse_dataset_csv(io.StringIO(DATASET_CSV)), time="0")

    def test_median_time_per_region_supports_different_time_ranges(self):
        from src.sadd import median_time_per_region, parse_dataset_csv

        rows = parse_dataset_csv(io.StringIO(DATASET_CSV))
        rows.append(
            {
                "image": "000004.png",
                "region": "102",
                "time": "58",
                "type": "good_stripes",
                "split": "val",
                "subset_a": "0",
                "subset_b": "1",
            }
        )
        selected = median_time_per_region(rows)
        self.assertEqual([row["image"] for row in selected], ["000001.png", "000003.png", "000004.png"])


if __name__ == "__main__":
    unittest.main()
