import unittest


class CliTests(unittest.TestCase):
    def test_parse_arguments_defaults_to_pilot(self):
        from main import parse_args

        args = parse_args(["--config", "configs/pilot.yaml"])
        self.assertEqual(args.config, "configs/pilot.yaml")
        self.assertEqual(args.mode, "pilot")

    def test_missing_config_is_rejected(self):
        from main import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--mode", "smoke"])


if __name__ == "__main__":
    unittest.main()
