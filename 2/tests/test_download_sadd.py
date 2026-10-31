import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def workspace_tempdir():
    return tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_sadd.py"
SPEC = importlib.util.spec_from_file_location("download_sadd", SCRIPT_PATH)
download_sadd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_sadd)


class DownloadSaddTests(unittest.TestCase):
    def test_complete_file_skips_network_and_verifies_md5(self):
        with workspace_tempdir() as temp_dir:
            destination = Path(temp_dir) / "SADD_v1.1.zip"
            destination.write_bytes(b"already complete")
            with (
                mock.patch.object(download_sadd, "EXPECTED_SIZE", 16),
                mock.patch.object(download_sadd, "SADD_MD5", "5c043db4163b5bae319a4ca021b9b975"),
                mock.patch.object(download_sadd.urllib.request, "urlopen") as urlopen,
            ):
                self.assertEqual(download_sadd.main([str(destination), "--download"]), 0)
            urlopen.assert_not_called()

    def test_oversized_file_is_rejected_without_network(self):
        with workspace_tempdir() as temp_dir:
            destination = Path(temp_dir) / "SADD_v1.1.zip"
            destination.write_bytes(b"too large")
            with (
                mock.patch.object(download_sadd, "EXPECTED_SIZE", 4),
                mock.patch.object(download_sadd.urllib.request, "urlopen") as urlopen,
            ):
                self.assertEqual(download_sadd.main([str(destination), "--download"]), 1)
            urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
