import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request


SADD_URL = "https://zenodo.org/api/records/18301911/files/SADD_v1.1.zip/content"
SADD_MD5 = "3b694539f8fa12bc12af2ee4f939f064"
EXPECTED_SIZE = 4964010557


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify SADD v1.1")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--download", action="store_true", help="actually download the 4.96GB archive")
    args = parser.parse_args(argv)
    if not args.download:
        print("Dry run. Pass --download to start the 4.96GB download.")
        return 0
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    resume_from = args.destination.stat().st_size if args.destination.exists() else 0
    if resume_from > EXPECTED_SIZE:
        print(
            f"File is larger than expected: {resume_from} > {EXPECTED_SIZE}",
            file=sys.stderr,
        )
        return 1
    if args.destination.exists():
        print(f"Existing file size: {resume_from} bytes")
    if resume_from < EXPECTED_SIZE:
        args.destination.touch()
        request = urllib.request.Request(SADD_URL, headers={"Range": f"bytes={resume_from}-"})
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise RuntimeError("server did not accept resume request; restart download from zero")
            print(f"Downloading from byte {resume_from} to {EXPECTED_SIZE}", flush=True)
            with args.destination.open("ab") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    if handle.tell() % (100 * 1024 * 1024) < 1024 * 1024:
                        print(f"progress: {handle.tell() / 1024 / 1024:.1f} MiB", flush=True)
    actual = md5_file(args.destination)
    if actual != SADD_MD5:
        print(f"MD5 mismatch: expected {SADD_MD5}, got {actual}", file=sys.stderr)
        return 1
    print(f"MD5 verified: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
