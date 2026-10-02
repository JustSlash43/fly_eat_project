"""
Downloads the public MaleCNS v1.0 connectome (Janelia FlyEM + Google Research),
released under CC-BY 4.0. No account or API key needed.

Source: https://male-cns.janelia.org/download/
Bucket: gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
HTTPS mirror: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/

By default this fetches only the three files the simulator needs. The raw
synapse-point files (syn-points / syn-partners / tbar-neurotransmitters) are
6-13 GB each and are not required to build a weighted connectivity graph, so
they're skipped unless --full is passed.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

CORE_FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

FULL_EXTRA_FILES = [
    "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "syn-points-male-cns-v1.0-minconf-0.5.feather",
    "syn-partners-male-cns-v1.0-minconf-0.5.feather",
    "tbar-neurotransmitters-male-cns-v1.0.feather",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_file(filename: str, dest_dir: Path, chunk_size: int = 1 << 20) -> None:
    url = f"{BASE_URL}/{filename}"
    dest = dest_dir / filename
    dest_dir.mkdir(parents=True, exist_ok=True)

    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))

    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        print(f"[skip] {filename} already downloaded ({remote_size / 1e6:.1f} MB)")
        return

    print(f"[get]  {filename} ({remote_size / 1e6:.1f} MB) <- {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f, tqdm(
            total=remote_size, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also download the large raw synapse-point files (~22 GB extra)",
    )
    parser.add_argument(
        "--dest", default=str(DATA_DIR), help="destination directory (default: ./data)"
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    files = list(CORE_FILES)
    if args.full:
        files += FULL_EXTRA_FILES

    print(f"Downloading MaleCNS v1.0 ({len(files)} file(s)) into {dest_dir}\n")
    for filename in files:
        try:
            download_file(filename, dest_dir)
        except requests.HTTPError as e:
            print(f"[error] failed to fetch {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nDone. Files in", dest_dir)
    for p in sorted(dest_dir.glob("*.feather")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
