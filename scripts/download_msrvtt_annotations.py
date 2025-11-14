#!/usr/bin/env python3
"""
Utility script to download MSR-VTT annotation splits used by CALM training.

The script fetches the JSON files hosted on Hugging Face (friedrichor/MSR-VTT)
and stores them under the expected project layout:

datasets/
└── msrvtt/
    └── annotations/
        ├── train.json
        └── val.json

Example:
    python scripts/download_msrvtt_annotations.py \
        --output-dir ./datasets/msrvtt/annotations
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import requests


HUGGING_FACE_BASE_URL = (
    "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main"
)

# Mapping from local filename to remote resource
ANNOTATION_FILES: Dict[str, str] = {
    "train.json": "msrvtt_train_9k.json",
    "val.json": "msrvtt_test_1k.json",
}


def download_file(url: str, destination: Path) -> None:
    """Download a file streaming its content to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s → %s", url, destination)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)


def download_annotations(output_dir: Path, overwrite: bool = False) -> None:
    """Download train/val annotations into the desired directory."""
    for local_name, remote_name in ANNOTATION_FILES.items():
        url = f"{HUGGING_FACE_BASE_URL}/{remote_name}"
        destination = output_dir / local_name

        if destination.exists() and not overwrite:
            logging.info("Skipping existing file %s", destination)
            continue

        download_file(url, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MSR-VTT annotations required for CALM training."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/msrvtt/annotations"),
        help="Directory to store the downloaded annotation JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = args.output_dir.resolve()
    logging.info("Saving MSR-VTT annotations to %s", output_dir)

    try:
        download_annotations(output_dir, overwrite=args.overwrite)
    except requests.HTTPError as exc:
        logging.error("HTTP error while downloading annotations: %s", exc)
        raise SystemExit(1) from exc
    except requests.RequestException as exc:
        logging.error("Network error while downloading annotations: %s", exc)
        raise SystemExit(1) from exc

    logging.info("MSR-VTT annotations are ready.")


if __name__ == "__main__":
    main()

