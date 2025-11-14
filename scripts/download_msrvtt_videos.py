#!/usr/bin/env python3
"""
Download the MSR-VTT video archive from Hugging Face and extract it into the
expected directory structure for CALM training.

This script relies on `huggingface_hub` to pull the `MSRVTT_Videos.zip` asset
from the dataset repository `friedrichor/MSR-VTT`. The videos are extracted to
`datasets/msrvtt/videos/` by default so that `scripts/train.py` can locate the
clips.

Example:
    python scripts/download_msrvtt_videos.py \
        --output-dir ./datasets/msrvtt/videos
"""

from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - import guard for optional dep
    raise SystemExit(
        "huggingface_hub is required. Install it with `pip install huggingface_hub`."
    ) from exc


REPO_ID = "friedrichor/MSR-VTT"
ARCHIVE_FILENAME = "MSRVTT_Videos.zip"


def archive_present(output_dir: Path) -> bool:
    """Return True if videos appear to already exist in the target directory."""
    return any(output_dir.glob("*.mp4"))


def download_archive(destination_dir: Path, overwrite: bool) -> Path:
    """
    Download the MSR-VTT video archive to `destination_dir`.

    Returns the path to the downloaded archive.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / ARCHIVE_FILENAME

    if archive_path.exists() and not overwrite:
        logging.info("Using existing archive at %s", archive_path)
        return archive_path

    logging.info(
        "Downloading %s from Hugging Face dataset %s", ARCHIVE_FILENAME, REPO_ID
    )

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ARCHIVE_FILENAME,
        repo_type="dataset",
        local_dir=str(destination_dir),
        local_dir_use_symlinks=False,
    )

    logging.info("Archive stored at %s", downloaded_path)
    return Path(downloaded_path)


def extract_archive(archive_path: Path, output_dir: Path, overwrite: bool) -> None:
    """Extract the archive contents into `output_dir`."""
    if archive_present(output_dir) and not overwrite:
        logging.info("Videos already present in %s; skipping extraction", output_dir)
        return

    if output_dir.exists() and overwrite:
        logging.info("Clearing existing directory %s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Extracting archive to %s", output_dir)

    with zipfile.ZipFile(archive_path) as zip_file:
        members = zip_file.namelist()
        # If the archive nests files under a top-level folder, strip it out.
        has_root_folder = all("/" in name for name in members if not name.endswith("/"))

        if has_root_folder:
            top_level = {name.split("/", 1)[0] for name in members if "/" in name}
            logging.info("Detected top-level folder(s) in archive: %s", top_level)

        zip_file.extractall(path=output_dir)

        if has_root_folder:
            # Move contents one level up so mp4 files sit directly in output_dir.
            for child in output_dir.iterdir():
                if child.is_dir():
                    for item in child.iterdir():
                        target = output_dir / item.name
                        if target.exists():
                            logging.warning(
                                "Target %s already exists; leaving item in %s",
                                target,
                                child,
                            )
                        else:
                            item.rename(target)
                    child.rmdir()

    logging.info("Extraction complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract MSR-VTT video clips."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/msrvtt/videos"),
        help="Directory where video files should be extracted.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("./datasets/msrvtt"),
        help="Directory to store the downloaded archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-extract even if files already exist.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded ZIP archive after extraction.",
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

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()

    logging.info("Preparing to download MSR-VTT videos to %s", output_dir)
    try:
        archive_path = download_archive(archive_dir, overwrite=args.overwrite)
        extract_archive(archive_path, output_dir, overwrite=args.overwrite)
    except Exception as exc:  # pragma: no cover - log any unexpected issue
        logging.error("Failed to download or extract videos: %s", exc)
        raise SystemExit(1) from exc

    if not args.keep_archive:
        try:
            archive_path.unlink()
            logging.info("Removed archive file %s", archive_path)
        except OSError:
            logging.warning("Could not remove archive file %s", archive_path)

    logging.info("MSR-VTT videos are ready.")


if __name__ == "__main__":
    main()

