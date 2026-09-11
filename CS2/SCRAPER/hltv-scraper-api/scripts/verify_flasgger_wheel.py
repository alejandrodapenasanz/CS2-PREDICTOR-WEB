#!/usr/bin/env python3
"""Verify the vendored flasgger wheel by digest and package metadata.

Input is an optional wheel path. By default the script verifies the single
allowlisted artifact below wheels/. Execute with Python 3.13 from the scraper
project root; it performs no network access and writes no files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Sequence
from zipfile import BadZipFile, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHEEL_FILENAME = "flasgger-0.9.7.1-py2.py3-none-any.whl"
EXPECTED_SHA256 = "a25e4a666b2894ba8833acec2d0827646e675f914f25664d68bc88f65e2e7783"
EXPECTED_NAME = "flasgger"
EXPECTED_VERSION = "0.9.7.1"


class WheelVerificationError(RuntimeError):
    """Report a wheel that does not match the allowlisted artifact contract."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_members(archive: ZipFile) -> list[str]:
    """Return member names after rejecting unsafe archive paths."""

    names: list[str] = []
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise WheelVerificationError(f"wheel contains an unsafe archive path: {info.filename!r}")
        names.append(info.filename)
    return names


def verify_flasgger_wheel(path: Path | str) -> Path:
    """Validate filename, digest, archive paths, name, and version."""

    wheel_path = Path(path).resolve()
    if wheel_path.name != WHEEL_FILENAME:
        raise WheelVerificationError(f"unexpected wheel filename {wheel_path.name!r}; expected {WHEEL_FILENAME!r}")
    if not wheel_path.is_file():
        raise WheelVerificationError(f"vendored wheel not found: {wheel_path}")

    actual_digest = sha256_file(wheel_path)
    if actual_digest != EXPECTED_SHA256:
        raise WheelVerificationError(
            f"vendored wheel SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_digest}"
        )

    try:
        with ZipFile(wheel_path) as archive:
            names = _validated_members(archive)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise WheelVerificationError("wheel must contain exactly one .dist-info/METADATA file")
            metadata_text = archive.read(metadata_names[0]).decode("utf-8")
    except (BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise WheelVerificationError(f"wheel archive is invalid: {exc}") from exc

    metadata = Parser().parsestr(metadata_text)
    name = (metadata.get("Name") or "").strip().lower()
    version = (metadata.get("Version") or "").strip()
    if name != EXPECTED_NAME or version != EXPECTED_VERSION:
        raise WheelVerificationError(
            f"wheel metadata mismatch: expected {EXPECTED_NAME}=={EXPECTED_VERSION}, got {name}=={version}"
        )
    return wheel_path


def build_parser() -> argparse.ArgumentParser:
    """Build the offline verification CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheel",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "wheels" / WHEEL_FILENAME,
        help="wheel to verify (default: the vendored flasgger artifact)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run verification and return a shell-friendly status code."""

    arguments = build_parser().parse_args(argv)
    try:
        verified = verify_flasgger_wheel(arguments.wheel)
    except WheelVerificationError as exc:
        print(f"Wheel verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {verified.name}: sha256={EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
