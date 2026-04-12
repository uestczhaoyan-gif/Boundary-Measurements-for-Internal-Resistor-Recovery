from __future__ import annotations

import re
import sys
from pathlib import Path


def _parse_required_version_from_name(name: str) -> tuple[int, int] | None:
    match = re.search(r"py(\d{2,3})", name.lower())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 2:
        return int(digits[0]), int(digits[1])
    return int(digits[0]), int(digits[1:])


def prepend_vendor_dir(vendor_dir: Path, *, required_version: tuple[int, int] | None = None) -> bool:
    if not vendor_dir.exists():
        return False
    if required_version is None:
        required_version = _parse_required_version_from_name(vendor_dir.name)
    if required_version is not None and sys.version_info[:2] != required_version:
        return False
    vendor_path = str(vendor_dir)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    return True
