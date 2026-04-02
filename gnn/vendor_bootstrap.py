import platform
import sys
from dataclasses import dataclass
from pathlib import Path


_LOGGED_SKIPS = set()


@dataclass(frozen=True)
class VendorBootstrapResult:
    path: Path
    inserted: bool
    skipped: bool
    reason: str
    wheel_tags: tuple[str, ...]


def _normalize_runtime_platform(runtime_platform=None):
    if runtime_platform:
        return str(runtime_platform).strip().lower()
    return (platform.system() or sys.platform or "unknown").strip().lower()


def _is_windows_runtime(runtime_platform):
    return runtime_platform.startswith("win") or runtime_platform in {"nt", "windows"}


def _collect_wheel_tags(vendor_dir):
    tags = []
    for wheel_file in sorted(vendor_dir.glob("*.dist-info/WHEEL")):
        try:
            lines = wheel_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("Tag:"):
                tag = line.split(":", 1)[1].strip()
                if tag and tag not in tags:
                    tags.append(tag)
    return tuple(tags)


def _format_tag_preview(wheel_tags):
    if not wheel_tags:
        return "no wheel tags found"
    if len(wheel_tags) <= 3:
        return ", ".join(wheel_tags)
    return f"{', '.join(wheel_tags[:3])}, ..."


def _log_skip_once(vendor_dir, runtime_platform, wheel_tags):
    key = (str(vendor_dir), runtime_platform)
    if key in _LOGGED_SKIPS:
        return
    _LOGGED_SKIPS.add(key)
    print(
        "[VendorBootstrap] Skipping incompatible vendor "
        f"{vendor_dir} for runtime '{runtime_platform}' "
        f"(wheel tags: {_format_tag_preview(wheel_tags)})"
    )


def _bootstrap_single_vendor(vendor_dir, runtime_platform, verbose):
    if not vendor_dir.exists():
        return VendorBootstrapResult(
            path=vendor_dir,
            inserted=False,
            skipped=False,
            reason="missing",
            wheel_tags=(),
        )

    wheel_tags = _collect_wheel_tags(vendor_dir)
    has_windows_wheel = any("win_amd64" in tag for tag in wheel_tags)
    if wheel_tags and has_windows_wheel and not _is_windows_runtime(runtime_platform):
        if verbose:
            _log_skip_once(vendor_dir, runtime_platform, wheel_tags)
        return VendorBootstrapResult(
            path=vendor_dir,
            inserted=False,
            skipped=True,
            reason=f"incompatible with runtime '{runtime_platform}'",
            wheel_tags=wheel_tags,
        )

    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
        return VendorBootstrapResult(
            path=vendor_dir,
            inserted=True,
            skipped=False,
            reason="inserted",
            wheel_tags=wheel_tags,
        )

    return VendorBootstrapResult(
        path=vendor_dir,
        inserted=False,
        skipped=False,
        reason="already on sys.path",
        wheel_tags=wheel_tags,
    )


def bootstrap_vendor_paths(project_root, runtime_platform=None, verbose=True):
    project_root = Path(project_root).resolve()
    runtime_label = _normalize_runtime_platform(runtime_platform)
    vendor_dirs = (
        project_root / ".vendor_torchpy311",
        project_root / "inverse_identifiability" / ".vendor",
    )
    return tuple(
        _bootstrap_single_vendor(vendor_dir, runtime_label, verbose)
        for vendor_dir in vendor_dirs
    )


def format_dependency_import_error(module_name, original_error, bootstrap_results, runtime_platform=None):
    runtime_label = _normalize_runtime_platform(runtime_platform)
    skipped = [result for result in bootstrap_results if result.skipped]

    lines = [f"Failed to import '{module_name}' while starting GNN joint inference."]
    if skipped:
        lines.append(f"Skipped incompatible local vendor directories for runtime '{runtime_label}':")
        for result in skipped:
            lines.append(f"- {result.path} (wheel tags: {_format_tag_preview(result.wheel_tags)})")
        lines.append(
            "Install compatible dependencies in the active environment or provide "
            f"{runtime_label}-compatible vendor packages for these directories."
        )
    else:
        lines.append(
            "No compatible local vendor fallback resolved this dependency. "
            "Install it in the active environment or provide a matching vendor package."
        )
    lines.append(f"Import error summary: {original_error.__class__.__name__}: {original_error}")
    return "\n".join(lines)
