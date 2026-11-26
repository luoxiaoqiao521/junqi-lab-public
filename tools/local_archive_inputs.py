"""Read replay selection metadata kept outside source control."""
import json
from pathlib import Path, PureWindowsPath

MANIFEST = Path(__file__).resolve().parents[1] / "build/private/archive-inputs.json"


def archive_value(field):
    try:
        with MANIFEST.open("rb") as source:
            data = source.read(65_537)
        if len(data) > 65_536:
            raise ValueError("private manifest size")
        payload = json.loads(data)
        return payload[field]
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError("Required local archive metadata is missing or invalid") from None


def selected_replay(root, name, limit):
    if (not isinstance(name, str) or ":" in name or PureWindowsPath(name).name != name or
            not name.lower().endswith(".jgs")):
        raise ValueError("Expected a local JGS filename")
    try:
        folder = root.resolve(strict=True)
        path = (folder / name).resolve(strict=True)
        if path.parent != folder or not path.is_file():
            raise ValueError("Selected source must stay in the local archive directory")
        with path.open("rb") as source:
            data = source.read(limit + 1)
        if not 0 < len(data) <= limit:
            raise ValueError("Selected source is outside the supported size")
        return data
    except OSError:
        raise ValueError("Selected source is unavailable") from None
