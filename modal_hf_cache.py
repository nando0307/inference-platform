"""Offline inventory and path-safety checks for pinned Hugging Face snapshots."""

import json
import os
from pathlib import Path

MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
HF_HOME = Path("/opt/huggingface")
MANIFEST = HF_HOME / "verified-snapshots.json"
REQUIRED = {"config.json", "tokenizer.json", "tokenizer_config.json"}


def snapshot_path(root, repo, revision):
    return Path(root) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def inventory(snapshot, volume_root):
    snapshot = Path(snapshot)
    volume_root = Path(volume_root).resolve()
    paths = set()
    for base, dirs, files in os.walk(snapshot, followlinks=False):
        for name in tuple(dirs):
            path = Path(base) / name
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not target.is_relative_to(volume_root):
                    raise ValueError(f"symlink escapes Volume: {path.relative_to(snapshot)} -> {target}")
                dirs.remove(name)
                paths.add(path.relative_to(snapshot).as_posix())
        for name in files:
            path = Path(base) / name
            relative = path.relative_to(snapshot).as_posix()
            if path.is_symlink() and not path.resolve(strict=True).is_relative_to(volume_root):
                raise ValueError(f"symlink escapes Volume: {relative} -> {path.resolve()}")
            if not path.is_file():
                raise ValueError(f"unsupported snapshot path: {relative}")
            paths.add(relative)
    return paths


def verify_snapshot(snapshot, volume_root, expected):
    if not isinstance(expected, (list, tuple, set)) or not all(isinstance(path, str) for path in expected):
        raise ValueError("invalid snapshot path list")
    snapshot = Path(snapshot)
    actual = inventory(snapshot, volume_root)
    missing = set(expected) - actual
    extra = actual - set(expected)
    if missing or extra:
        raise ValueError(f"snapshot inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if not REQUIRED <= actual:
        raise ValueError(f"snapshot lacks required metadata: {sorted(REQUIRED - actual)}")
    if not any(Path(path).name.endswith((".safetensors", ".bin", ".pt")) for path in actual):
        raise ValueError("snapshot has no model weight files")
    return sorted(actual)


def verify_manifest(root=HF_HOME):
    root = Path(root)
    try:
        entries = json.loads((root / MANIFEST.name).read_text())
        for repo, revision in MODELS.items():
            if entries[repo]["revision"] != revision:
                raise ValueError(f"manifest revision mismatch for {repo}")
            verify_snapshot(snapshot_path(root, repo, revision), root, entries[repo]["paths"])
    except (OSError, KeyError, ValueError, TypeError) as e:
        raise RuntimeError(f"cache invalid; re-run a T4 lab command: {e}") from e
    return True
