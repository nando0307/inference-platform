"""CPU-only provision and offline verification for the pinned HF cache."""

import json

import modal

from modal_hf_cache import HF_HOME, MANIFEST, MODELS, snapshot_path, verify_snapshot

app = modal.App("inference-hf-cache")
volume = modal.Volume.from_name("inference-huggingface-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0")


@app.function(image=image, volumes={str(HF_HOME): volume}, cpu=4, timeout=3600, retries=0)
def provision():
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest = {}
    for repo, revision in MODELS.items():
        expected = sorted(api.list_repo_files(repo, revision=revision))
        snapshot_download(repo, revision=revision, cache_dir=str(HF_HOME))
        snapshot = snapshot_path(HF_HOME, repo, revision)
        paths = verify_snapshot(snapshot, HF_HOME, expected)
        manifest[repo] = {"revision": revision, "paths": paths}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return {repo: len(item["paths"]) for repo, item in manifest.items()}


@app.function(image=image, volumes={str(HF_HOME): volume}, cpu=1, timeout=300, retries=0)
def verify():
    from modal_hf_cache import verify_manifest

    return verify_manifest(HF_HOME)


@app.local_entrypoint()
def main(command: str = "verify"):
    if command == "provision":
        result = provision.remote()
    elif command == "verify":
        result = verify.remote()
    else:
        raise ValueError("command must be provision or verify")
    print(json.dumps(result, indent=2))
