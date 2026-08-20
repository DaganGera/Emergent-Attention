"""Serves the pre-built sample-image gallery (see scripts/build_web_samples.py)."""

import json
import os

_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "samples")


def list_samples(domain: str) -> list[dict]:
    manifest_path = os.path.join(_SAMPLES_DIR, domain, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    for item in manifest:
        item["url"] = f"/static/samples/{domain}/{item['file']}"
    return manifest


def resolve_sample_path(domain: str, sample_id: str) -> str:
    for item in list_samples(domain):
        if item["id"] == sample_id:
            return os.path.join(_SAMPLES_DIR, domain, item["file"])
    raise FileNotFoundError(f"sample '{sample_id}' not found for domain '{domain}'")
