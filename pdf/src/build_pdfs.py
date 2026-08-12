"""
Builds the three camera-ready PDFs (paper, patent disclosure, companion
guide) from the HTML templates in this directory.

Templates use {{IMG:name}} placeholders instead of embedded images, so the
templates stay small, readable, and diffable in git. This script substitutes
each placeholder with a base64 data URI, then hands the fully self-contained
HTML to pagedjs-cli (https://pagedjs.org) for pagination -- proper running
headers, page numbers, and forced page breaks, matching a real camera-ready
paper rather than a plain browser print-to-PDF.

Requires (not vendored into this repo, install once):
    npm install puppeteer-core pagedjs-cli
Puppeteer-core does not bundle Chromium; point it at a local Chrome install
via CHROME_PATH below, or install `puppeteer` (full) instead.

Usage:
    python pdf/src/build_pdfs.py
"""

import base64
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "paper")
OUT_DIR = os.path.join(REPO_ROOT, "pdf")

# name used in {{IMG:name}} -> (filename in figures/paper/, mime type)
PAPER_IMAGES = {
    "architecture": ("architecture.png", "image/png"),
    "corruption_robustness": ("corruption_robustness.png", "image/png"),
    "noise_drift": ("noise_drift.png", "image/png"),
    "gradcam_noisy": ("gradcam_noisy.png", "image/png"),  # swap to .jpg here if you re-export one
    "busi_seeds": ("busi_seeds.png", "image/png"),
    "receptive_field": ("receptive_field_block0.png", "image/png"),
}
PATENT_IMAGES = {
    "architecture": ("architecture.png", "image/png"),
    "corruption_robustness": ("corruption_robustness.png", "image/png"),
    "busi_seeds": ("busi_seeds.png", "image/png"),
}
COMPANION_IMAGES = {
    "architecture": ("architecture.png", "image/png"),
    "corruption_robustness": ("corruption_robustness.png", "image/png"),
    "noise_drift": ("noise_drift.png", "image/png"),
    "gradcam_noisy": ("gradcam_noisy.png", "image/png"),
    "busi_seeds": ("busi_seeds.png", "image/png"),
    "receptive_field": ("receptive_field_block0.png", "image/png"),
}


def embed_images(template_path: str, images: dict, out_path: str) -> None:
    with open(template_path, encoding="utf-8") as f:
        html = f.read()
    for key, (filename, mime) in images.items():
        path = os.path.join(FIGURES_DIR, filename)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        html = html.replace("{{IMG:" + key + "}}", f"data:{mime};base64,{b64}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def render(html_path: str, pdf_path: str, pagedjs_cli: str) -> None:
    subprocess.run([pagedjs_cli, html_path, "-o", pdf_path], check=True)
    print(f"Saved: {pdf_path}")


def main() -> None:
    pagedjs_cli = shutil.which("pagedjs-cli")
    if pagedjs_cli is None:
        sys.exit(
            "pagedjs-cli not found on PATH. Install it first:\n"
            "  npm install puppeteer-core pagedjs-cli\n"
            "then run this script from an environment where node_modules/.bin is on PATH."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(SRC_DIR, "_build")
    os.makedirs(tmp_dir, exist_ok=True)

    paper_html = os.path.join(tmp_dir, "paper_print_final.html")
    patent_html = os.path.join(tmp_dir, "patent_print_final.html")
    companion_html = os.path.join(tmp_dir, "companion_print_final.html")
    embed_images(os.path.join(SRC_DIR, "paper_print.html"), PAPER_IMAGES, paper_html)
    embed_images(os.path.join(SRC_DIR, "patent_print.html"), PATENT_IMAGES, patent_html)
    embed_images(os.path.join(SRC_DIR, "companion_print.html"), COMPANION_IMAGES, companion_html)

    render(paper_html, os.path.join(OUT_DIR, "emergent_attention_paper.pdf"), pagedjs_cli)
    render(patent_html, os.path.join(OUT_DIR, "patent_disclosure.pdf"), pagedjs_cli)
    render(companion_html, os.path.join(OUT_DIR, "emergent_attention_companion.pdf"), pagedjs_cli)


if __name__ == "__main__":
    main()
