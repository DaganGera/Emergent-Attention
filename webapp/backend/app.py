"""
FastAPI backend for the Emergent Attention demo website.

Serves the static frontend, runs the trained Hybrid NCA-ViT model on an
uploaded or sample image (inference.py), attaches a Grad-CAM visual
explanation, and asks a hosted language model for a plain-language
explanation of the result (reasoning.py).

Run with:
    python -m uvicorn webapp.backend.app:app --reload --port 8000
(from the repository root), then open http://localhost:8000
"""

import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

_BACKEND_DIR = Path(__file__).parent
_WEBAPP_DIR = _BACKEND_DIR.parent
_FRONTEND_DIR = _WEBAPP_DIR / "frontend"

# Makes `import inference` etc. resolve regardless of whether this file was
# loaded as a script or via the dotted "webapp.backend.app" module path.
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, '#' comments) -- avoids an
    extra dependency for something this small. Does not override variables
    already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(_BACKEND_DIR / ".env")

import inference  # noqa: E402 -- import after env vars are loaded
import reasoning  # noqa: E402
import samples  # noqa: E402

app = FastAPI(title="Emergent Attention — Explainable Classifier Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_BACKEND_DIR / "static")), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/style.css")
def style() -> FileResponse:
    return FileResponse(str(_FRONTEND_DIR / "style.css"))


@app.get("/script.js")
def script() -> FileResponse:
    return FileResponse(str(_FRONTEND_DIR / "script.js"))


@app.get("/api/samples/{domain}")
def get_samples(domain: str) -> list[dict]:
    if domain not in ("cifar100", "busbra"):
        raise HTTPException(404, "unknown domain")
    return samples.list_samples(domain)


@app.post("/api/predict")
async def predict(
    domain: str = Form(...),
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
) -> dict:
    if domain not in ("cifar100", "busbra"):
        raise HTTPException(400, "domain must be 'cifar100' or 'busbra'")

    if file is not None:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw))
    elif sample_id:
        try:
            path = samples.resolve_sample_path(domain, sample_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        image = Image.open(path)
    else:
        raise HTTPException(400, "provide either a file upload or a sample_id")

    try:
        return inference.predict(domain, image)
    except FileNotFoundError as exc:
        raise HTTPException(
            500,
            f"Model checkpoint not found ({exc}). Make sure checkpoints/nca_vit_hybrid/"
            f"{domain}/seed42/best.pt exists.",
        ) from exc


class TopPrediction(BaseModel):
    label: str
    confidence: float


class ReasonRequest(BaseModel):
    domain: str
    predicted_label: str
    confidence: float
    top_predictions: list[TopPrediction]


@app.post("/api/reason")
def reason(req: ReasonRequest) -> dict:
    text = reasoning.explain(
        req.domain,
        req.predicted_label,
        req.confidence,
        [p.model_dump() for p in req.top_predictions],
    )
    return {"explanation": text}
