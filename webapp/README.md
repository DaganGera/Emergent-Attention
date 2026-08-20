# Emergent Attention — demo website

A local website that runs the actual trained Hybrid NCA-ViT model (no baseline
comparison, no re-training) on an uploaded or sample image, explains the
decision with a Grad-CAM visual highlight, and writes a plain-language
explanation via a hosted language model. Two modes: general photos
(CIFAR-100, 100 classes) and medical ultrasound (BUSI, 3 diagnostic classes).

## Run it

```bash
pip install -r requirements.txt        # if not already installed (torch, timm, ...)
pip install -r webapp/requirements.txt # fastapi, uvicorn, openai, python-multipart
cp webapp/backend/.env.example webapp/backend/.env   # then fill in a real key
python -m uvicorn webapp.backend.app:app --port 8000
```

Open **http://localhost:8000**.

Everything runs from the repository root — the backend finds the checkpoints
at `checkpoints/nca_vit_hybrid/{cifar100,busi}/seed42/best.pt` and the CIFAR-100
class names at `data/cifar-100-python/meta` relative to the repo, not to
`webapp/`.

## What's in here

```
webapp/
  backend/
    app.py         FastAPI app: serves the frontend + /api/predict, /api/samples, /api/reason
    inference.py   loads both checkpoints, runs the model, computes Grad-CAM
    reasoning.py   sends the decision to a hosted LLM, gets back plain-language text
    samples.py     serves the pre-built sample gallery (static/samples/)
    static/        sample images + the workflow diagram, served at /static/...
    .env.example   copy to .env; holds the reasoning-engine API key (never committed)
  frontend/
    index.html, style.css, script.js   no build step, no framework, no CDN dependency
  requirements.txt  extra deps beyond the main project requirements.txt
```

`scripts/build_web_samples.py` regenerates the sample gallery (12 CIFAR-100
images across random classes, 9 BUSI images, 3 per diagnostic class) from the
datasets already in `data/`. Re-run it if you want a different random sample
of thumbnails; it's deterministic (seeded), so re-running without changing
anything reproduces the same set.

## The reasoning engine

`webapp/backend/reasoning.py` talks to any OpenAI-compatible chat-completion
API — set three environment variables in `.env` and nothing else changes:

| Variable | What it's for |
|---|---|
| `REASONING_BASE_URL` | the API's base URL |
| `REASONING_API_KEY` | your key |
| `REASONING_MODEL` | model name on that provider |

Built and tested against NVIDIA's hosted NIM API (`integrate.api.nvidia.com`);
switching to Groq, OpenRouter, or a local server is a `.env` edit, not a code
change — see the commented-out example in `.env.example`. If no key is set,
the site still works: the classification and Grad-CAM heatmap are unaffected,
and the reasoning panel says plainly that it isn't configured instead of
failing silently.

## Honesty notes baked into the demo

- **CIFAR-100 mode on your own photos is illustrative, not benchmark-accurate.**
  The model was trained and evaluated on CIFAR-100's native 32×32 frames
  upsampled to 224×224; an arbitrary photo you drop in gets resized/cropped
  the same way a real photo would for this model, but the measured 82%
  figure is from the actual CIFAR-100 test set, not from photos like yours.
- **The BUSI mode carries an explicit on-screen disclaimer** ("demonstration
  only... never be used for real diagnosis") whenever ultrasound mode is
  selected. This is a research artifact, not a cleared medical device.
- **The malignant sample images are a real, known weak point** — the model's
  malignant recall is lower than its benign recall at this seed (documented
  in `paper.md` §5.7 and `scripts/gradcam_busi.py`), so don't be surprised
  if one of the malignant samples gets misclassified. That's the honest,
  reproducible result, left in rather than swapped for an easier example.

## What I couldn't verify myself

Every backend endpoint was tested with real HTTP requests against the running
server (`/`, `/style.css`, `/script.js`, `/api/samples/*`, `/api/predict` for
both domains, `/api/reason` against the live NVIDIA API) and all returned
correct, real results — not mocked. `script.js` was syntax-checked with
`node --check`. I do not have a browser available in this environment, so I
have not visually confirmed the page renders as designed — only that every
piece of markup, CSS, and JS is internally consistent on inspection. Open it
yourself first; if anything looks off, tell me what you see and I'll fix it.
