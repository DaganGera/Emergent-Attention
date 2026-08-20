# Emergent Attention — demo website

A local website that runs the actual trained Hybrid NCA-ViT model (no baseline
comparison, no re-training at request time) on an uploaded or sample image,
explains the decision with a Grad-CAM visual highlight, and writes a
plain-language explanation via a hosted language model. Two modes: general
photos (CIFAR-100, 100 classes) and medical ultrasound (BUS-BRA, binary
benign/malignant).

## Run it

```bash
pip install -r requirements.txt        # if not already installed (torch, timm, ...)
pip install -r webapp/requirements.txt # fastapi, uvicorn, openai, python-multipart
cp webapp/backend/.env.example webapp/backend/.env   # then fill in a real key
python -m uvicorn webapp.backend.app:app --port 8000
```

Open **http://localhost:8000**.

Everything runs from the repository root — the backend finds the CIFAR-100
checkpoint at `checkpoints/nca_vit_hybrid/cifar100/seed42/best.pt`, the
ultrasound checkpoint at
`outputs/exp/busbra_hybrid/checkpoints/nca_vit_hybrid/busbra/seed42/best.pt`,
and the CIFAR-100 class names at `data/cifar-100-python/meta`, all relative
to the repo, not to `webapp/`.

## Why ultrasound mode runs on BUS-BRA, not BUSI

The first version of this demo used BUSI (780 images, 3 classes: benign /
malignant / normal). Two real problems came up, in order:

1. **A checkpoint bug.** `checkpoints/nca_vit_hybrid/busi/seed42/` was trained
   with the CIFAR-100 recipe (Mixup/CutMix, unweighted cross-entropy) on a
   severely imbalanced 3-class set (56% "benign") and collapsed to predicting
   "benign" almost regardless of input — 0/26 recall on "normal", verified by
   evaluating it directly. Fixed by pointing at the corrected-recipe checkpoint
   (`outputs/exp/busi_v2_hybrid/`, no Mixup/CutMix, class-weighted CE).
2. **A ceiling problem.** Even the corrected single checkpoint topped out
   around 47% balanced accuracy (chance = 33%). Ensembling all 3 available
   seeds (42/123/7 — each weak on a *different* class) got that to 65.5%, but
   BUSI itself has two structural limits no amount of ensembling fixes: its
   public release has **no patient IDs**, so no patient-level split is
   constructible (paper.md §5.7's own stated caveat), and 780 images across
   3 severely-imbalanced classes is a small, noisy evaluation set.

**BUS-BRA** (Gómez-Flores, Gregorio-Calas, Pereira, *Medical Physics* 2024;
CC BY 4.0; via Kaggle `orvile/bus-bra-a-breast-ultrasound-dataset`) fixes both:
1,875 images from 1,064 patients, biopsy-confirmed pathology for most cases,
and a `Case` column that *is* a real patient ID — so `src/data/busbra.py`
splits train/val by patient, not by image; no patient appears on both sides.
Binary benign/malignant at 68/32 (vs. BUSI's 56/27/17 three-way) is also a
less severe imbalance.

Trained fresh, same recipe as BUSI (no Mixup/CutMix, class-weighted CE, 300
epochs, seed 42), on the patient-level split, against a same-recipe DeiT-Tiny
baseline trained identically:

| Metric | DeiT-Tiny (baseline) | Hybrid NCA-ViT (this system) |
|---|---:|---:|
| Balanced accuracy | 59.47% | **74.93%** |
| Macro-F1 | 49.57% | **73.17%** |

The baseline doesn't just do worse — it collapses toward predicting
"malignant" indiscriminately (88.9% malignant recall but only 30.0% benign
recall). The hybrid doesn't (73.7% benign recall, 76.2% malignant recall).
Single seed so far (42); see `results/nca_vit_hybrid_busbra.json` and
`results/deit_tiny_busbra.json` for the full confusion matrices
(`scripts/evaluate_busbra.py` produces both).

## What's in here

```
webapp/
  backend/
    app.py         FastAPI app: serves the frontend + /api/predict, /api/samples, /api/reason
    inference.py   loads the checkpoints, runs the model, computes Grad-CAM
    reasoning.py   sends the decision to a hosted LLM, gets back plain-language text
    samples.py     serves the pre-built sample gallery (static/samples/)
    static/        sample images + the workflow diagram, served at /static/...
    .env.example   copy to .env; holds the reasoning-engine API key (never committed)
  frontend/
    index.html, style.css, script.js   no build step, no framework, no CDN dependency
  requirements.txt  extra deps beyond the main project requirements.txt
```

`scripts/build_web_samples.py` regenerates the sample gallery (12 CIFAR-100
images across random classes, 8 BUS-BRA images, 4 per class) from the datasets
already in `data/`. Re-run it if you want a different random sample of
thumbnails; it's deterministic (seeded), so re-running without changing
anything reproduces the same set.

Training the ultrasound model yourself (GPU recommended — this repo's own
run used an RTX 4060, ~2 hours for 300 epochs at this dataset size):

```bash
python scripts/train.py model=nca_vit_hybrid data=busbra model.num_classes=2 \
  training.augmentation.mixup_alpha=0 training.augmentation.cutmix_alpha=0 \
  training.training.class_weighted=true \
  hydra.job.chdir=true hydra.run.dir=outputs/exp/busbra_hybrid
```

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
- **The ultrasound mode carries an explicit on-screen disclaimer** ("demonstration
  only... never be used for real diagnosis") whenever it's selected. This is
  a research artifact, not a cleared medical device.
- **74.9% balanced accuracy still means roughly 1 in 4 wrong.** Don't be
  surprised by an occasional wrong call on the sample gallery — that's the
  honest, measured, reproducible result on a genuinely hard task, not
  swapped for easier examples. It's a large, verified improvement over the
  conventional baseline (59.5%), not a claim of solved diagnosis.
- **Single seed (42) so far** for the BUS-BRA numbers above — BUSI's headline
  numbers in `paper.md` §5.7 are averaged over 3 seeds; that hasn't been
  redone yet for BUS-BRA. Worth doing before citing this in anything more
  formal than a demo.

## What I couldn't verify myself

Every backend endpoint was tested with real HTTP requests against the running
server (`/`, `/style.css`, `/script.js`, `/api/samples/*`, `/api/predict` for
both domains, `/api/reason` against the live NVIDIA API) and all returned
correct, real results — not mocked. `script.js` was syntax-checked with
`node --check`. I do not have a browser available in this environment, so I
have not visually confirmed the page renders as designed — only that every
piece of markup, CSS, and JS is internally consistent on inspection. Open it
yourself first; if anything looks off, tell me what you see and I'll fix it.
