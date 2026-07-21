# PLAN — HOME PC (RTX 4060)

> Runbook for the **home PC**. A separate runbook, `PLAN_LAB_PC.md`, runs in parallel on the AI-lab
> PC. This file is self-contained — you can hand it to a fresh Claude Code instance here.

## Context
You are on the home PC (`d:\Research\Emergent-Attention`, RTX 4060, torch 2.5.1+cu121 already
installed and working). Two jobs:
1. Push the current code + both plan files to GitHub so the lab PC can clone them.
2. Train the two missing baselines (ViT-Tiny, Swin-Tiny), unattended, at the exact existing recipe.

The lab PC runs a *different* set of experiments (the split-ratio ablation). **Do not run those here.**

## INVARIANTS — do not change (keeps new models comparable to the existing ones)
Existing checkpoints (DeiT 73.58, Pure-NCA 73.99, Hybrid 81.91) used one fixed recipe. Reuse it:
- `configs/training/default.yaml` **untouched**: 300 epochs, AdamW lr 5e-4, wd 0.05, cosine + 10-ep
  warmup, label smoothing 0.1, Mixup/CutMix, AMP, EMA 0.9999, **seed 42**.
- `data.batch_size=64`.
- The **only** thing that changes per run is the model name. Nothing else.

---

## Step 1 — Push code safely (do this FIRST; the lab PC waits on it)
Local code has uncommitted changes not on GitHub. **Never `git add -A`** — `.kaggle/kaggle.json` is a
live API secret and would be published.
```powershell
cd d:\Research\Emergent-Attention
git add src configs scripts tests requirements.txt paper.md implementation.md PLAN_HOME_PC.md PLAN_LAB_PC.md
git status                      # CONFIRM: no ".kaggle" / "kaggle.json" line appears
git commit -m "Sync code + machine runbooks before AI-lab training"
git push origin main
```
If `kaggle.json` ever appears staged: `git reset .kaggle` then re-commit.
Then tell the person on the lab PC they can clone.

## Step 2 — Train ViT-Tiny baseline (300 ep, unattended)
```powershell
$env:WANDB_MODE="disabled"
cd d:\Research\Emergent-Attention
python scripts/train.py model=vit_tiny_baseline data=cifar100 | Tee-Object -FilePath outputs\vit_tiny.log
```
Auto-resumes from `checkpoints/vit_tiny_patch16_224/cifar100/seed42/latest.pt` if interrupted — just
re-run the same command.

## Step 3 — Train Swin-Tiny baseline (after Step 2 finishes)
```powershell
python scripts/train.py model=swin_tiny_baseline data=cifar100 | Tee-Object -FilePath outputs\swin_tiny.log
```

## Step 4 — Collect the numbers
```powershell
python -c "import torch,glob; [print(p, round(torch.load(p,map_location='cpu',weights_only=False)['best_acc'],2)) for p in glob.glob('checkpoints/*_tiny*/cifar100/seed42/latest.pt')]"
```
Record ViT-Tiny and Swin-Tiny best top-1 for the main results table (alongside DeiT 73.58,
Pure-NCA 73.99, Hybrid 81.91).

## Optional later — finish any unfinished LAB runs here
When lab access ends, copy an unfinished lab run folder (from the lab's Drive backup) into
`d:\Research\Emergent-Attention\outputs\exp\<tag>\`, then run that run's exact command (from
`PLAN_LAB_PC.md` Step 8) on this PC; it resumes from `latest.pt` and finishes. Nothing is thrown away.

## Verification
- `git push` succeeds; GitHub shows `PLAN_LAB_PC.md`.
- Baseline epoch lines stream into the `.log`; `nvidia-smi` shows python on the 4060.
- Re-running a command prints `Resumed from epoch N, best_acc=…`.

## Estimated time (RTX 4060)
- Each timm baseline, 300 ep: ~6–9 h. Both run unattended overnight (~1–2 nights total).
