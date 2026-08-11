# PLAN — LAB PC (RTX 5060 Ti, Blackwell)

> Runbook for the **AI-lab PC**. A separate runbook, `PLAN_HOME_PC.md`, runs in parallel on the home
> PC (it trains the baselines). This file is self-contained — hand it to a fresh Claude Code instance
> on the lab PC. You get this file by cloning the repo in Step 5.

## STATUS AS OF 2026-08-07 — read this first

**Steps 1–8a/8b/8c (the split-ratio ablation) are DONE.** Results, with a second seed added for
variance (seed 123), are already written into `paper.md` §5.4:

| NCA:Attn split | seed 42 | seed 123 |
|---|---:|---:|
| 0:12 (pure attention) | 76.33% | 77.04% |
| 3:9 | 80.92% | 81.26% |
| **6:6 (Hybrid)** | 81.29–81.91% | **81.59%** |
| 9:3 | 80.69% | 81.19% |
| 12:0 (pure NCA) | 73.99% | 73.52% |

6:6 wins on both seeds — the paper's central "depth-ordering matters" claim is now fully backed,
not pending. This has already been committed and pushed to `origin/main` (commit `f9e15a0`).

**Also done since this plan was written:** PlantVillage dataset support was added to
`src/data/plantvillage.py` + `configs/data/plantvillage.yaml` and is in `origin/main` too.

**Your next task is Step 13 — the data-efficiency sweep. Go there now. Step 12 (K ablation) is
DEFERRED** — it is a paper-completeness footnote, whereas Step 13 produces the central figure for
both the paper and the patent filing. Do Step 12 only if Step 13 finishes and lab time remains.

Steps 1–7 (environment, GPU, clone) you've already done once; skip straight to the `git pull` in
Step 13a if this machine still has the conda env from before.

---

## Context
Fresh AI-lab PC: **RTX 5060 Ti (Blackwell / sm_120)**, Core Ultra 9, 32 GB RAM, admin rights,
internet via Chrome Remote Desktop. Access ~11 AM–3 PM; the machine is **shut down / logged out at
3 PM daily**, so every job must be kill-safe and resumable. Your job: set up the environment, then
run the **split-ratio ablation** — the experiment that defends the paper's novelty. **Do not train
the baselines here; the home PC does those.**

Kill-safety: `src/training/trainer.py` writes `latest.pt` every epoch and `scripts/train.py`
auto-resumes from it, so a 3 PM shutdown costs <1 epoch and re-running the same command continues.

## INVARIANTS — do not change (keeps new models comparable to the existing ones)
- `configs/training/default.yaml` **untouched**: 300 epochs, seed 42, and the full recipe.
- `data.batch_size=64` — **do NOT raise it even though 16 GB allows it**; a different batch size
  changes optimization and breaks comparability with the existing 6:6 (81.9) and 12:0 (74.0) points.
- The **only** override per run is `model.nca_depth`/`model.attn_depth` (or `model.nca_steps`).

## Why these runs (context)
Existing points: 6 NCA : 6 attn = 81.9 %, 12 NCA : 0 attn = 74.0 %. This ablation adds
**0:12 (pure attention, same skeleton), 3:9, 9:3** to complete the curve. If 0:12 lands below 6:6, it
proves NCA-early/attention-late beats both pure attention and pure NCA — the core novelty vs AdaNCA.

---

## Step 1 — Confirm GPU
```powershell
nvidia-smi
```
Note VRAM and driver (must be ≥ 570 for Blackwell).

## Step 2 — Install tools (admin), then CLOSE and REOPEN PowerShell
```powershell
winget install -e --id Anaconda.Miniconda3
winget install -e --id Git.Git
```

## Step 3 — Environment
```powershell
conda create -n emergent python=3.11 -y
conda activate emergent
```

## Step 4 — PyTorch for Blackwell (CRITICAL — old torch won't use this GPU)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); a=torch.randn(4096,4096,device='cuda'); print(float((a@a).sum()))"
```
Must print `True` + `NVIDIA GeForce RTX 5060 Ti` + a number. If it errors with `sm_120` / `no kernel
image`, use the nightly and re-verify:
```powershell
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
**Do not proceed until this passes.**

## Step 5 — Clone repo + install deps
```powershell
cd C:\
git clone https://github.com/DaganGera/Emergent-Attention.git
cd C:\Emergent-Attention
pip install einops hydra-core omegaconf timm fvcore torchattacks lpips scikit-learn matplotlib seaborn pandas wandb pytest
$env:WANDB_MODE="disabled"
```
(Installing torch first, in Step 4, means these won't downgrade it.)

## Step 6 — Disk-persistence test (decides your morning routine)
```powershell
"persist test $(Get-Date)" | Out-File C:\Emergent-Attention\PERSIST_TEST.txt
```
Check tomorrow (Step 11). Survives → resume is automatic. Gone → the lab wipes on logout; restore
from the Drive backup each morning.

## Step 7 — Calibrate (download CIFAR + measure epoch time, 2 epochs)
```powershell
python scripts/train.py model=nca_vit_hybrid data=cifar100 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  training.training.epochs=2 training.scheduler.epochs=2 `
  hydra.job.chdir=true hydra.run.dir=outputs/_probe
```
Read the `(N s)` per epoch. 300-epoch run time ≈ N × 300 / 3600 hours. (If the loader hangs on
Windows, add `data.num_workers=0`.) Epochs are overridden **only** for this calibration; the real
runs below use the full 300.

## Step 8 — Split-ratio ablation (run ONE at a time, in this order)
Each isolates its checkpoints via `hydra.run.dir` and shares one CIFAR download via the absolute
`data_root`. **To resume a killed run, re-run its exact command.**

**8a — 0 NCA : 12 attention (most decisive):**
```powershell
$env:WANDB_MODE="disabled"; cd C:\Emergent-Attention
python scripts/train.py model=nca_vit_hybrid model.nca_depth=0 model.attn_depth=12 `
  data=cifar100 data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/split_0_12 `
  | Tee-Object -FilePath outputs\split_0_12.log
```
**8b — 3 NCA : 9 attention:**
```powershell
python scripts/train.py model=nca_vit_hybrid model.nca_depth=3 model.attn_depth=9 `
  data=cifar100 data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/split_3_9 `
  | Tee-Object -FilePath outputs\split_3_9.log
```
**8c — 9 NCA : 3 attention:**
```powershell
python scripts/train.py model=nca_vit_hybrid model.nca_depth=9 model.attn_depth=3 `
  data=cifar100 data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/split_9_3 `
  | Tee-Object -FilePath outputs\split_9_3.log
```
Monitor in a 2nd tab: `nvidia-smi` and `Get-Content outputs\split_0_12.log -Wait -Tail 5`. You may
disconnect Chrome Remote Desktop while training — the run continues until 3 PM logout.

*(This step is DONE — see the STATUS banner at the top of this file. Skip to Step 12.)*

## Step 9 — Before 3 PM every day: back up (insurance against disk wipe)
```powershell
cd C:\Emergent-Attention
Compress-Archive -Path outputs\exp, outputs\*.log -DestinationPath C:\lab_backup_$(Get-Date -Format yyyyMMdd_HHmm).zip -Force
```
Upload the zip to your Google Drive via the browser in the remote session.

## Step 10 — Collect the numbers (each run's best EMA val top-1)
```powershell
python -c "import torch,glob; [print(p, round(torch.load(p,map_location='cpu',weights_only=False)['best_acc'],2)) for p in glob.glob('outputs/exp/*/checkpoints/**/latest.pt',recursive=True)]"
```
Report the split-ratio table: 0:12, 3:9, 6:6 (=81.9 have), 9:3, 12:0 (=74.0 have).

## Step 11 — Next morning
```powershell
Get-Content C:\Emergent-Attention\PERSIST_TEST.txt
```
Exists → re-run the Step 8 command that was in progress to resume. Gone → download + extract
yesterday's Drive zip into `C:\Emergent-Attention\outputs\`, then re-run the same command to resume.

## Verification
- Step 4 prints `True` + the 5060 Ti + a number.
- Training epoch lines stream to the `.log`; `nvidia-smi` shows python using GPU memory.
- Ctrl-C then re-run prints `Resumed from epoch N, best_acc=…`.
- Each `outputs/exp/<tag>/checkpoints/...` tree is separate — runs never overwrite each other.
- Novelty result: 0:12 `best_acc` < 6:6 (81.9) → ordering claim proven.

## Risks & mitigations
- Blackwell/torch mismatch → Step 4 nightly fallback; don't train until verified.
- Disk wipes on logout → Step 9 daily backup + Step 11 restore.
- Batch-size drift → keep 64 (invariants lock).

## Estimated time (RTX 5060 Ti)
- Each split run, 300 ep: ~7–11 h (confirm with Step 7). Lab's ~8h realistically finishes ~1 run
  (start with 0:12); continue the rest on the home PC via the backup + resume.

---

## Step 12 — NCA depth K ablation (the actual next task — start here)

**Why:** `paper.md` §5.4 explicitly lists this as the one remaining un-ablated design choice
("NCA depth K — only K=4 has been run"). K=4 is the value used in every headline model; this fills
in K∈{1,2,8} to show the paper's "iteration is essential" claim is a real curve, not an assertion.

**12a — Sync first, before doing anything else:**
```powershell
cd C:\Emergent-Attention
git pull origin main
```
This pulls the finalized split-ratio results, PlantVillage support, and the current paper — confirm
`git log -1` shows commit `f9e15a0` or later before proceeding, so you're not working from a stale
copy that would redo work already done.

**12b — Same invariants as before**: 300 epochs, seed 42, `batch_size=64`, untouched
`configs/training/default.yaml`. Model is `nca_vit_tiny` (pure NCA-ViT, 12 blocks) — the only
override is `model.nca_steps`.

```powershell
$env:WANDB_MODE="disabled"; cd C:\Emergent-Attention
python scripts/train.py model=nca_vit_tiny model.nca_steps=1 data=cifar100 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/K1 | Tee-Object outputs\K1.log

python scripts/train.py model=nca_vit_tiny model.nca_steps=2 data=cifar100 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/K2 | Tee-Object outputs\K2.log

python scripts/train.py model=nca_vit_tiny model.nca_steps=8 data=cifar100 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/K8 | Tee-Object outputs\K8.log
```
Run one at a time, same resume/backup/kill-safety rules as Step 8–11 above (re-run the same
command to resume after a 3 PM shutdown; back up `outputs/exp/` to Drive before logout each day).

**12c — When all three finish, collect the numbers:**
```powershell
python -c "import torch,glob; [print(p, round(torch.load(p,map_location='cpu',weights_only=False)['best_acc'],2)) for p in glob.glob('outputs/exp/K*/checkpoints/**/latest.pt',recursive=True)]"
```
You'll have K∈{1,2,4,8} (K=4 is the existing 73.99% headline pure-NCA number) — a full depth curve.

**12d — Report back by pushing to git, not just a Drive zip** (this repo already has working push
access — commit `f9e15a0` proves it):
```powershell
git add outputs/K1.log outputs/K2.log outputs/K8.log
git commit -m "Add NCA depth K ablation (K=1,2,8) to complete the depth-iteration curve"
git push origin main
```
Don't commit `outputs/exp/**/checkpoints/` (large binaries, not needed — the log + printed
`best_acc` numbers above are what go in the paper). If you want the checkpoints preserved too, zip
`outputs/exp/K*` and upload to the same Drive folder as before, same as Step 9.

**12e — Tell the home PC.** Once pushed, say so — the K-depth table needs to be added to `paper.md`
§5.4 (replacing "NCA depth K — only K=4 has been run" with the real K∈{1,2,4,8} curve), which should
happen on whichever machine is talking to you next.

---

## Step 13 — Data-efficiency sweep (**THE PRIORITY TASK — start here**)

### Why this run (read before starting)

Every result so far points at the same thing, but none of them measures it directly:

| dataset | train size | Hybrid vs DeiT |
|---|---:|---|
| PlantVillage | ~46,000 (clean, saturated) | ~0 |
| CIFAR-100 | 50,000 | +8.4 |
| PlantDoc | 2,176 (real-world, noisy) | +3.22 |
| MVTec bottle | ~200 | DeiT never converged (45.0%); Hybrid did (71.67%) |

Hypothesis: **attention must learn locality from data; the NCA's fixed Sobel perception already is
locality.** With enough data attention catches up and the advantage disappears; with scarce data it
never gets there. This sweep tests that on one dataset with everything else held constant, by
varying only how much of CIFAR-100 the model is allowed to see.

Expected shape: a large gap at 5–10% that narrows toward 100%. **A flat curve falsifies the
hypothesis** — that is a real outcome, report it as-is, do not tune to rescue it.

### Step 13a — Sync first (a new flag landed; the sweep will not run without it)

```powershell
cd C:\Emergent-Attention
git pull origin main
git log --oneline -3
```

Must show `96cebc7` (or later) — "Add stratified train_fraction subsampling". Verify:

```powershell
python -c "import yaml; c=yaml.safe_load(open('configs/data/cifar100.yaml')); print('train_fraction' in c)"
```
Must print `True`. If not, the pull failed — stop and fix it.

### Step 13b — Invariants (unchanged)

300 epochs · batch_size 64 · seed 42 · `configs/training/default.yaml` untouched. The **only** new
override is `data.train_fraction`. The val split is never subsampled, so every point on the curve is
scored on the identical 10,000-image test set.

`data.subsample_seed=42` is fixed and separate from the training seed: at a given fraction, Hybrid
and DeiT train on the **exact same images**. Do not change it, or the curve measures which subset
got drawn instead of which architecture learns from less data.

### Step 13c — Run these 8 jobs, in this order, one at a time

Order matters: smallest fractions are both the **fastest** and the **most decisive**. If lab time
runs out you still hold the informative left half of the curve.

```powershell
$env:WANDB_MODE="disabled"; cd C:\Emergent-Attention

# --- 5% (2,500 images) — fastest, most decisive ---
python scripts/train.py model=nca_vit_hybrid data=cifar100 data.train_fraction=0.05 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac05_hybrid | Tee-Object outputs\frac05_hybrid.log

python scripts/train.py model=deit_tiny_baseline data=cifar100 data.train_fraction=0.05 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac05_deit | Tee-Object outputs\frac05_deit.log

# --- 10% (5,000 images) ---
python scripts/train.py model=nca_vit_hybrid data=cifar100 data.train_fraction=0.10 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac10_hybrid | Tee-Object outputs\frac10_hybrid.log

python scripts/train.py model=deit_tiny_baseline data=cifar100 data.train_fraction=0.10 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac10_deit | Tee-Object outputs\frac10_deit.log

# --- 25% (12,500 images) ---
python scripts/train.py model=nca_vit_hybrid data=cifar100 data.train_fraction=0.25 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac25_hybrid | Tee-Object outputs\frac25_hybrid.log

python scripts/train.py model=deit_tiny_baseline data=cifar100 data.train_fraction=0.25 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac25_deit | Tee-Object outputs\frac25_deit.log

# --- 50% (25,000 images) ---
python scripts/train.py model=nca_vit_hybrid data=cifar100 data.train_fraction=0.50 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac50_hybrid | Tee-Object outputs\frac50_hybrid.log

python scripts/train.py model=deit_tiny_baseline data=cifar100 data.train_fraction=0.50 `
  data.data_root=C:/Emergent-Attention/data data.batch_size=64 `
  hydra.job.chdir=true hydra.run.dir=outputs/exp/frac50_deit | Tee-Object outputs\frac50_deit.log
```

**100% is already done** — Hybrid 81.29%, DeiT 72.89%. Do not re-run it.

At the start of each run the log must print a line like
`[data] train_fraction=0.05 -> 2500/50000 images (100 classes, stratified, subsample_seed=42)`.
If that line is missing, the flag did not take effect — stop, re-check Step 13a.

Cost: fractions sum to 0.9 per model, so all 8 runs ≈ **1.8 full-length trainings**. The 5% pair
should finish inside a single lab session.

**Expected absolute numbers are low** at small fractions (roughly 20–40% top-1) — 300 epochs over
2,500 images is 20× fewer gradient steps than the full run. That is fine and expected. Both models
get an identical budget; **the gap is the result, not the absolute value.**

Resume after the 3 PM shutdown by re-running that run's exact command — `latest.pt` auto-resumes.

### Step 13d — Collect

```powershell
python -c "import torch,glob,os; [print(os.path.basename(os.path.dirname(p.split('checkpoints')[0].rstrip('/'))) or p, p.split('exp/')[1].split('/')[0], round(torch.load(p,map_location='cpu',weights_only=False)['best_acc'],2)) for p in sorted(glob.glob('outputs/exp/frac*/checkpoints/**/latest.pt',recursive=True))]"
```

Report as a table: fraction | Hybrid | DeiT | gap. Include the known 100% row (81.29 / 72.89 / +8.40).

### Step 13e — Push results back

```powershell
cd C:\Emergent-Attention
git add outputs/frac*.log
git commit -m "Lab: data-efficiency sweep on CIFAR-100 (5/10/25/50%)"
git push origin main
```

Commit **only the `.log` files** — never `outputs/exp/**/checkpoints/`, those are large binaries.
Still do the Step 9 zip backup each day; the logs in git are the durable record of the numbers.

Then tell whoever picks up next: the curve goes into `paper.md` as the data-efficiency figure, and
into the patent draft as the demonstrated technical effect (accuracy retained under scarce labelled
data), which is what §3(k) needs.
