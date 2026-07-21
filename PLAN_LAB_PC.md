# PLAN — LAB PC (RTX 5060 Ti, Blackwell)

> Runbook for the **AI-lab PC**. A separate runbook, `PLAN_HOME_PC.md`, runs in parallel on the home
> PC (it trains the baselines). This file is self-contained — hand it to a fresh Claude Code instance
> on the lab PC. You get this file by cloning the repo in Step 5.

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
**If all three finish and time remains — NCA depth K ablation** (you already have K=4):
```powershell
python scripts/train.py model=nca_vit_tiny model.nca_steps=1 data=cifar100 data.data_root=C:/Emergent-Attention/data data.batch_size=64 hydra.job.chdir=true hydra.run.dir=outputs/exp/K1 | Tee-Object outputs\K1.log
python scripts/train.py model=nca_vit_tiny model.nca_steps=2 data=cifar100 data.data_root=C:/Emergent-Attention/data data.batch_size=64 hydra.job.chdir=true hydra.run.dir=outputs/exp/K2 | Tee-Object outputs\K2.log
python scripts/train.py model=nca_vit_tiny model.nca_steps=8 data=cifar100 data.data_root=C:/Emergent-Attention/data data.batch_size=64 hydra.job.chdir=true hydra.run.dir=outputs/exp/K8 | Tee-Object outputs\K8.log
```
Monitor in a 2nd tab: `nvidia-smi` and `Get-Content outputs\split_0_12.log -Wait -Tail 5`. You may
disconnect Chrome Remote Desktop while training — the run continues until 3 PM logout.

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
