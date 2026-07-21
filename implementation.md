# IMPLEMENTATION.md — Emergent Attention: NCA-Based Vision Transformer

## 1. Core Novel Idea: Emergent Attention Mechanism

**Emergent Attention** replaces standard O(n²) multi-head self-attention with **O(n·K) iterated local NCA (Neural Cellular Automata) message passing** in vision transformers. The innovation is that global receptive fields **emerge** through successive local convolution iterations, not quadratic attention matrices.

### Key Algorithmic Differences

**Standard Multi-Head Attention:**
- O(n²) attention matrix `Attention(Q,K,V) = softmax(QK^T/√d)V`
- All-to-all token mixing in a single step

**NCA-Attention:**
- **Per-cell perception:** Local 3×3 depthwise convolution with fixed (optionally learnable) kernels
  - Sobel-X, Sobel-Y (gradient detection)
  - Identity (spatial locality)
  - Laplacian (available, unused by default)
- **Update rule:** Cell state `s_t` evolves as:
  ```
  δ_t       = MLP(concat([Sobel-X(s_t), Sobel-Y(s_t), Identity(s_t)]))
  s_{t+1}   = s_t + Bernoulli(p=0.5) * δ_t     [training]
  s_{t+1}   = s_t + δ_t                        [inference]
  ```
- **K iterations** (default K=4) grow the receptive field additively rather than all-at-once.

### Code Location & Formulation

**File:** `src/models/nca_attention.py`

**Core class:** `NCAAttention(nn.Module)`

Key method (lines 92–126):
```python
def _nca_step(self, grid: torch.Tensor) -> torch.Tensor:
    """Single NCA update: grid (B,D,Hp,Wp) -> updated grid (B,D,Hp,Wp)"""
    perc  = self.perception(grid)                      # (B, D*M, Hp, Wp), M=3 filters
    perc  = rearrange(perc, "b dm hp wp -> b hp wp dm")
    h     = self.act(self.linear1(perc))               # (B, Hp, Wp, H_mlp)
    delta = self.linear2(h)                            # (B, Hp, Wp, D)
    delta = rearrange(delta, "b hp wp d -> b d hp wp")

    if self.training and self.stochastic_rate < 1.0:
        mask = torch.bernoulli(torch.full((B, 1, Hp, Wp),
                                          self.stochastic_rate,
                                          device=grid.device))
        grid = grid + mask * delta
    else:
        grid = grid + delta
    return grid
```

**Perception module:** `src/utils/perception.py` (lines 20–101)

Depthwise grouped convolution; kernel shapes:
```python
FILTER_REGISTRY = {
    "sobel_x":  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "sobel_y":  [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}
```
Weight tensor shape: `(D*M, 1, 3, 3)` with `groups=D` → per-channel independent convolution.

**Forward flow (lines 128–159):**
1. Separate CLS token (unchanged throughout block).
2. Pre-norm patch tokens.
3. Sequence → grid: `rearrange(patches, 'b (hp wp) d -> b d hp wp')`.
4. Run K iterations of `_nca_step`.
5. Compute accumulated delta: `grid_final − grid_initial`.
6. Return delta; residual added by block wrapper.

---

## 2. Architecture

### NCA-ViT Model (pure NCA)

**File:** `src/models/nca_vit.py` — `NCAViT`, lines 83–242.

```
Input: (B, 3, 224, 224)
  ↓  Patch Embed: Conv2d(3, 192, kernel=16, stride=16)   → (B, 192, 14, 14)
  ↓  Flatten                                             → (B, 196, 192)
  ↓  Prepend CLS token                                   → (B, 197, 192)
  ↓  + Learnable Positional Embedding (1, 197, 192)
  ↓
  12× NCAViTBlock:
        ├─ NCAAttention(dim=192, nca_steps=4, hidden_dim=384,
        │                grid_size=(14,14), stochastic_rate=0.5)
        ├─ MLP(192, mlp_ratio=4.0 → hidden=768)
        └─ DropPath (linearly scaled 0 → 0.1 across depth)
  ↓  LayerNorm(192)
  ↓  GAP over patch tokens (CLS excluded)
  ↓  Linear(192, 100)
Output: (B, 100) logits
```

**Dimensions:**
- `embed_dim = 192` (D)
- `depth = 12`
- `nca_steps = 4` (K)
- `nca_hidden_dim = 384` (2 × D)
- `grid_size = (14, 14)` (Hp, Wp for 224/16)
- **Total parameters:** ~7.2 M

**NCAViTBlock (lines 38–80):**
```python
class NCAViTBlock(nn.Module):
    def forward(self, x):
        x = x + DropPath(NCAAttention(x))              # pre-norm inside NCAAttention
        x = x + DropPath(MLP(LayerNorm(x)))
        return x
```

### Hybrid NCA-ViT

**File:** `src/models/hybrid_nca_vit.py`

Splits 12 blocks: first 6 NCA blocks, last 6 standard Multi-Head Self-Attention (MHSA).

```python
class HybridNCAViT:
    nca_blocks  = ModuleList([NCAViTBlock(...)     for _ in range(6)])
    attn_blocks = ModuleList([AttentionBlock(...)  for _ in range(6)])
```

**AttentionBlock (lines 15–51):**
```python
class AttentionBlock(nn.Module):
    def __init__(self, dim=192, num_heads=3, mlp_ratio=4.0, ...):
        self.norm1 = LayerNorm(dim)
        self.attn  = MultiheadAttention(dim, num_heads, ...)
        self.norm2 = LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, ...)
```

**HybridNCAViT parameters:** ~6.5 M.

### Baseline Models

**File:** `src/models/baselines.py` — wraps timm models:
- `vit_tiny_patch16_224`
- `deit_tiny_patch16_224`
- `swin_tiny_patch4_window7_224`

---

## 3. Training Pipeline

### Configuration Schema

**Master config:** `configs/train.yaml`
```yaml
defaults:
  - model:    nca_vit_tiny
  - data:     cifar100
  - training: default
```

**Training hyperparameters** (`configs/training/default.yaml`):

| Parameter            | Value         | Notes                               |
|----------------------|---------------|-------------------------------------|
| Optimizer            | AdamW         |                                     |
| lr                   | 5.0e-4        | Base learning rate                  |
| weight_decay         | 0.05          | L2 regularization                   |
| betas                | [0.9, 0.999]  | Adam momentum                       |
| Scheduler            | Cosine+Warmup |                                     |
| epochs               | 300           | (overridable per run)               |
| warmup_epochs        | 10            | Linear 0 → lr                       |
| min_lr               | 1.0e-6        | Cosine floor                        |
| label_smoothing      | 0.1           | Applied in mixup / CE               |
| mixup_alpha          | 0.8           | Beta distribution                   |
| cutmix_alpha         | 1.0           | Beta distribution                   |
| mixup_prob           | 0.5           | Per-batch                           |
| cutmix_prob          | 0.5           | Per-batch                           |
| rand_augment         | magnitude=9   | RandAugment                         |
| random_erasing       | 0.25          | Probability per image               |
| grad_clip            | 1.0           | Max gradient norm                   |
| amp                  | true          | Mixed precision (CUDA)              |
| ema                  | true          | Exponential moving average          |
| ema_decay            | 0.9999        | EMA momentum                        |
| seed                 | 42            | Reproducibility                     |
| checkpoint_freq      | 10            | Epochs between checkpoints          |

**Data config** (`configs/data/cifar100.yaml`):
```yaml
dataset: cifar100
batch_size: 64
num_workers: 4
pin_memory: true
```

### Training Loop

**File:** `src/training/trainer.py` (278 lines).

```python
class Trainer:
    def train_one_epoch(self, epoch: int) -> dict:
        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            outputs = self.model(images)
            loss    = self.criterion(outputs, targets)

        self.scaler.scale(loss).backward()
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)

        if self.ema_model:
            self.ema_model.update(self.model)

        return {"loss": avg_loss, "acc": acc, "time_s": elapsed}

    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> dict:
        eval_model = self.ema_model.module if use_ema else self.model
        # Top-1 and Top-5 accuracy
        return {"top1": ..., "top5": ...}
```

**Optimizer:** `src/training/optimizer.py`
```python
def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
        eps=cfg.training.optimizer.eps,
    )
```

**Scheduler:** `src/training/scheduler.py` — `CosineWarmupScheduler` (lines 10–51)
```python
def _lr_lambda(self, epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
```

**Entry point:** `scripts/train.py` (Hydra config + reproducible seeding).

---

## 4. Data Pipeline

**File:** `src/data/datasets.py`

**Dataset:** CIFAR-100 (50 000 train / 10 000 validation).

**Train augmentation** (lines 16–25):
```python
RandomCrop(32, padding=4)
  -> RandomHorizontalFlip()
  -> Resize(224, BICUBIC)
  -> RandAugment(num_ops=2, magnitude=9)
  -> ToTensor()
  -> Normalize(mean=[0.5071, 0.4867, 0.4408],
               std =[0.2675, 0.2565, 0.2761])
  -> RandomErasing(p=0.25)
```

**Val augmentation** (lines 28–33):
```python
Resize(224, BICUBIC) -> ToTensor() -> Normalize(...)
```

**DataLoader config:**
- `batch_size`: 64 (override to 128 on RTX 4060)
- `num_workers`: 4
- `pin_memory`: True
- `drop_last`: True (train), False (val)

**Mixup / CutMix:** `src/data/augmentations.py`
```python
def build_mixup(mixup_alpha=0.8, cutmix_alpha=1.0,
                mixup_prob=0.5, cutmix_prob=0.5, ...):
    return timm.data.Mixup(mixup_alpha, cutmix_alpha,
                           prob, switch_prob, label_smoothing=0.1)
```

---

## 5. Experiments & Results

### Models Trained
- `nca_vit_tiny`   (all 12 blocks NCA-based)
- `nca_vit_hybrid` (6 NCA + 6 MHSA)
- `deit_tiny`      (baseline, all MHSA)
- `vit_tiny`       (baseline)
- `swin_tiny`      (baseline)

**Results path:** `results/`

**NCA Hybrid on CIFAR-100:**
```json
{
  "model": "nca_vit_hybrid",
  "top1": 81.91,
  "top5": 95.66,
  "params_M": 6.458,
  "gflops": 2.406,
  "images_per_sec": 585.38,
  "robustness_clean": 81.40625,
  "robustness_fgsm":  28.90625,
  "robustness_pgd20":  1.40625
}
```

**DeiT-Tiny baseline on CIFAR-100:**
```json
{
  "model": "deit_tiny",
  "top1": 73.58,
  "top5": 91.37,
  "params_M": 5.543,
  "gflops": 1.079,
  "images_per_sec": 1672.39,
  "checkpoint": "checkpoints/deit_tiny_patch16_224/cifar100/seed42/best.pt"
}
```

**Headline delta:** **+8.33 % top-1** vs DeiT-Tiny (6.5 M vs 5.5 M params).

### Checkpoints & Logging

```
checkpoints/
├─ nca_vit_hybrid/cifar100/seed42/
│   ├─ best.pt
│   ├─ latest.pt
│   └─ checkpoint_epoch*.pt
└─ deit_tiny_patch16_224/cifar100/seed42/
    └─ [similar]
```

**W&B metrics** (trainer.py lines 150–159):
- `train/loss_step`, `train/grad_norm`, `train/lr`
- `system/gpu_memory_MB`
- `val/top1`, `val/top5`

**Visualization artifacts** (`figures/`):
- `emergence_block0.gif` — layer-wise receptive-field growth animation
- `receptive_field_block0.pdf`, `block2.pdf`, `block5.pdf` — receptive field heatmaps

---

## 6. Tests

**Path:** `tests/`

| File                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `test_nca_attention.py`  | Shape / gradient tests for `NCAAttention`           |
| `test_perception.py`     | Perception kernel correctness                       |
| `test_shapes.py`         | Full forward-pass shape assertions                  |
| `test_training.py`       | Trainer loop, checkpoint, resume logic              |

Example (`test_nca_attention.py`, lines 34–50):
```python
@pytest.fixture
def nca_tiny():
    return NCAAttention(dim=192, nca_steps=4, hidden_dim=384, grid_size=(14, 14))

class TestNCAAttentionShape:
    def test_output_shape_matches_input(self, nca_tiny, sample_input):
        out = nca_tiny(sample_input)
        assert out.shape == sample_input.shape   # (B, 197, 192)
```

Run: `pytest tests/ -v`.

---

## 7. Source Code Inventory (`src/`)

| File                              | Purpose                                  | Key Exports                                                                          |
|-----------------------------------|------------------------------------------|--------------------------------------------------------------------------------------|
| `models/nca_attention.py`         | NCA-based attention block                | `NCAAttention(dim, nca_steps, filter_names, hidden_dim, stochastic_rate, grid_size, learnable_filters)` |
| `models/nca_vit.py`               | Full NCA-ViT model                       | `NCAViT(..., depth=12, nca_steps=4)`, `NCAViTBlock`, `MLP`                           |
| `models/hybrid_nca_vit.py`        | Hybrid NCA + MHSA model                  | `HybridNCAViT(nca_depth=6, attn_depth=6)`, `AttentionBlock`                          |
| `models/baselines.py`             | timm wrapper                             | `create_baseline(name, num_classes, pretrained)`                                     |
| `models/__init__.py`              | Model factory                            | `build_model(cfg)` dispatches `nca_vit_tiny` / `nca_vit_hybrid` / baseline           |
| `utils/perception.py`             | Depthwise conv filters                   | `PerceptionModule(dim, filter_names, learnable)`, `FILTER_REGISTRY`                  |
| `utils/checkpoint.py`             | Save / load / resume                     | `save_checkpoint`, `load_checkpoint`, `save_best_model`                              |
| `utils/wandb_utils.py`            | W&B integration                          | `init_wandb(cfg)`, `finish_run()`                                                    |
| `data/datasets.py`                | CIFAR-100 loaders                        | `build_loaders(cfg)` → `(train_loader, val_loader)`                                  |
| `data/augmentations.py`           | Mixup / CutMix                           | `build_mixup(...)` → `timm.Mixup`                                                    |
| `training/trainer.py`             | Training engine                          | `Trainer(model, train_loader, val_loader, cfg, ...)`                                 |
| `training/optimizer.py`           | AdamW factory                            | `build_optimizer(model, cfg)`                                                        |
| `training/scheduler.py`           | Cosine + warmup LR                       | `CosineWarmupScheduler`, `build_scheduler(...)`                                      |
| `evaluation/flops.py`             | FLOPs counter                            | FLOP utilities                                                                       |
| `evaluation/metrics.py`           | Metric computation                       | Top-1, Top-5, robustness metrics                                                     |
| `evaluation/robustness.py`        | Adversarial robustness                   | FGSM, PGD                                                                            |
| `evaluation/visualization.py`     | Attention / receptive-field plots        | Heatmaps, emergence GIF                                                              |

---

## 8. Potentially Patentable Elements

### Novel Algorithmic Contributions

1. **Local NCA-Based Attention Replacement**
   - *Claim:* Replacing O(n²) multi-head self-attention with **O(n·K) iterated local cellular automata message passing** to achieve global receptive fields through emergent local interactions.
   - *Unique:* Combination of 3×3 perception filters (Sobel-X, Sobel-Y, Identity) with MLP-based cell-state updates and stochastic regularization during training.
   - *Code:* `src/models/nca_attention.py::_nca_step()` (lines 92–126).

2. **Stochastic Cell-State Update**
   - *Claim:* Per-cell Bernoulli masking during training to regularize NCA dynamics:
     ```python
     mask  = torch.bernoulli(torch.full((B, 1, Hp, Wp), stochastic_rate, ...))
     grid  = grid + mask * delta   # training
     grid  = grid + delta          # inference
     ```
   - *Benefit:* Improves generalization; prevents over-reliance on all cells each iteration.

3. **Hybrid NCA-ViT Architecture**
   - *Claim:* Layered composition — early blocks (1–6) use NCA-Attention for local feature learning, later blocks (7–12) use MHSA for global refinement.
   - *Code:* `src/models/hybrid_nca_vit.py` (lines 74–204).
   - *Result:* +8 % accuracy over DeiT-Tiny baseline with comparable parameter count.

4. **CLS Token Exclusion Strategy**
   - *Claim:* CLS token isolated from NCA iterations; classification head uses **GAP over patch tokens**, not CLS.
   - *Code:* `src/models/nca_vit.py::forward_features()` (lines 196–228).
   - *Rationale:* Prevents CLS from becoming an information sink in NCA message passing.

5. **Zero-Initialized NCA Update Layers**
   - *Claim:* Output layer weights and bias zero-initialized so δ ≈ 0 at step 0 → identity mapping at init, stable early gradient flow.
   - *Code:* `src/models/nca_attention.py` (lines 88–90, 192–194):
     ```python
     nn.init.zeros_(self.linear2.weight)
     nn.init.zeros_(self.linear2.bias)
     ```

6. **Learnable Perception Filters (Optional)**
   - *Claim:* Sobel / Laplacian kernels may be set trainable, letting the network adapt its edge-detection basis during training.
   - *Code:* `src/utils/perception.py` (lines 87–90).
   - *Config:* `learnable_filters: true` in `configs/model/nca_vit_hybrid.yaml`.

### Performance Summary

- **+8.33 %** top-1 accuracy vs DeiT-Tiny on CIFAR-100 (81.91 % vs 73.58 %).
- **Parameters:** 6.5 M vs 5.5 M (~1.17× overhead).
- **Throughput:** 585 img/s (NCA-Hybrid) vs 1 672 img/s (DeiT), justified by increased FLOPs from local iterative communication.
- **Robustness:** 81.4 % clean, 28.9 % FGSM, 1.4 % PGD-20.

---

**Codebase size:** 22 Python source files, ~2 500 LOC (excluding tests), 4 model configs, 3 experiment configs. Fully reproducible via Hydra + checkpoint resumption.
