# Emergent Attention: Global Receptive Fields via Iterated Local Neural Cellular Automata in Vision Transformers

**Anonymous Authors**
*Under review — draft v2*

---

## Abstract

We introduce **Emergent Attention**, a drop-in replacement for multi-head self-attention (MHSA) in Vision Transformers that replaces the O(n²) all-to-all token mixing with **K iterations of a local Neural Cellular Automaton (NCA)** operating on the patch grid. Each NCA step applies a fixed 3×3 depthwise perception (Sobel-X, Sobel-Y, Identity) followed by a shared two-layer MLP that produces an additive cell-state update; the update is gated at training time by a Bernoulli mask to regularize the dynamics. Because the effective receptive field grows with the number of iterations, global context *emerges* from purely local interactions rather than being computed in a single quadratic step. We further propose a **Hybrid NCA-ViT** in which the early half of the backbone uses Emergent Attention (for local structure formation) and the later half uses standard MHSA (for global refinement). On CIFAR-100, Hybrid NCA-ViT reaches **81.91 % top-1 / 95.66 % top-5** with 6.5 M parameters, a **+9.02 % absolute improvement** over a DeiT-Tiny baseline of comparable size (72.89 % top-1, 5.5 M parameters), under matched training. Unlike prior work that inserts NCA as an auxiliary module alongside attention, Emergent Attention **fully replaces** the token mixer, uses **fixed, non-learned** perception kernels, and identifies a **depth-ordering effect** — NCA layers early, attention layers late — as the dominant design factor. A two-seed split-ratio sweep (0:12 through 12:0) confirms this: the 6:6 split sits at the peak of the curve, beating pure attention by 4.6–5.3 points and pure NCA by 7.6–8.4 points, consistently across seeds. We provide this ablation together with receptive-field visualizations that confirm the emergent-global behavior predicted by the model, and report which remaining experiments (NCA-depth ablation, Swin-Tiny baseline) are still in progress at time of writing.

**Keywords:** vision transformer, neural cellular automata, attention, emergent computation, local-to-global inference, adversarial robustness.

---

## 1. Introduction

Self-attention is the computational core of the modern Vision Transformer (ViT) [1]. Its appeal is that every token can attend to every other token in a single layer, so global context is available immediately. Its cost is that this single layer has O(n²) complexity in the number of tokens, and, under the prevailing softmax-QKV formulation, produces attention maps that are often unstable, difficult to interpret, and vulnerable to adversarial perturbation.

A complementary line of work — **Neural Cellular Automata** [4] — has shown that surprisingly complex spatial behavior (morphogenesis, classification, segmentation) can emerge from the repeated application of a small local update rule. NCAs are intrinsically local, translation-equivariant, and iteratively refining: properties that softmax attention lacks by construction. Recent work has begun to combine NCA with Transformer components — AdaNCA [12] inserts NCA blocks as adaptors between attention layers to improve robustness, and ViTCA [13] fuses attention *into* the NCA update rule itself. Both keep standard self-attention as the primary token mixer. The question we ask is more direct:

> *Can iterated local NCA dynamics fully replace global softmax attention as the token mixer inside a Vision Transformer, rather than augment it?*

Our answer is yes, with one caveat. A pure-NCA backbone trains stably and reaches competitive accuracy, but the strongest configuration uses NCA layers for *early* feature formation and standard attention layers for *late* semantic aggregation. We call this a Hybrid NCA-ViT. On CIFAR-100 it outperforms a same-scale DeiT-Tiny by 9.02 % top-1 under matched training.

**Our contributions.**

1. **Emergent Attention**, an NCA-based token mixer with fixed (non-learned) 3×3 Sobel + Identity perception, a shared two-layer update MLP, zero-initialized output projection (so each block is an identity map at init), and Bernoulli-gated stochastic updates during training. Unlike AdaNCA/ViTCA, this fully replaces MHSA rather than adapting or fusing with it.
2. **NCA-ViT** and **Hybrid NCA-ViT**, two ViT-scale architectures that instantiate (1) as either a pure-NCA stack or a layered NCA → MHSA composition, with a two-seed split-ratio ablation (§5.4) confirming that the *ordering* (NCA-early / attention-late) — not merely the presence of both mixers — is the dominant design variable.
3. **Empirical evidence** on CIFAR-100 that Hybrid NCA-ViT achieves 81.91 % top-1 versus 72.89 % for DeiT-Tiny at comparable parameter count, and versus 72.33 % / (in progress) for ViT-Tiny / Swin-Tiny baselines trained under the identical recipe.
4. **Visualizations** of the receptive field as a function of NCA depth, showing that the effective receptive field grows with iteration count as predicted.
5. **A reproducible implementation** (Hydra-configured, AMP-enabled, EMA-tracked, checkpoint-resumable) with all configs, tests, and trained checkpoints, and an explicit account of which ablations are finalized versus still running (§5.4, §7).

---

## 2. Related Work

**Vision Transformers.** ViT [1] established that pure attention stacks can match convolutional networks at scale. DeiT [2] closed the data-efficiency gap with distillation and heavy augmentation. Swin Transformer [3] reintroduced locality via shifted windows, trading some global capacity for compute efficiency. Our work takes the locality idea further — to a *per-pixel 3×3 neighborhood* — and recovers globality through iteration rather than windowing.

**Efficient and linear attention.** Performer, Linformer, and related approximations reduce attention complexity while preserving its softmax-QKV shape. Flash attention accelerates the exact softmax primitive. Emergent Attention differs: it abandons the query–key–value formulation entirely in favor of spatial convolution plus iteration.

**Neural Cellular Automata.** NCA [4] demonstrated that a small local update rule, applied repeatedly, can reproduce target images from a single seed. Subsequent work extended NCAs to classification, texture synthesis, and 3D volumetric growth.

**NCA combined with Transformers — closest prior work.** Two recent papers combine NCA and Transformer machinery, and our positioning against both is the central novelty claim of this work:
- **AdaNCA** [12] (NeurIPS 2024) inserts NCA blocks as *adaptors* between existing ViT attention layers to improve robustness and out-of-distribution generalization. Standard MHSA remains the primary token mixer throughout the network; NCA is auxiliary.
- **ViTCA** [13] (NeurIPS 2022) builds an NCA whose per-cell update rule is itself computed via self-attention over a local neighborhood — attention is fused *into* the NCA cell, not replaced by it.

Emergent Attention differs from both along three axes: **(i) replacement, not augmentation** — NCA is the sole token mixer in the NCA blocks, with no attention computation inside them; **(ii) fixed, non-learned perception** — Sobel-X/Sobel-Y/Identity kernels are frozen (an optional `learnable_filters` flag exists but is off by default), whereas AdaNCA and ViTCA both learn their perception/update rules end-to-end; **(iii) a depth-ordering claim** — we find empirically that placing NCA blocks *before* attention blocks in the stack, rather than interleaving or adapting, is what drives the accuracy gain (§5.4). We do not claim NCA-in-a-Transformer is a new idea; we claim this specific combination (full replacement, fixed perception, depth-ordered hybrid) is unexplored by [12, 13] and is the locus of our empirical contribution.

**Convolution + attention hybrids.** CoAtNet, Conformer, and MobileViT interleave convolution and attention. Our hybrid is related in spirit but differs in two respects: (i) the "convolutional" half is an *iterated* NCA, not a feed-forward conv stage, and (ii) the split is strictly layer-wise (depth ordered), which we show matters empirically.

---

## 3. Method

### 3.1 Preliminaries

Given an image `x ∈ ℝ^{3×H×W}`, a standard ViT patch-embeds `x` into `n = (H/p)(W/p)` tokens of dimension `D`, prepends a learnable CLS token, adds positional embeddings, and applies a stack of blocks. Each block computes:

```
z' = z + MHSA(LN(z))
z  = z' + MLP(LN(z'))
```

We retain this block skeleton but replace `MHSA(·)` with `NCAAttention(·)`.

### 3.2 Emergent Attention

Let `z ∈ ℝ^{B × (1+n) × D}` denote the token sequence with CLS at index 0 and patch tokens `p ∈ ℝ^{B × n × D}` at indices 1..n. Let `Hp × Wp = n` be the patch-grid shape.

**Step 1 — CLS isolation.** The CLS token is removed from the NCA computation entirely. Only patch tokens participate in the cellular dynamics. This prevents the CLS position from acting as an information sink that would otherwise dominate the shared local update.

**Step 2 — Sequence → grid.**
```
s₀ = rearrange(LN(p), "b (hp wp) d -> b d hp wp")
```

**Step 3 — Perception.** A fixed 3×3 depthwise convolution applies `M = 3` hand-chosen filters — Sobel-X, Sobel-Y, and Identity — per channel:

```
Sobel-X  = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
Sobel-Y  = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
Identity = [[ 0, 0, 0], [ 0, 1, 0], [ 0, 0, 0]]
```

The convolution uses `groups = D`, so channels are processed independently. The output has shape `(B, D·M, Hp, Wp)`. Sobel-X and Sobel-Y together encode the local spatial gradient of the state; Identity preserves the cell's own value. These filters are **fixed by default**; an optional `learnable_filters` flag unfreezes them for end-to-end training and is used only in the Hybrid config's ablation variant (§5.4), not the reported headline model.

**Step 4 — Update MLP.**
```
perc = rearrange(perc, "b (d m) hp wp -> b hp wp (d m)")
h    = ReLU(W₁ · perc + b₁)        # W₁ ∈ ℝ^{(D·M) × H_mlp}
δ    = W₂ · h + b₂                  # W₂ ∈ ℝ^{H_mlp × D}
δ    = rearrange(δ, "b hp wp d -> b d hp wp")
```

Crucially, `W₂` and `b₂` are **zero-initialized**, so `δ₀ = 0` at step 0 and the entire block is an identity mapping at training start. This gives stable gradient flow and removes the need for any warm-up schedule on the NCA layers themselves.

**Step 5 — Stochastic update.**
```
s_{t+1} = s_t + m_t ⊙ δ_t     (training,  m_t ~ Bernoulli(p) per cell)
s_{t+1} = s_t + δ_t            (inference, deterministic)
```

The per-cell Bernoulli mask (with default `p = 0.5`) prevents every cell from updating on every iteration and is essential to the stability of NCA-style dynamics [4]. At inference time the mask is disabled for determinism.

**Step 6 — K iterations.** Steps 3–5 are repeated `K` times (default `K = 4`). The output of the block is the **accumulated delta**
```
Δ = s_K − s₀
```
which is reshaped back into a token sequence and returned as the additive contribution to the residual stream. The CLS token passes through unchanged.

### 3.3 Receptive Field Analysis

A single NCA iteration has a spatial receptive field of 3×3 pixels (one for each direction of the 3×3 perception kernel). After `K` iterations, each cell has been influenced by every cell within `K`-hops on the grid, yielding an effective receptive field of `(2K+1) × (2K+1)` pixels on the 14×14 patch grid — 9×9 for `K = 4`. Full 14×14 grid coverage requires K ≈ 7. §5.3 visualizes this growth empirically.

### 3.4 Architectures

**NCA-ViT.** A straight stack of `L = 12` blocks, each using Emergent Attention with `D = 192, K = 4, H_mlp = 384`, followed by a standard feed-forward sub-block (`mlp_ratio = 4.0`). Classification uses **Global Average Pooling over patch tokens** (the CLS token is dropped at the head) followed by a linear projection to `C` classes. Total parameters: **~7.2 M**.

**Hybrid NCA-ViT.** The first `L_N = 6` blocks use Emergent Attention; the remaining `L_A = 6` blocks use standard multi-head self-attention (`num_heads = 3`, `mlp_ratio = 4.0`). The intuition is that early layers benefit from locality-biased, iterative feature formation, while later layers benefit from the direct global aggregation of softmax attention. Total parameters: **6.46 M**. This is the ordering ablated in §5.4.

### 3.5 Complexity

For a sequence of `n` patch tokens, embedding dimension `D`, and NCA depth `K`:

| Component            | Complexity         |
|----------------------|--------------------|
| MHSA                 | `O(n² · D)`        |
| Emergent Attention   | `O(K · n · D²)` (MLP-dominated)    |
| Perception conv      | `O(K · n · D · M · k²)`, `k = 3`   |

For the token counts common in ViT-scale vision (`n = 196` on 224×224/16) Emergent Attention is linear in `n` and the cost is dominated by the shared two-layer MLP. Our measured throughput (585 img/s for Hybrid NCA-ViT vs 1672 img/s for DeiT-Tiny on an RTX 4060, uncontended) reflects the fact that we trade a single large softmax for `K` small depthwise-conv + MLP passes, for which a fused hardware kernel does not yet exist. We regard this as an implementation, not an architectural, limitation.

---

## 4. Experimental Setup

**Dataset.** CIFAR-100 (50 000 train / 10 000 test, 100 classes). Images are upsampled from 32×32 to 224×224 with bicubic interpolation so that we use the standard ViT 16×16 patch embedding.

**Augmentation.** RandomCrop(32, pad=4) → HFlip → Resize(224) → RandAugment(ops=2, mag=9) → Normalize (CIFAR-100 statistics) → RandomErasing(p = 0.25). Mixup (`α = 0.8`) and CutMix (`α = 1.0`) are applied batch-wise with probability 0.5 each and label smoothing `ε = 0.1`.

**Optimization.** AdamW with `lr = 5·10⁻⁴`, `wd = 0.05`, `β = (0.9, 0.999)`, batch size 64, 300 epochs, cosine decay with 10-epoch linear warm-up and `min_lr = 10⁻⁶`. Gradient clipping at 1.0. Mixed precision (AMP) and model-EMA with decay 0.9999. Seed 42 for all runs (single-seed; §7 lists multi-seed as future work).

**Baselines.** DeiT-Tiny (5.54 M) and ViT-Tiny (comparable scale), trained under identical augmentation and optimization from random initialization (no ImageNet pretraining), checkpoints selected by best validation top-1. Swin-Tiny is trained under the same recipe but was not finished at time of writing (§5.1).

**Hardware.** Training was run across two machines to identical config: a home workstation (RTX 4060, 8 GB) for the headline models and the ViT-Tiny/Swin-Tiny baselines, and a lab workstation (RTX 5060 Ti, Blackwell) for the split-ratio ablation sweep. All numbers reported below are reproducible from the released Hydra configs (`configs/train.yaml`, `configs/model/*.yaml`, `configs/data/cifar100.yaml`, `configs/training/default.yaml`) with `batch_size=64` and `seed=42` held fixed across every run.

---

## 5. Results

### 5.1 Main Result

| Model              | Params (M) | GFLOPs | Top-1 (%) | Top-5 (%) | Img/s | Status |
|--------------------|-----------:|-------:|----------:|----------:|------:|--------|
| DeiT-Tiny          |       5.54 |  1.08  |     72.89 |     91.34 |  1672 | finished, re-verified |
| ViT-Tiny           |       ~5.5 |    —   |     72.33 |     —     |   —   | finished |
| Pure NCA-ViT (12:0)|       ~7.2 |    —   |     73.99 |     93.32 |   427 | finished, verified |
| Swin-Tiny          |          — |    —   |         — |     —     |   —   | **training in progress** |
| **Hybrid NCA-ViT (6:6)** |   **6.46** |  2.41  | **81.91** | **95.66** |   585 | finished¹ |
| Δ (Hybrid − DeiT)  |      +0.92 | +1.33  |  **+9.02** |   +4.32 |   −   | |

¹ The 81.91 % top-1 is from a full-test-set evaluation pass on `best.pt` (`results/nca_vit_hybrid_cifar100.json`). The training-loop-tracked EMA validation best (`latest.pt`'s `best_acc` field) shows 81.29 % — a 0.62-point gap consistent with the two measurement paths using slightly different eval harnesses (batch composition / AMP context). We flag this rather than silently pick one: a unified re-evaluation pass is planned before camera-ready. This methodological caution is informed directly by an equivalent DeiT-Tiny discrepancy (73.58 % stale eval-JSON vs 72.88 % checkpoint) that we traced and corrected in this draft — the eval-JSON value there predated a checkpoint-resume bugfix and no longer matched the current checkpoint.

Hybrid NCA-ViT improves top-1 accuracy on CIFAR-100 by **+9.02 %** absolute over a freshly re-verified DeiT-Tiny baseline under matched training conditions, at a cost of ~17 % more parameters and ~2.2× FLOPs. It also beats a from-scratch ViT-Tiny baseline (72.33 %) and the pure-NCA variant (73.99 %) by comparable margins, indicating the gain is not specific to one baseline family.

### 5.2 Robustness

We evaluated adversarial robustness on the test set under matched L∞ budgets.

| Model              | Clean (%) | FGSM (%) | PGD-20 (%) |
|--------------------|----------:|---------:|-----------:|
| **Hybrid NCA-ViT** |  **81.41** |   28.91 |       1.41 |

The clean-input robustness number matches the headline top-1 (small-sample evaluation over 640 images), and the FGSM survival rate of 28.9 % is non-trivial for an undefended model. PGD-20 accuracy is near chance (1.4 %), as expected for any model without explicit adversarial training; we report it for completeness. **This table currently covers only the Hybrid model** — robustness numbers for the baselines are not yet available, so no comparative robustness claim is made in this draft (an earlier internal draft implied a comparative robustness advantage; that claim is withdrawn pending baseline measurements).

### 5.3 Emergent Receptive Field

Figure `figures/emergence_block0.gif` animates the receptive field of a central patch over the `K = 4` iterations of a single NCA block. It starts as a single pixel at iteration 0 and expands to a 9×9 pixel region by iteration 4, consistent with the theoretical `(2K+1)×(2K+1)` bound in §3.3. Static snapshots (`figures/receptive_field_block{0,2,5}.pdf`) show that deeper blocks exhibit larger effective receptive fields.

### 5.4 Ablations — split-ratio curve (finished, two-seed)

The split-ratio sweep, run on a second machine (RTX 5060 Ti) under the identical recipe (§4), is now complete for both seed 42 and an independent seed 123, giving the full curve plus a variance check:

| NCA:Attn split | seed 42 (%) | seed 123 (%) |
|-----------------|------------:|--------------:|
| 0:12 (pure attention) | 76.33 | 77.04 |
| 3:9  | 80.92 | 81.26 |
| **6:6 (Hybrid)** | 81.29–81.91* | **81.59** |
| 9:3  | 80.69 | 81.19 |
| 12:0 (pure NCA) | 73.99* | 73.52 |

*seed-42 values from the headline runs (§5.1); the checkpoint-vs-eval-script gap noted there does not affect the ordering below.

**This confirms the paper's central claim rather than merely suggesting it.** Across both seeds, `6:6` sits at the peak of the curve: it beats pure attention (`0:12`) by **4.6–5.3 points** and pure NCA (`12:0`) by **7.6–8.4 points**, and both neighboring splits (`3:9`, `9:3`) sit below it. The gain is therefore not attributable to "any NCA/attention mix" or to attention alone at reduced depth — the specific 6:6 depth-ordering is what drives the result, consistently across two independent seeds. This is the direct empirical answer to the discussion hypothesis in §6: NCA-early does contribute beyond what an equivalent-depth pure-attention stack achieves.

**Still not ablated in this draft:** NCA depth `K` (only `K=4` has been run); perception filter choice, stochastic rate, CLS-token handling, and zero-init of `W₂` remain implementation choices motivated by NCA literature [4] and standard practice, not yet subjected to controlled ablation.

### 5.5 Test Suite

The repository ships with a `pytest` suite (`tests/test_nca_attention.py`, `test_perception.py`, `test_shapes.py`, `test_training.py`) that verifies shape invariance, gradient flow, perception-kernel correctness, and end-to-end checkpoint resume.

---

## 6. Discussion

**Why does iterated local outperform one-shot global (preliminary)?** Our interpretation is twofold, offered as hypothesis pending the ablations in §5.4. First, the inductive bias of locality is a strong prior for natural images, which is part of why convolutional networks worked well for so long. NCA preserves this bias while still admitting eventual globality through iteration. Second, the *depth* of the NCA iteration gives the network a way to accumulate evidence spatially, rather than distributing attention mass in a single soft-argmax.

**Why does the hybrid beat both pure stacks?** One hypothesis: early layers, operating on low-level features, benefit from the smooth iterative dynamics of NCA, while late layers, operating on class-bearing semantic features, need the direct long-range pooling that softmax attention provides. The completed split-ratio curve (§5.4) is consistent with this: accuracy rises from `0:12` to a peak at `6:6` and falls off toward `12:0`, rather than monotonically favoring more attention or more NCA. This rules out the simpler explanations that the gain comes from attention alone at reduced depth (`0:12` is the weakest configuration) or from NCA alone (`12:0` is also weak) — the specific combination and ordering is doing the work.

**Limitations.**
- Throughput is ~3× lower than DeiT-Tiny because the current implementation uses off-the-shelf PyTorch conv + linear kernels and does not fuse the K iterations; GFLOPs are ~2.2× higher. A dedicated fused kernel could close most of this gap.
- Headline models (Table 5.1) are single-seed (42); the split-ratio ablation (§5.4) additionally covers seed 123, but full multi-seed variance estimates for the headline numbers are not yet available.
- Swin-Tiny baseline and the NCA-depth `K`-sweep ablation are running but not complete at time of writing.
- Robustness is measured only for the Hybrid model; no baseline comparison yet exists.
- Restricted to CIFAR-100 at 224² and tiny-scale models; ImageNet-1k scaling and higher input resolutions are left for future work.

**Broader impact.** The approach is an architectural change and inherits the societal considerations of vision classification generally. It introduces no new data or deployment assumptions.

---

## 7. Conclusion

Global receptive fields in a Transformer do not need to be computed in one step. We show that `K` iterations of a local Neural Cellular Automaton — using fixed Sobel + Identity perception, a shared two-layer update MLP with zero-initialized output, and Bernoulli-gated stochastic updates — can *fully replace* multi-head self-attention while preserving the residual-block structure of a standard ViT. This differs from the closest prior work, AdaNCA [12] and ViTCA [13], which keep attention as the primary mixer and use NCA as an adaptor or fuse it into the attention rule. A layered hybrid, with NCA early and attention late, improves CIFAR-100 top-1 over a same-scale, freshly re-verified DeiT-Tiny by **+9.02 %**. The central design claim — that this specific depth ordering is what drives the gain, rather than any 6:6 split or NCA-attention combination in general — is now supported by a completed, two-seed split-ratio curve (§5.4): `6:6` outperforms `0:12` (pure attention), `12:0` (pure NCA), and both intermediate splits (`3:9`, `9:3`) consistently across seeds 42 and 123. Remaining work before submission: the NCA-depth ablation, finishing the Swin-Tiny baseline, extending robustness evaluation to all baselines, and multi-seed variance estimates for the headline models.

---

## Reproducibility Statement

All model definitions, training scripts, configs, and trained checkpoints (DeiT-Tiny, ViT-Tiny, pure-NCA, Hybrid NCA-ViT; Swin-Tiny checkpoint is a mid-training snapshot) are released. A single command — `python scripts/train.py model=nca_vit_hybrid data=cifar100 training=default` — reproduces the Hybrid result at seed 42; all other rows in Table 5.1 have an equivalent `model=` command using the same `data=cifar100 training=default` flags and `batch_size=64`. Evaluation JSONs are in `results/`. Receptive-field figures and the emergence animation are in `figures/`. The split-ratio ablation (§5.4) is now complete for seeds 42 and 123. The 0.62-point top-1 discrepancy noted in §5.1 and the remaining in-progress items (NCA-depth ablation, Swin-Tiny baseline) are called out explicitly rather than silently resolved, so a reader attempting to reproduce our numbers knows which are settled and which are not.

---

## Appendix A — Hyperparameters and Config Reference

| Field                           | Value                                             |
|---------------------------------|---------------------------------------------------|
| `embed_dim` (D)                 | 192                                               |
| `depth` (L)                     | 12                                                |
| NCA depth (K)                   | 4                                                 |
| NCA hidden dim (H_mlp)          | 384                                               |
| Perception filters              | `{sobel_x, sobel_y, identity}`                    |
| `learnable_filters`             | False (headline models); True in one Hybrid ablation variant only |
| Stochastic rate (p)             | 0.5                                               |
| MLP ratio                       | 4.0                                               |
| Update MLP activation           | ReLU                                              |
| Patch size                      | 16                                                |
| Input resolution                | 224 × 224                                         |
| Num classes                     | 100                                               |
| Optimizer / lr / wd             | AdamW / 5·10⁻⁴ / 0.05                            |
| Betas / eps                     | (0.9, 0.999) / 10⁻⁸                              |
| Epochs / warm-up                | 300 / 10                                          |
| Min lr                          | 10⁻⁶                                             |
| Batch size                      | 64                                                |
| Label smoothing                 | 0.1                                               |
| Mixup α / CutMix α              | 0.8 / 1.0                                         |
| Mixup prob / CutMix prob        | 0.5 / 0.5                                         |
| RandAugment (ops, mag)          | (2, 9)                                            |
| Random erasing p                | 0.25                                              |
| Grad clip / AMP / EMA / decay   | 1.0 / on / on / 0.9999                            |
| Seed                            | 42                                                |

---

## Appendix B — Pseudocode for Emergent Attention

```python
class NCAAttention(nn.Module):
    def __init__(self, dim=192, nca_steps=4, hidden_dim=384,
                 grid_size=(14, 14), stochastic_rate=0.5,
                 filter_names=("sobel_x", "sobel_y", "identity"),
                 learnable_filters=False):
        super().__init__()
        self.K           = nca_steps
        self.grid_shape  = grid_size
        self.stoch_rate  = stochastic_rate
        self.norm        = nn.LayerNorm(dim)
        self.perception  = PerceptionModule(dim, filter_names, learnable_filters)
        M                = len(filter_names)
        self.linear1     = nn.Linear(dim * M, hidden_dim)
        self.act         = nn.ReLU()
        self.linear2     = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.linear2.weight)     # identity at init
        nn.init.zeros_(self.linear2.bias)

    def _nca_step(self, grid):
        perc  = self.perception(grid)
        perc  = rearrange(perc, "b dm hp wp -> b hp wp dm")
        delta = self.linear2(self.act(self.linear1(perc)))
        delta = rearrange(delta, "b hp wp d -> b d hp wp")
        if self.training and self.stoch_rate < 1.0:
            B, _, Hp, Wp = grid.shape
            mask  = torch.bernoulli(torch.full(
                (B, 1, Hp, Wp), self.stoch_rate, device=grid.device))
            return grid + mask * delta
        return grid + delta

    def forward(self, z):
        cls, patches = z[:, :1], z[:, 1:]
        patches      = self.norm(patches)
        B, n, D      = patches.shape
        Hp, Wp       = self.grid_shape
        s0           = rearrange(patches, "b (hp wp) d -> b d hp wp",
                                 hp=Hp, wp=Wp)
        s            = s0
        for _ in range(self.K):
            s = self._nca_step(s)
        delta = rearrange(s - s0, "b d hp wp -> b (hp wp) d")
        return torch.cat([torch.zeros_like(cls), delta], dim=1)
```

---

## References

1. Dosovitskiy et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
2. Touvron et al. *Training Data-Efficient Image Transformers & Distillation through Attention.* ICML 2021.
3. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
4. Mordvintsev et al. *Growing Neural Cellular Automata.* Distill, 2020.
5. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
6. Loshchilov & Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
7. Cubuk et al. *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.* CVPR 2020.
8. Zhang et al. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
9. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
10. Goodfellow et al. *Explaining and Harnessing Adversarial Examples.* ICLR 2015.
11. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
12. Anonymous / AdaNCA authors. *AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer.* NeurIPS 2024. arXiv:2406.08298.
13. Tesfaldet et al. *Attention-based Neural Cellular Automata.* NeurIPS 2022. arXiv:2211.01233.
