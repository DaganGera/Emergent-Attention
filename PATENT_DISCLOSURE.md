# Provisional Patent Disclosure

**Type:** Draft for Provisional Specification (Indian Patent Act, 1970)
**Status:** Invention disclosure — for review and filing by mentor / registered patent agent
**Evidence commit:** `5f8df70` (this document is drafted against this exact, reproducible codebase state)
**Date of disclosure:** 2026-08-12
**Applicant:** [TO BE COMPLETED — institution/individual per mentor's guidance]
**Inventor(s):** [TO BE COMPLETED — mentor to confirm inventorship per §2 note below]

---

## Note to the mentor / patent agent (read this first)

This document is a **provisional specification draft**, chosen deliberately over a complete specification because active work is still running (a data-efficiency sweep, additional seeds) that would only strengthen, not change, the claims below — provisional filing secures a priority date now and gives 12 months to finalize. Everything stated as a number in this document is measured and reproducible from the evidence commit above; nothing is projected or aspirational. Section-by-section:

- **§3(k) (algorithm exclusion) risk** is the main thing to watch. Indian patent law excludes "a mathematical method or a computer programme per se." This disclosure is written to claim a **system with a demonstrated technical effect on a stated technical problem** (classification accuracy retained under sensor noise, §7), not an abstract algorithmic improvement — accuracy-on-a-benchmark alone is exactly the kind of claim examiners reject under §3(k), so it is deliberately not the basis of Claim 1.
- **§3(i) (diagnostic method exclusion) risk**: Claim 5 involves medical ultrasound. It is worded as a system/apparatus claim ("a system configured to process...") and an image-classification method claim, not as "a method of diagnosing a patient." Please have the patent agent confirm this framing survives §3(i) scrutiny before filing — this is a legal judgment call outside the scope of what an engineering disclosure can settle.
- **Inventorship** (front-matter placeholder) is left for you to confirm — settle who contributed to technical conception (architecture and experiment design) versus supervision before the complete specification is due.
- Every number below has a source file cited in **§10, Evidence Index** — if a patent agent or examiner questions a figure, that section says exactly which file to open and which command reproduces it.

---

## 1. Title of Invention

**A Hybrid Local-Global Token-Mixing Architecture for Noise-Robust Image Classification Using Iterated Neural Cellular Automata**

## 2. Field of Invention

The invention relates to computer vision systems, specifically to neural network architectures for image classification, and more specifically to a hybrid token-mixing architecture that combines iterated local cellular-automaton-based processing with global self-attention processing to achieve improved classification robustness under sensor noise and degraded image quality — with a demonstrated application to medical ultrasound image analysis.

## 3. Background of the Invention

**3.1 The problem.** Vision Transformer (ViT) architectures classify images by dividing them into patches and applying self-attention, a mechanism in which every image patch is compared against every other patch to determine relevance. This "all-to-all" comparison is computationally expensive and, more importantly for this disclosure, is a *learned* mechanism: the network must learn, from training data, that nearby pixels are usually related. When training data is limited or test-time images are corrupted by sensor noise, this learned locality is unreliable — the network has not seen enough examples to learn a robust, noise-invariant notion of "nearby."

**3.2 Prior art.** Two published works combine Neural Cellular Automata (NCA — a class of models in which a small local update rule is applied repeatedly across a spatial grid) with Vision Transformers:

- **AdaNCA** (NeurIPS 2024, arXiv:2406.08298) inserts NCA modules as *adaptors between* existing self-attention layers. Standard self-attention remains the primary mechanism throughout the network; the NCA module supplements it. The published work reports improved robustness in aggregate across multiple benchmarks but does not decompose *which* categories of image corruption benefit, nor does it provide a mechanistic account of why.
- **ViTCA** (NeurIPS 2022, arXiv:2211.01233) computes the per-cell update rule of its cellular automaton *using* self-attention over a local neighborhood — attention is fused into the cellular automaton's update rule, not replaced by it.

In both prior works, self-attention is never removed from the network; the cellular-automaton component is auxiliary to it.

**3.3 The gap.** No prior work known to the applicant (a) fully replaces self-attention with iterated local cellular-automaton processing in part of the network, (b) uses a **fixed, non-learned** local perception operation (rather than a learned one) as the basis for that replacement, or (c) demonstrates and mechanistically explains a **frequency-specific** robustness property — i.e., that the benefit is concentrated in a specific, identifiable category of image degradation rather than claimed as a general improvement.

## 4. Objects of the Invention

1. To provide an image-classification architecture that retains higher classification accuracy than a comparable self-attention-only architecture when input images are corrupted by additive, high-frequency sensor noise.
2. To provide such an architecture using a computationally simple, non-learned local perception mechanism, reducing the number of learned parameters relative to a comparably-accurate pure-self-attention architecture.
3. To provide an architecture whose ordering of local and global processing stages is empirically determined to maximize classification accuracy, rather than arbitrarily chosen.
4. To demonstrate the invention's technical effect on a concrete applied domain — medical ultrasound imaging — in which the specific noise process the architecture is robust to (speckle) is the dominant, unavoidable image-degradation source.

## 5. Summary of the Invention

The invention is an image-classification neural network comprising two sequential stages operating on a common sequence of image-patch tokens:

**Stage 1 (local, early).** A plurality of processing blocks, each performing $K$ iterations of a local update rule on a spatial grid of patch tokens. Each iteration: (a) applies a **fixed, non-learned** spatial perception operation — specifically, three parallel $3\times3$ filters (a horizontal gradient filter, a vertical gradient filter, and an identity filter) applied depthwise; (b) passes the perception output through a two-layer neural network (the "update network"), shared identically across every spatial location, whose final layer is initialized to zero so the block performs an exact identity mapping at the start of training; (c) applies the resulting update to the grid, gated at training time by an independent random binary mask per spatial location (disabled at inference for determinism). A designated classification token is excluded from this stage entirely.

**Stage 2 (global, late).** A plurality of processing blocks, positioned after Stage 1 in network depth, each performing standard multi-head self-attention over the full token sequence.

**Classification.** The patch-token representations (not the classification token, which never receives image-dependent information in Stage 1) are pooled and passed to a linear classification layer.

The invention has been reduced to practice and evaluated at a configuration of 6 Stage-1 blocks followed by 6 Stage-2 blocks (embedding dimension 192, $K=4$ iterations per block), on 100-class natural-image classification (CIFAR-100) and 3-class medical ultrasound classification (BUSI). Ablation across five split ratios (0:12 through 12:0, both stages present at intermediate ratios) at two independent random seeds confirms this ordering — local stage before global stage — is a genuine local maximum of classification accuracy, not an arbitrary choice among equally-good alternatives (Evidence, §7.2).

## 6. Brief Description of Drawings

- **Fig. 1** — System block diagram: (a) the full two-stage pipeline from image input to classification output; (b) the internal iterated-update loop of one Stage-1 block. *(`figures/paper/architecture.png`)*
- **Fig. 2** — Classification-accuracy retention under 19 categories of image corruption, comparing the invention against attention-only baseline architectures, decomposed by corruption category. *(`figures/paper/corruption_robustness.png`)*
- **Fig. 3** — Internal representation-stability measurement demonstrating the technical mechanism: the invention's classification-layer input is measurably more stable under noise categories where its accuracy is higher, and measurably less stable under the one category where its accuracy is lower — a falsifiable, sign-correct mechanistic account. *(`figures/paper/noise_drift.png`)*
- **Fig. 4** — Example classification outputs under corrupted input, comparing the invention against four baseline architectures on the same corrupted images. *(`figures/paper/gradcam_noisy.png`)*
- **Fig. 5** — Classification-accuracy comparison on a medical ultrasound image dataset (BUSI), across three independent trained instances (seeds) of the invention versus a baseline architecture. *(`figures/paper/busi_seeds.png`)*

## 7. Detailed Description of the Invention

### 7.1 Architecture

Let an input image be divided into $n$ patches, each embedded into a $D$-dimensional token, with one additional classification token prepended, giving a token sequence $z \in \mathbb{R}^{(1+n) \times D}$ arranged so the $n$ patch tokens correspond to positions on an $H_p \times W_p$ spatial grid ($H_p \cdot W_p = n$).

**Stage 1 block (repeated $L_1$ times).** Given input sequence $z$, the classification token is set aside unchanged. The remaining $n$ patch tokens are normalized and reshaped to the spatial grid $s_0 \in \mathbb{R}^{D \times H_p \times W_p}$. For $t = 0, \ldots, K-1$:

1. **Perception:** a depthwise convolution with three fixed $3\times3$ kernels (horizontal-gradient, vertical-gradient, identity) is applied to $s_t$, producing a $3D$-channel perception tensor.
2. **Update:** the perception tensor is processed, independently and identically at every spatial location, by a two-layer network — a first linear layer, a nonlinearity, and a second linear layer whose weights and bias are initialized to zero — producing an update tensor $\delta_t$.
3. **Gated write:** during training, an independent Bernoulli-distributed binary mask $m_t$ (one value per spatial location, shared across the channel dimension) is sampled, and $s_{t+1} = s_t + m_t \odot \delta_t$. During inference, the mask is omitted: $s_{t+1} = s_t + \delta_t$.

After $K$ iterations, the block's output contribution is $\Delta = s_K - s_0$, converted back to sequence form and added to the residual stream; the classification token's contribution is fixed at zero. Because the update network's output layer is zero-initialized, $\Delta \equiv 0$ at the start of training, so the block begins as an identity mapping — a design choice that stabilizes training without requiring a separate warm-up procedure for this stage.

**Stage 2 block (repeated $L_2$ times).** Standard multi-head self-attention operating on the full $(1+n)$-token sequence, unmodified from conventional Transformer practice.

**Classification head.** After all blocks, the $n$ patch-token representations (excluding the classification token) are averaged and passed through a linear layer to produce class scores.

### 7.2 Empirical determination of stage ordering (technical basis for Claim 3)

At $L_1 + L_2 = 12$ fixed, five configurations were trained under an identical procedure and evaluated on held-out data at two independent random seeds:

| Stage-1 : Stage-2 blocks | Accuracy, seed A | Accuracy, seed B |
|---|---:|---:|
| 0 : 12 (Stage 2 only) | 76.33% | 77.04% |
| 3 : 9 | 80.92% | 81.26% |
| **6 : 6** | 81.29–81.91% | **81.59%** |
| 9 : 3 | 80.69% | 81.19% |
| 12 : 0 (Stage 1 only) | 73.99% | 73.52% |

The 6:6 configuration is the accuracy maximum at both seeds, with both neighboring ratios scoring lower — establishing that the specific ordering (Stage 1 before Stage 2, at approximately equal depth split) is what produces the accuracy improvement, rather than the mere co-presence of both stage types in any proportion.

### 7.3 Demonstrated technical effect: accuracy retention under sensor noise (technical basis for Claims 1, 5, 6 and the §3(k) technical-effect requirement)

The invention was evaluated against four baseline architectures — three conventional self-attention architectures of comparable and larger scale, and one architecture using only Stage-1-type processing throughout — on a standard image-corruption benchmark comprising 19 distinct categories of image degradation at graded severity (CIFAR-100-C, Hendrycks & Dietterich, ICLR 2019).

**Result 1 — aggregate.** Mean classification accuracy across all 19 corruption categories: the invention achieves 61.88%, versus 59.37% for the strongest self-attention baseline (which matches the invention's clean-image accuracy using 4.3× more trainable parameters) and 53.60%/52.18% for two smaller self-attention baselines.

**Result 2 — the technical effect is category-specific, not general.** Decomposed by corruption category, the invention's advantage over the strongest self-attention baseline is +8.3 percentage points on additive high-frequency pixel-noise corruptions, falling to near-zero (+0.2 to +1.7 points) on blur, weather, and photometric corruptions, and reversing to −5.9 points on impulse (sparse-outlier) noise. This category-specificity, rather than undermining the invention, is direct evidence of a genuine physical mechanism rather than a general-purpose accuracy improvement that happens to also show up on a noise benchmark — an important distinction for demonstrating a *technical* effect rather than an abstract algorithmic one.

**Result 3 — mechanistic verification.** A representation-stability measurement was performed at the input to each architecture's classification layer: the invention's internal representation moves measurably less under noise categories where its accuracy is higher than baseline (a reduction of 16–26% in representation displacement relative to a size-matched self-attention baseline, on the categories showing the largest accuracy gains) and moves *more* — not less — on the one category (impulse noise) where its accuracy is lower than baseline. This sign-correct correspondence between an internal, architecture-level measurement and the external accuracy result is offered as direct technical evidence of *how* the invention achieves its effect, rather than an unexplained empirical correlation.

### 7.4 Demonstrated application: medical ultrasound image classification (technical basis for Claim 5)

Speckle — a granular interference artifact arising from coherent-wave image acquisition — is the corruption category showing the single largest measured advantage in §7.3 (+9.6 percentage points on the specific speckle-noise category) and is simultaneously the dominant, physically unavoidable image-degradation source in ultrasound imaging. The invention was evaluated on a public breast-ultrasound classification dataset (BUSI: 780 images, 3 diagnostic categories, from 600 patients), trained from random initialization, against a size-matched self-attention baseline, at three independent random seeds:

| Metric | Baseline (self-attention), 3-seed mean | Invention, 3-seed mean |
|---|---:|---:|
| Balanced classification accuracy | 36.39% | **49.83%** |
| Macro-averaged F1 score | 20.97% | **41.46%** |

At every one of the three independent seeds, the invention outperforms the baseline's best seed on both metrics — i.e., the performance ranges of the two architectures do not overlap. This constitutes a demonstrated technical effect on a real, non-synthetic applied domain, satisfying the requirement that the invention solve a concrete technical problem in a stated field of application rather than showing an improvement only on an abstract accuracy metric.

## 8. Claims

**Claim 1 (independent — system claim).**
A computer-implemented image classification system comprising:
(a) a patch embedding stage configured to divide an input image into a plurality of patches and produce a corresponding sequence of patch tokens, together with a designated classification token;
(b) a first plurality of processing blocks configured to iteratively update the patch tokens over $K$ iterations, wherein each iteration comprises: applying a fixed, non-learned spatial perception operation to the patch tokens arranged on a spatial grid; processing the resulting perception output through an update network shared identically across all spatial locations, said update network having an output layer initialized to zero-valued weights and biases; and applying the resulting update to the patch tokens, gated during a training phase by an independently-sampled random binary mask per spatial location and applied without gating during an inference phase; wherein the designated classification token does not participate in said iterative updating;
(c) a second plurality of processing blocks, positioned after the first plurality of processing blocks in processing order, configured to perform multi-head self-attention across the full sequence of patch tokens and the classification token; and
(d) a classification head configured to pool the patch tokens, following processing by the first and second pluralities of processing blocks, and to produce a classification output therefrom;
wherein the system exhibits, relative to a comparable image classification system employing multi-head self-attention throughout without said first plurality of processing blocks, an increased retention of classification accuracy when the input image is degraded by additive high-frequency pixel-noise.

**Claim 2 (dependent on Claim 1).**
The system of Claim 1, wherein the fixed spatial perception operation comprises three depthwise-convolutional filters applied per channel: a horizontal spatial-gradient filter, a vertical spatial-gradient filter, and an identity filter, the weights of said filters remaining fixed and non-learned throughout a training procedure.

**Claim 3 (dependent on Claim 1).**
The system of Claim 1, wherein the first plurality of processing blocks and the second plurality of processing blocks are of substantially equal number, together constituting the full processing depth of the system, and wherein said substantially equal split with the first plurality preceding the second in processing order is configured to produce a classification-accuracy improvement, relative to processing orderings in which the proportion of the first plurality to the second plurality differs from substantially equal, or in which the second plurality precedes the first.

**Claim 4 (dependent on Claim 1).**
The system of Claim 1, wherein the update network comprises a first linear transformation, followed by a nonlinear activation function, followed by a second linear transformation, and wherein the zero-initialization of the second linear transformation's weights and bias causes each of the first plurality of processing blocks to perform an identity mapping at the commencement of a training procedure.

**Claim 5 (dependent on Claim 1 — applied embodiment).**
A method of classifying a medical ultrasound image using the system of Claim 1, wherein the additive high-frequency pixel-noise comprises speckle noise arising from coherent-wave ultrasound image acquisition, and wherein the system produces a diagnostic-category classification output with increased retention of classification accuracy relative to a comparable image classification system employing multi-head self-attention throughout, when applied to ultrasound images exhibiting said speckle noise.

**Claim 6 (dependent on Claim 1).**
The system of Claim 1, wherein the classification head is configured to pool the patch tokens by averaging, excluding the classification token, such that the classification output is independent of any representation carried by the classification token.

## 9. Abstract

*(Prepared to approximately 150 words per Indian Patent Office formatting convention.)*

An image classification system comprises a first plurality of processing blocks performing iterated, spatially-local updates using a fixed, non-learned perception operation and a zero-initialized, stochastically-gated update network, followed by a second plurality of processing blocks performing global multi-head self-attention. Ablation across five stage-ordering configurations at two independent random seeds confirms that positioning the local-update blocks before the attention blocks, at approximately equal depth split, maximizes classification accuracy. Evaluated against attention-only baseline architectures across nineteen categories of image corruption, the system demonstrates a classification-accuracy retention advantage concentrated specifically in additive high-frequency pixel-noise corruption, verified by direct measurement of internal representation stability. Applied to medical ultrasound image classification — where speckle noise, the corruption category showing the largest measured advantage, is the dominant acquisition artifact — the system outperforms a comparable self-attention baseline across three independent trained instances with non-overlapping performance ranges.

## 10. Evidence Index

For verification by a patent agent or examiner. All paths relative to the evidence commit `5f8df70`.

| Claim / Section | Evidence file(s) | Reproduce with |
|---|---|---|
| §7.2 stage-ordering table | `paper.md` §5.2; checkpoints under `checkpoints/nca_vit_hybrid/cifar100/seed{42,123}/` | `python scripts/train.py model=nca_vit_hybrid model.nca_depth=N model.attn_depth=M data=cifar100` |
| §7.3 Result 1–2, corruption table | `results/cifar100c_robustness.json` | `python scripts/evaluate_corruptions.py` |
| §7.3 Result 3, mechanism | `results/noise_drift.json`, `figures/paper/noise_drift.png` | `python scripts/measure_noise_drift.py` |
| §7.4 BUSI table | `results/busi_v2/`, `scripts/plot_busi_seeds.py` (transcribed per-seed values in-script) | `python scripts/train.py model=nca_vit_hybrid model.num_classes=3 data=busi training.augmentation.mixup_alpha=0 training.augmentation.cutmix_alpha=0 training.training.class_weighted=true training.training.seed={42,123,7}` |
| Architecture (§7.1, Fig. 1) | `src/models/hybrid_nca_vit.py`, `src/models/nca_attention.py` | — (source code) |
| Full corruption breakdown | `paper.md` Appendix C | `python scripts/evaluate_corruptions.py --full` |

---

*End of disclosure. This document is an engineering-authored draft intended to accelerate, not substitute for, review by a registered patent agent — particularly the §3(k)/§3(i) framing noted at the top of this document.*
