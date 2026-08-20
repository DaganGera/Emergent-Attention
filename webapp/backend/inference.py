"""
Live single-image inference + Grad-CAM for the demo website.

Loads the same Hybrid NCA-ViT checkpoints used throughout the project
(checkpoints/nca_vit_hybrid/{cifar100,busi}/seed42/best.pt) with the exact
architecture and preprocessing used at training/evaluation time (see
src/data/datasets.py and scripts/gradcam_busi.py) -- this module does not
retrain or fine-tune anything, only runs the already-trained model forward.
"""

import os
import sys
import io
import base64
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.cm as cm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.hybrid_nca_vit import HybridNCAViT
from src.utils.checkpoint import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact hyperparameters used to train every nca_vit_hybrid checkpoint in this
# repo (configs/model/nca_vit_hybrid.yaml); only num_classes differs by domain.
_MODEL_KWARGS = dict(
    embed_dim=192, nca_depth=6, attn_depth=6, nca_steps=4,
    filter_names=["sobel_x", "sobel_y", "identity"], nca_hidden_dim=384,
    stochastic_rate=0.5, mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
    learnable_filters=True,
)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

BUSI_CLASSES = ["benign", "malignant", "normal"]

_DOMAINS = {
    "cifar100": dict(
        num_classes=100,
        ckpt=os.path.join(_REPO_ROOT, "checkpoints", "nca_vit_hybrid", "cifar100", "seed42", "best.pt"),
        mean=_CIFAR100_MEAN, std=_CIFAR100_STD,
    ),
    "busi": dict(
        num_classes=3,
        ckpt=os.path.join(_REPO_ROOT, "checkpoints", "nca_vit_hybrid", "busi", "seed42", "best.pt"),
        mean=_IMAGENET_MEAN, std=_IMAGENET_STD,
    ),
}


def _load_cifar100_classes() -> list[str]:
    meta_path = os.path.join(_REPO_ROOT, "data", "cifar-100-python", "meta")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    return meta["fine_label_names"]


def _classes_for(domain: str) -> list[str]:
    if domain == "cifar100":
        return _load_cifar100_classes()
    return BUSI_CLASSES


# CIFAR-100 was trained on its native 32x32 frames upsampled directly to 224;
# real uploaded photographs are not 32x32 to begin with, so a photo upload
# gets resize+center-crop instead -- this mode is evaluated as a benchmark
# on CIFAR-100 itself, and is illustrative (not benchmark-accurate) on
# arbitrary user photos, which the frontend states explicitly.
def _transform_for(domain: str) -> T.Compose:
    if domain == "cifar100":
        mean, std = _CIFAR100_MEAN, _CIFAR100_STD
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


_models: dict[str, torch.nn.Module] = {}


def _get_model(domain: str) -> torch.nn.Module:
    if domain not in _DOMAINS:
        raise ValueError(f"unknown domain: {domain}")
    if domain in _models:
        return _models[domain]

    spec = _DOMAINS[domain]
    model = HybridNCAViT(num_classes=spec["num_classes"], **_MODEL_KWARGS)
    state = load_checkpoint(spec["ckpt"])
    state = state["model"] if "model" in state else state
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    _models[domain] = model
    return model


def _denormalize(img_t: torch.Tensor, mean, std) -> np.ndarray:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    x = img_t.detach().cpu() * s + m
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def _gradcam(model: torch.nn.Module, img_t: torch.Tensor, target_class: int | None):
    """Grad-CAM on the last global-context (self-attention) block -- same
    technique and target layer as scripts/gradcam_busi.py."""
    activation = {}

    def hook(_module, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t.retain_grad()
        activation["value"] = t

    handle = model.attn_blocks[-1].register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logits = model(img_t)
    probs = F.softmax(logits[0], dim=0)
    if target_class is None:
        target_class = int(probs.argmax())
    score = logits[0, target_class]
    score.backward()
    handle.remove()

    act = activation["value"][0].detach()
    grad = activation["value"].grad[0].detach()
    patch_act, patch_grad = act[1:], grad[1:]  # drop CLS token
    n = patch_act.shape[0]
    side = int(round(n ** 0.5))
    patch_act = patch_act.reshape(side, side, -1)
    patch_grad = patch_grad.reshape(side, side, -1)

    weights = patch_grad.mean(dim=(0, 1))
    cam = torch.relu((patch_act * weights).sum(dim=-1))
    cam = cam / (cam.max() + 1e-8)
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=img_t.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().cpu().numpy(), probs.detach().cpu().numpy(), target_class


def _overlay(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * base_rgb + alpha * heat, 0, 1)


def _to_base64_png(arr_float01: np.ndarray) -> str:
    img = Image.fromarray((arr_float01 * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def predict(domain: str, pil_image: Image.Image) -> dict:
    """Run the trained model on one image; return the decision, a ranked
    list of alternatives, and a Grad-CAM visual explanation."""
    pil_image = pil_image.convert("RGB")
    model = _get_model(domain)
    transform = _transform_for(domain)
    img_t = transform(pil_image).unsqueeze(0).to(DEVICE)

    cam, probs, pred_idx = _gradcam(model, img_t, target_class=None)

    classes = _classes_for(domain)
    topk = min(5, len(classes))
    top_idx = np.argsort(-probs)[:topk]
    top_predictions = [
        {"label": classes[i], "confidence": float(probs[i])} for i in top_idx
    ]

    spec = _DOMAINS[domain]
    base_rgb = _denormalize(img_t[0], spec["mean"], spec["std"])
    overlay_rgb = _overlay(base_rgb, cam)

    return {
        "domain": domain,
        "predicted_label": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_predictions": top_predictions,
        "input_image_b64": _to_base64_png(base_rgb),
        "heatmap_image_b64": _to_base64_png(overlay_rgb),
    }
