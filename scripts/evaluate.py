"""
Evaluation entry point.

Loads a checkpoint and runs:
  - Top-1 / Top-5 accuracy
  - FLOPs + parameter count + throughput
  - Adversarial robustness (FGSM, PGD-20)

Saves results to results/<run_name>.json

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/nca_vit_tiny/cifar100/seed42/best.pt \
        --model nca_vit_tiny --dataset cifar100
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nca_vit import NCAViT
from src.models.hybrid_nca_vit import HybridNCAViT
from src.models.baselines import create_baseline
from src.data.datasets import build_cifar100_loaders, build_imagenet100_loaders
from src.evaluation.metrics import evaluate_accuracy
from src.evaluation.flops import count_flops, count_parameters, measure_throughput
from src.evaluation.robustness import evaluate_robustness
from src.utils.checkpoint import load_checkpoint


MODEL_BUILDERS = {
    "nca_vit_tiny": lambda nc: NCAViT(num_classes=nc),
    "nca_vit_hybrid": lambda nc: HybridNCAViT(num_classes=nc),
    "vit_tiny": lambda nc: create_baseline("vit_tiny", num_classes=nc),
    "deit_tiny": lambda nc: create_baseline("deit_tiny", num_classes=nc),
    "swin_tiny": lambda nc: create_baseline("swin_tiny", num_classes=nc),
}

DATASET_NUM_CLASSES = {"cifar100": 100, "imagenet100": 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet100"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--robustness", action="store_true", help="Run adversarial eval")
    parser.add_argument("--autoattack", action="store_true", help="Run AutoAttack (slow)")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = DATASET_NUM_CLASSES[args.dataset]

    # Build model
    model = MODEL_BUILDERS[args.model](num_classes).to(device)

    # Load checkpoint
    state = load_checkpoint(args.checkpoint)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Data
    if args.dataset == "cifar100":
        _, val_loader = build_cifar100_loaders(batch_size=args.batch_size)
    else:
        _, val_loader = build_imagenet100_loaders(batch_size=args.batch_size)

    results = {"model": args.model, "dataset": args.dataset, "checkpoint": args.checkpoint}

    # Accuracy
    print("Computing accuracy...")
    acc = evaluate_accuracy(model, val_loader, device)
    results.update(acc)
    print(f"  Top-1: {acc['top1']:.2f}%  Top-5: {acc['top5']:.2f}%")

    # FLOPs + params + throughput
    print("Computing FLOPs and throughput...")
    flops = count_flops(model, device=device)
    params = count_parameters(model)
    throughput = measure_throughput(model, device=device)

    results["gflops"] = flops["total_gflops"]
    results["params_M"] = params["total"] / 1e6
    results["images_per_sec"] = throughput["images_per_sec"]
    results["ms_per_batch"] = throughput["ms_per_batch"]
    print(f"  GFLOPs: {flops['total_gflops']:.2f}  Params: {params['total']/1e6:.1f}M  "
          f"Throughput: {throughput['images_per_sec']:.0f} img/s")

    # Robustness
    if args.robustness:
        print("Computing adversarial robustness (FGSM + PGD-20)...")
        rob = evaluate_robustness(
            model, val_loader, device,
            run_autoattack=args.autoattack,
        )
        results["robustness"] = rob
        print(f"  Clean: {rob['clean']:.2f}%  FGSM: {rob['fgsm']:.2f}%  PGD-20: {rob['pgd20']:.2f}%")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"{args.model}_{args.dataset}"
    out_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
