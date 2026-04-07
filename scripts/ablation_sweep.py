"""
Ablation sweep launcher.

Generates and optionally runs all ablation configurations defined in
configs/sweep/ablation.yaml.

Usage:
    # Dry-run: print all commands
    python scripts/ablation_sweep.py --dry-run

    # Run all ablations sequentially
    python scripts/ablation_sweep.py --group A1

    # Run a specific ablation group
    python scripts/ablation_sweep.py --group A1  # K ablation
    python scripts/ablation_sweep.py --group A3  # stochastic rate
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ablation definitions — maps group ID to list of override strings
ABLATIONS = {
    "A1": [
        ("K1",  "model.nca_steps=1"),
        ("K2",  "model.nca_steps=2"),
        ("K4",  "model.nca_steps=4"),
        ("K8",  "model.nca_steps=8"),
        ("K16", "model.nca_steps=16"),
    ],
    "A3": [
        ("p03", "model.stochastic_rate=0.3"),
        ("p05", "model.stochastic_rate=0.5"),
        ("p07", "model.stochastic_rate=0.7"),
        ("p10", "model.stochastic_rate=1.0"),
    ],
    "A4": [
        ("h128", "model.update_hidden_dim=128"),
        ("h256", "model.update_hidden_dim=256"),
        ("h384", "model.update_hidden_dim=384"),
        ("h512", "model.update_hidden_dim=512"),
    ],
    "A5": [
        ("d6",  "model.depth=6"),
        ("d8",  "model.depth=8"),
        ("d10", "model.depth=10"),
        ("d12", "model.depth=12"),
    ],
    "A6": [
        ("cls_exclude",   "model.cls_strategy=exclude"),
    ],
}

BASE_CMD = (
    "python scripts/train.py "
    "model=nca_vit_tiny data=cifar100 "
    "training.training.seed=42"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(ABLATIONS.keys()) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    groups = list(ABLATIONS.keys()) if args.group == "all" else [args.group]

    for group in groups:
        print(f"\n=== Ablation Group {group} ===")
        for name, override in ABLATIONS[group]:
            cmd = f"{BASE_CMD} {override} hydra.run.dir=outputs/ablation/{group}/{name}"
            print(f"  [{group}-{name}] {cmd}")
            if not args.dry_run:
                result = subprocess.run(cmd, shell=True)
                if result.returncode != 0:
                    print(f"  ERROR: {group}-{name} failed with code {result.returncode}")


if __name__ == "__main__":
    main()
