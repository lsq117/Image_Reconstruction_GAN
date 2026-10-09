"""Create a reproducible patient-level split from the user's own data."""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from resunet_gan.config import deep_update, load_config
from resunet_gan.data import discover_lesion_samples, load_or_create_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reconstruction_gan.yaml")
    parser.add_argument("--data-root", help="Folder containing one subfolder per patient")
    parser.add_argument("--output", help="Destination JSON; defaults to split.path in the YAML")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing split explicitly")
    args = parser.parse_args()
    config = load_config(args.config)
    updates = {}
    if args.data_root:
        updates["data.lesion_root"] = args.data_root
    if args.output:
        updates["split.path"] = args.output
    config = deep_update(config, updates)
    records, warnings = discover_lesion_samples(config)
    if not records:
        raise ValueError("No complete cases found. Check the folder and filename examples in README.md.")
    if warnings:
        raise ValueError("Some cases are incomplete; correct them before splitting:\n" + "\n".join(warnings[:10]))
    split = load_or_create_split(records, config, overwrite=args.overwrite)
    print(f"Split saved or reused: {config['split']['path']}")
    print(
        f"Patients: train={len(split['train'])}, "
        f"validation={len(split['val'])}, test={len(split['test'])}"
    )


if __name__ == "__main__":
    main()
