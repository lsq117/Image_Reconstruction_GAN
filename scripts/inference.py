from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from resunet_gan.config import deep_update, load_config
from resunet_gan.engine import inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reconstruction_gan.yaml")
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    updates = {}
    if args.checkpoint is not None:
        updates["inference.checkpoint"] = args.checkpoint
    if args.device is not None:
        updates["train.device"] = args.device
    config = deep_update(config, updates)
    paths = inference(config, args.case_root, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
