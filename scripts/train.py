from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from resunet_gan.config import deep_update, load_config
from resunet_gan.engine import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reconstruction_gan.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume")
    checkpoint_group.add_argument("--init-checkpoint")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    updates = {}
    if args.epochs is not None:
        updates["train.epochs"] = args.epochs
    if args.batch_size is not None:
        updates["train.batch_size"] = args.batch_size
    if args.num_workers is not None:
        updates["data.num_workers"] = args.num_workers
    if args.device is not None:
        updates["train.device"] = args.device
    if args.output_dir is not None:
        updates["project.output_dir"] = args.output_dir
        updates["project.output_root"] = args.output_dir
    if args.resume is not None:
        updates["train.resume"] = args.resume
        updates["train.init_checkpoint"] = None
    if args.init_checkpoint is not None:
        updates["train.init_checkpoint"] = args.init_checkpoint
        updates["train.resume"] = None
    if args.max_train_batches is not None:
        updates["train.max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        updates["train.max_val_batches"] = args.max_val_batches
    config = deep_update(config, updates)
    train(config)


if __name__ == "__main__":
    main()
