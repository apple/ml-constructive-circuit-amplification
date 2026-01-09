# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import os
import subprocess

import click
from gsm_utils import logger

# Use artifact directory if available
if os.environ.get("ARTIFACT_DIR", None) is not None:
    OUTPUT_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", ""), "experiments")
else:
    OUTPUT_DIR = os.path.join(os.getcwd(), "results", "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"OUTPUT DIR: {OUTPUT_DIR}")


@click.command()
@click.option(
    "--config_file", type=str, required=True, help="Config file to use for fixing pipeline."
)
def main(config_file: str):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        config = json.load(f)

    default_cmd = [
        "python",
        "math_reasoning/fixing_pipeline.py",
        "--model_name",
        config.get("model_name"),
        "--train_template_idx",
        str(config.get("train_template_idx")),
        "--sampled_templates_fraction",
        str(config.get("sampled_templates_fraction", 1.0)),
        "--train_size",
        str(config.get("train_size")),
        "--test_size",
        str(config.get("test_size")),
        "--val_size",
        str(config.get("val_size")),
        "--train_batch_size",
        str(config.get("train_batch_size")),
        "--test_batch_size",
        str(config.get("test_batch_size")),
        "--mask_learning_rate",
        str(config.get("mask_learning_rate")),
        "--scheduler_gamma",
        str(config.get("scheduler_gamma")),
        "--n_epochs",
        str(config.get("n_epochs")),
        "--n_steps",
        str(config.get("n_steps")),
        "--max_new_tokens",
        str(config.get("max_new_tokens")),
        "--output_dir",
        OUTPUT_DIR,
        "--seed",
        str(config.get("seed")),
        "--update_using_mask",
        str(config.get("update_using_mask")),
        "--eval_interval",
        str(config.get("eval_interval")),
        "--save_interval",
        str(config.get("save_interval")),
    ]

    if config.get("dataset_path") is not None:
        default_cmd.extend(["--dataset_path", config.get("dataset_path")])
    if config.get("learned_masks_path") is not None:
        default_cmd.extend(["--learned_masks_path", config.get("learned_masks_path")])
    if config.get("gradients_path") is not None:
        default_cmd.extend(["--gradients_path", config.get("gradients_path")])
    if config.get("force_dataset_generation", False):
        default_cmd.append("--force_dataset_generation")
    if config.get("force_mask_generation", False):
        default_cmd.append("--force_mask_generation")
    if config.get("force_gradients_generation", False):
        default_cmd.append("--force_gradients_generation")

    if isinstance(config.get("reg_coeff"), list) and isinstance(config.get("update_coeff"), float):
        default_cmd.extend(["--update_coeff", str(config.get("update_coeff"))])
        for reg_coeff in config.get("reg_coeff"):
            cmd = default_cmd.copy()
            cmd.extend(["--reg_coeff", str(reg_coeff)])
            subprocess.run(cmd, check=False)
    elif isinstance(config.get("reg_coeff"), float) and isinstance(
        config.get("update_coeff"), list
    ):
        default_cmd.extend(["--reg_coeff", str(config.get("reg_coeff"))])
        for coeff in config.get("update_coeff"):
            cmd = default_cmd.copy()
            cmd.extend(["--update_coeff", str(coeff)])
            subprocess.run(cmd, check=False)
    else:
        default_cmd.extend(["--reg_coeff", str(config.get("reg_coeff"))])
        default_cmd.extend(["--update_coeff", str(config.get("update_coeff"))])
        subprocess.run(default_cmd, check=False)


if __name__ == "__main__":
    main()
