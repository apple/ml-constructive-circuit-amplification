# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Run a pipeline job controller for parameter sweeps.

This script generates parameter combinations for sweeps.
Adapt it to run jobs sequentially or use a job orchestration tool for parallel execution.
"""

import copy
import hashlib
import json
import os

import click
from tqdm import tqdm  # type: ignore

from math_reasoning.utils import logger

if os.environ.get("ARTIFACT_DIR", None) is not None:
    OUTPUT_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", ""), "experiments")
else:
    OUTPUT_DIR = os.path.join(os.getcwd(), "results", "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def chunk_list(lst, n):
    """Split a list into sublists of maximum length n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def hash_dict(d: dict) -> str:
    """Generate a hash for a dictionary.

    Args:
        d (dict): The dictionary to hash.

    Returns:
        str: a hash of the dictionary as a hex string.

    """
    dict_str = json.dumps(d, sort_keys=True)
    hash_obj = hashlib.sha256(dict_str.encode())
    return hash_obj.hexdigest()[0:32]


@click.command()
@click.option("--base_config_path", type=str)
@click.option(
    "--mask_regularization_coefficients",
    type=str,
    help="Comma-separated mask regularization coefficients",
)
@click.option(
    "--learn_mask_options", type=str, help="Comma-separated learn mask options (True/False)"
)
@click.option("--grad_learning_rates", type=str, help="Comma-separated gradient learning rates")
@click.option("--seeds", type=str, help="Comma-separated seeds")
def main(
    base_config_path: str,
    mask_regularization_coefficients: str,
    learn_mask_options: str,
    grad_learning_rates: str,
    seeds: str,
):
    logger.info("fixing pipeline controller")

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    with open(base_config_path) as f:
        base_config = json.load(f)

    # Parse comma-separated strings into lists
    mask_regularization_coefficients_list = (
        [float(x.strip()) for x in mask_regularization_coefficients.split(",")]
        if mask_regularization_coefficients
        else []
    )
    learn_mask_options_list = (
        [x.strip().lower() == "true" for x in learn_mask_options.split(",")]
        if learn_mask_options
        else []
    )
    grad_learning_rates_list = (
        [float(x.strip()) for x in grad_learning_rates.split(",")] if grad_learning_rates else []
    )
    seeds_list = [int(x.strip()) for x in seeds.split(",")] if seeds else []

    # compute the cross product of all the options skipping mask_reguralization_coefficients
    # if learn_mask_option is False
    pipeline_options = []
    for grad_learning_rate in grad_learning_rates_list:
        for seed in seeds_list:
            for learn_mask in learn_mask_options_list:
                if not learn_mask:
                    child_config = copy.deepcopy(base_config) | {
                        "reg_coeff": 0.01,
                        "update_using_mask": learn_mask,
                        "update_coeff": grad_learning_rate,
                        "seed": seed,
                        "output_dir": OUTPUT_DIR,
                    }
                    pipeline_options.append(child_config)
                else:
                    for mask_reg_coefficient in mask_regularization_coefficients_list:
                        child_config = copy.deepcopy(base_config) | {
                            "reg_coeff": mask_reg_coefficient,
                            "update_using_mask": learn_mask,
                            "update_coeff": grad_learning_rate,
                            "seed": seed,
                            "output_dir": OUTPUT_DIR,
                        }
                        pipeline_options.append(child_config)

    logger.info("="*60)
    logger.warning("This script requires adaptation for job execution.")
    logger.info("="*60)
    logger.info(f"\nGenerated {len(pipeline_options)} configurations to run:")
    
    # Print all configurations that would be run
    for i, pipeline_config in enumerate(pipeline_options, 1):
        logger.info(f"\nConfiguration {i}:")
        logger.info(f"  Reg coeff: {pipeline_config.get('reg_coeff')}")
        logger.info(f"  Update using mask: {pipeline_config.get('update_using_mask')}")
        logger.info(f"  Update coeff: {pipeline_config.get('update_coeff')}")
        logger.info(f"  Seed: {pipeline_config.get('seed')}")
        
        # Build example command
        cmd_parts = ["python math_reasoning/fixing_pipeline.py"]
        for key, val in pipeline_config.items():
            if key in ["force_dataset_generation", "force_gradients_generation", "force_mask_generation"]:
                if val:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key} '{str(val)}'")
        
        logger.info(f"  Command: {' '.join(cmd_parts)}")
    
    logger.info("\n" + "="*60)
    logger.info("To run these experiments locally, you can:")
    logger.info("1. Run math_reasoning/fixing_pipeline.py directly with each config")
    logger.info("2. Use GNU Parallel or similar tools for parallel execution")
    logger.info("3. Adapt this script to use subprocess.Popen for local orchestration")
    logger.info("="*60)


if __name__ == "__main__":
    main()
