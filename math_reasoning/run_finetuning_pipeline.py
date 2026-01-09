# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import click
from filelock import FileLock
from gsm_utils import logger


def run_command(command: list[str]):
    """Run a command and handle errors."""
    logger.info(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        logger.info(f"Command completed successfully: {' '.join(command)}")
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(command)}")
        raise


def construct_finetune_args_from_config(config_params: dict) -> list[str]:
    """Constructs a list of command-line arguments from a config dictionary."""
    args = []
    for param, value in config_params.items():
        args.append(f"--{param}")
        if isinstance(value, list):
            # Format lists for the command line without extra quotes
            args.append(str(value))
        else:
            args.append(str(value))
    return args


@click.command()
@click.option(
    "--config_file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the JSON config file for finetuning.",
)
@click.option("--run_gsm_eval", is_flag=True, help="Run GSM-Symbolic evaluation after finetuning.")
@click.option("--gsm_eval_batch_size", type=int, default=16, help="Batch size for GSM evaluation.")
@click.option("--run_lm_eval", is_flag=True, help="Run LM-Eval evaluation after finetuning.")
@click.option(
    "--lm_eval_benchmarks",
    type=str,
    default="mmlu,math,triviaqa,gsm8k,truthfulqa",
    help="Comma-separated list of benchmarks for LM-Eval.",
)
@click.option("--lm_eval_batch_size", type=int, default=None, help="Batch size for LM-Eval evaluation.")
@click.option(
    "--config_overrides",
    type=str,
    help="JSON string with parameter overrides to apply to base config.",
)
def main(
    config_file: str,
    run_gsm_eval: bool,
    gsm_eval_batch_size: int,
    run_lm_eval: bool,
    lm_eval_benchmarks: str,
    lm_eval_batch_size: int | None,
    config_overrides: str | None,
):
    """Orchestrator script to run finetuning and optionally evaluation using a config file.
    """
    # Load parameters from the config file
    with open(config_file) as f:
        config_params = json.load(f)
    
    # Apply overrides if provided
    if config_overrides:
        try:
            overrides = json.loads(config_overrides)
            config_params.update(overrides)
            logger.info(f"Applied config overrides: {overrides}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --config_overrides: {e}")
            raise click.ClickException(f"Invalid JSON in --config_overrides: {e}")

    # Construct the finetuning command
    finetune_args = construct_finetune_args_from_config(config_params)
    finetune_cmd = ["uv", "run", "python", "math_reasoning/finetune_gsm_symbolic.py"] + finetune_args
    run_command(finetune_cmd)

    # Get parameters needed for evaluation from the config
    if os.environ.get("ARTIFACT_DIR", None) is not None:
        OUTPUT_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", ""), "finetuned_models")
    else:
        OUTPUT_DIR = os.path.join(os.getcwd(), "finetuned_models")
    base_model_name = config_params.get("model_name")
    precision = config_params.get("precision", "fp16")

    if not (run_gsm_eval or run_lm_eval):
        logger.info("No evaluation flags set. Pipeline finished.")
        return

    if not base_model_name:
        logger.error("Required parameter 'model_name' not found in config file. Cannot run evaluations.")
        return

    best_model_path = Path(OUTPUT_DIR) / "best_model"
    if not best_model_path.exists():
        logger.error(f"Best model not found at {best_model_path}. Skipping evaluations.")
        return

    # Atomically update the model config file for the evaluations
    model_config_path = Path("math_reasoning/models/lm_eval_models.json")
    lock_path = model_config_path.with_suffix(".lock")
    
    unique_id = str(uuid.uuid4())[:8]
    model_key = f"finetuned_{base_model_name.replace('/', '_')}_{unique_id}"

    with FileLock(lock_path):
        # Read the existing config
        with open(model_config_path) as f:
            eval_models_config = json.load(f)
        
        # Add the new temporary entry
        eval_models_config[model_key] = {
            "path": str(best_model_path.resolve()),
            "base_model": base_model_name,
            "precision": precision,
            "description": f"Finetuned model from run {unique_id}",
        }
        
        # Write the updated config back to the file
        with open(model_config_path, "w") as f:
            json.dump(eval_models_config, f, indent=2)
        logger.info(f"Added model key '{model_key}' to {model_config_path}")

    # Run GSM evaluation if the flag is set
    if run_gsm_eval:
        gsm_eval_cmd = [
            "uv", "run", "python", "math_reasoning/eval_gsm_symbolic.py",
            "--model_key", model_key,
            "--batch_size", str(gsm_eval_batch_size),
            "--max_new_tokens", "1500",
            "--seed", "42",
        ]
        run_command(gsm_eval_cmd)

    # Run LM-Eval if the flag is set
    if run_lm_eval:
        lm_eval_cmd = [
            "uv", "run", "python", "math_reasoning/run_lm_eval.py",
            "--models", model_key,
            "--benchmarks", lm_eval_benchmarks,
        ]
        if lm_eval_batch_size:
            lm_eval_cmd.extend(["--batch-size", str(lm_eval_batch_size)])
        run_command(lm_eval_cmd)

    logger.info("Finetuning and evaluation pipeline completed.")


if __name__ == "__main__":
    main()