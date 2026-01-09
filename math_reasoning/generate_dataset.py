# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import os
import random

import click
import numpy as np
import torch
import yaml
from gsm_utils import filter_by_template_idx, load_gsm_symbolic, logger
from nnsight import LanguageModel
from pipeline_utils import (
    generate_localization_dataset_via_branching,
    generate_localization_dataset_via_prefix,
)
from transformers import set_seed as transformers_set_seed


@click.command()
@click.option("--config_file", type=str, required=True, help="Config file.")
@click.option("--config_key", type=str, required=True, help="Config Key.")
@click.option("--model_name", type=str, help="Model name to use (overrides config).")
@click.option(
    "--generation_method",
    type=click.Choice(["branching", "prefix"]),
    help="Generation method to use (overrides config).",
)
@click.option(
    "--greedy_dataset_path", type=str, help="Path to save/load the greedy reasoning dataset."
)
@click.option(
    "--non_greedy_dataset_path",
    type=str,
    help="Path to save/load the non-greedy reasoning dataset.",
)
@click.option("--batch_size", type=int, help="Batch size for generation (overrides config).")
@click.option(
    "--temperature_correct",
    type=float,
    help="Temperature for correct greedy generation (overrides config).",
)
@click.option(
    "--temperature_incorrect",
    type=float,
    help="Temperature for incorrect greedy generation (overrides config).",
)
def main(
    config_file: str,
    config_key: str,
    model_name: str,
    generation_method: str,
    batch_size: int,
    greedy_dataset_path: str | None = None,
    non_greedy_dataset_path: str | None = None,
    temperature_correct: float | None = None,
    temperature_incorrect: float | None = None,
):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

        # Load base step configuration from reasoning_clauses
        if config_key not in full_config.get("reasoning_clauses", {}):
            raise ValueError(f"Config key '{config_key}' not found in reasoning_clauses section")

        config = full_config["reasoning_clauses"][config_key].copy()

    logger.info(f"Base config loaded: {config_key}: {config}")

    if model_name is None:
        model_name = config["model_name"]

    if generation_method is None:
        generation_method = "prefix"

    # Apply model-specific overrides if they exist
    if "model_configs" in full_config and model_name in full_config["model_configs"]:
        model_overrides = full_config["model_configs"][model_name]
        logger.info(f"Applying model overrides for {model_name}: {model_overrides}")
        config.update(model_overrides)

    logger.info(f"Final resolved config: {config}")

    if temperature_correct is None:
        temperature_correct = config.get("temperature_correct")

    if temperature_incorrect is None:
        temperature_incorrect = config.get("temperature_incorrect")

    seed = config["seed"]
    train_template_idx = config["train_template_idx"]
    num_reasoning_steps = config.get("num_reasoning_steps")
    output_file_suffix = config["output_file_suffix"]
    max_new_tokens = config["max_new_tokens"]
    force_dataset_generation = True

    model_identifier = model_name.split("/")[-1] if "/" in model_name else model_name
    model_identifier = model_identifier.replace("-", "_")

    output_filename = (
        f"{model_identifier}_{generation_method}_{num_reasoning_steps}_steps_{output_file_suffix}"
    )

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using generation method: {generation_method}")

    # Use artifacts directory for output
    if os.environ.get("ARTIFACT_DIR", None) is not None:
        OUTPUT_DIR = os.environ["ARTIFACT_DIR"]
    else:
        OUTPUT_DIR = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    logger.info(f"output_path: {output_path}")

    # Set environment variable for Python hash seed to ensure deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    transformers_set_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info("Loading model...")
    model = LanguageModel(model_name, device_map="auto", torch_dtype=torch.float16, dispatch=True)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    model.eval()
    logger.info("Model loaded...")

    filtered_ds, cot_prompt = load_gsm_symbolic(num_reasoning_clauses=num_reasoning_steps)
    if train_template_idx == "all":
        train_template_idx = list(set(filtered_ds["original_id"]))
    elif not isinstance(train_template_idx, list):
        raise ValueError(f"Invalid train template index: {train_template_idx}")

    train_samples = filter_by_template_idx(
        ds=filtered_ds, cot_prompt=cot_prompt, template_idx=train_template_idx
    )

    logger.info(f"Total templates: {len(train_template_idx)}")
    logger.info(f"Train samples: {len(train_samples)}")

    # Get localization data
    if generation_method == "branching":
        generate_localization_dataset_via_branching(
            model=model,
            samples=train_samples,
            localization_dataset_path=output_path,
            greedy_dataset_path=greedy_dataset_path,
            non_greedy_dataset_path=non_greedy_dataset_path,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            force_generation=force_dataset_generation,
            temperature_correct=temperature_correct,
            temperature_incorrect=temperature_incorrect,
        )
    elif generation_method == "prefix":
        generate_localization_dataset_via_prefix(
            model=model,
            samples=train_samples,
            localization_dataset_path=output_path,
            greedy_dataset_path=greedy_dataset_path,
            non_greedy_dataset_path=non_greedy_dataset_path,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            force_generation=force_dataset_generation,
            temperature_correct=temperature_correct,
            temperature_incorrect=temperature_incorrect,
        )
    else:
        raise ValueError(
            "Invalid dataset generation method. Choose 'branching' or 'common_prefix'."
        )


if __name__ == "__main__":
    main()
