# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import os
import random

import click
import numpy as np
import torch
from datasets import Dataset
from gsm_utils import (
    eval_model,
    filter_by_template_idx,
    load_gsm_symbolic,
    logger,
)
from nnsight import LanguageModel
from run_lm_eval import (
    load_benchmark_config,
    load_model_config,
    load_pytorch_checkpoint_as_hf_model,
    prepare_local_model,
    setup_output_directory,
)
from transformers import set_seed as transformers_set_seed

# Use artifact directory if available
if os.environ.get("ARTIFACT_DIR", None) is not None:
    OUTPUT_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", ""), "results")
else:
    OUTPUT_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"OUTPUT DIR: {OUTPUT_DIR}")


@click.command()
@click.option("--model_key", type=str, required=True, help="Name of the model to evaluate")
@click.option(
    "--num_reasoning_steps",
    type=int,
    multiple=False,
    help="Number of reasoning steps to use for training. If not provided, will run for steps 2-7",
)
@click.option("--batch_size", type=int, required=True, help="Batch size for inference")
@click.option(
    "--max_new_tokens", type=int, required=True, help="Maximum number of new tokens to generate"
)
@click.option("--seed", type=int, required=True, help="Seed for random number generator")
@click.option("--dataset_path", type=str, required=False, help="Path to the dataset")
@torch.no_grad()
def main(model_key, num_reasoning_steps, batch_size, max_new_tokens, seed, dataset_path):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    transformers_set_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(mode=True, warn_only=False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Model key: {model_key}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Maximum number of new tokens: {max_new_tokens}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Dataset path: {dataset_path}")

    logger.info("Loading model...")

    # Load configurations
    model_config = load_model_config(model_key)

    # Determine model identifier and prepare local models if needed
    model = None
    if "path" in model_config:
        # Local model - prepare it first
        model_identifier = prepare_local_model(model_config["path"], model_config)

        # Handle all local model types: .pth, HF directories, and PEFT adapters
        is_pth = model_identifier.endswith(".pth")
        is_adapter = os.path.exists(os.path.join(model_identifier, "adapter_config.json"))
        is_hf = os.path.exists(os.path.join(model_identifier, "config.json"))
        
        if is_pth or is_adapter or is_hf:
            # Validate base_model for .pth and PEFT models
            requires_base_model = is_pth or is_adapter
            
            if requires_base_model and "base_model" not in model_config:
                raise ValueError(
                    "For .pth checkpoints and PEFT adapters, 'base_model' must be specified in model config. "
                    "Example: 'base_model': 'microsoft/phi-2'"
                )

            # Load using unified model loading function
            base_model_name = model_config.get("base_model")
            hf_model, tokenizer = load_pytorch_checkpoint_as_hf_model(
                model_identifier, base_model_name, model_config
            )
            #model = LanguageModel(hf_model, tokenizer=tokenizer, device_map="auto")
            model = LanguageModel(
                hf_model,
                tokenizer=tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
                dispatch=True,
                trust_remote_code=True,
            )
            
            # Set pad token for tokenizer (required for batch evaluation)
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            logger.info("Using standard GSM-Symbolic dataset")
        else:
            raise ValueError(f"Unknown local model format: {model_identifier}")

    else:
        # Remote HuggingFace model
        model = LanguageModel(
            model_config["name"],
            device_map="auto",
            torch_dtype=torch.float16,
            dispatch=True,
            trust_remote_code=True,
            tokenizer_kwargs={"trust_remote_code": True},
        )
        tokenizer = model.tokenizer
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model.eval()
    logger.info("Model loaded...")

    if not dataset_path:
        # Run evaluation for each reasoning step
        # Determine which reasoning steps to run
        if not num_reasoning_steps:
            reasoning_steps = list(range(2, 8))  # 2 to 7 inclusive
            logger.info(f"Running for reasoning steps: {reasoning_steps}")
        else:
            reasoning_steps = list(num_reasoning_steps)
            logger.info(f"Running for reasoning steps: {reasoning_steps}")

        for current_steps in reasoning_steps:
            logger.info(f"\n=== Evaluating with {current_steps} reasoning steps ===")

            filtered_ds, cot_prompt = load_gsm_symbolic(num_reasoning_clauses=current_steps)
            all_template_ids = list(set(filtered_ds["original_id"]))
            samples = filter_by_template_idx(filtered_ds, cot_prompt, all_template_ids)
            logger.info(f"Total number of samples: {len(samples)}")

            logger.info("Evaluating model...")
            model_output_path = os.path.join(OUTPUT_DIR, "model_outputs.json")
            template_accs = eval_model(
                model, tokenizer, samples, max_new_tokens, batch_size, model_output_path
            )
            for template_id in template_accs:
                logger.info(
                    f"Template ID: {template_id} | Correct: {template_accs[template_id]['correct']} | Total: {template_accs[template_id]['total']} | Accuracy: {template_accs[template_id]['accuracy']:.2f}"
                )

            # Get model name for saving results
            if "name" in model_config:
                model_name_short = model_config["name"].split("/")[-1]
            else:
                # For local models, use the model key or a default name
                model_name_short = model_key.replace("/", "_")

            # Save results to local directory
            if os.environ.get("ARTIFACT_DIR", None) is not None:
                artifacts_dir = os.environ["ARTIFACT_DIR"]
            else:
                artifacts_dir = os.path.join(os.getcwd(), "artifacts")

            model_artifacts_dir = os.path.join(artifacts_dir, "model_evals", model_name_short)
            os.makedirs(model_artifacts_dir, exist_ok=True)

            result_path = os.path.join(model_artifacts_dir, f"{current_steps}.json")
            with open(result_path, "w") as f:
                json.dump(template_accs, f, indent=4)
            logger.info(f"Results saved to: {result_path}")

    else:
        logger.info(f"Evaluating model {model_key} on dataset {dataset_path}...")
        
        if dataset_path.startswith(("http://", "https://")):
            raise ValueError(f"Remote URLs not supported: {dataset_path}")
        
        dataset = []
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))

        # Filter test samples from the dataset
        test_samples = [sample for sample in dataset if sample["split"] == "test"]
        test_samples = Dataset.from_list(test_samples)
        logger.info(f"Total number of test samples: {len(test_samples)}")

        model_output_path = os.path.join(
            OUTPUT_DIR, f"model_outputs_{model_key.split('/')[-1]}.json"
        )
        template_accs = eval_model(
            model, tokenizer, test_samples, max_new_tokens, batch_size, model_output_path
        )
        for template_id in template_accs:
            logger.info(
                f"Template ID: {template_id} | Correct: {template_accs[template_id]['correct']} | Total: {template_accs[template_id]['total']} | Accuracy: {template_accs[template_id]['accuracy']:.2f}"
            )

        overall_accuracy = sum(
            template_accs[template_id]["correct"] / template_accs[template_id]["total"]
            for template_id in template_accs
        ) / len(template_accs)
        logger.info(f"Overall accuracy: {overall_accuracy:.2f}")

        # Save the template accuracies to a file
        with open(
            os.path.join(OUTPUT_DIR, f"template_accs_{model_key.split('/')[-1]}.json"), "w"
        ) as f:
            json.dump(
                {"overall_accuracy": overall_accuracy, "template_accs": template_accs}, f, indent=4
            )


if __name__ == "__main__":
    main()
