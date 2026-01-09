# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import os
import random

import click
import numpy as np
import torch
from gsm_utils import (
    eval_model,
    load_pipeline_dataset,
    logger,
)
from nnsight import LanguageModel
from pipeline_utils import (
    get_per_component_mask,
    learn_binary_mask,
    train_and_evaluate_n_steps,
)
from transformers import set_seed as transformers_set_seed


@click.command()
@click.option("--seed", type=int, required=True, help="Seed for random number generator.")
@click.option("--model_name", required=True, help="Name of the language model to use")
@click.option(
    "--train_template_idx",
    type=str,
    required=True,
    help="List of template indices to use for training.",
)
@click.option(
    "--train_size",
    type=float,
    required=True,
    help="Proportion of instances to sample for training.",
)
@click.option(
    "--test_size", type=float, required=True, help="Proportion of instances to sample for testing."
)
@click.option(
    "--val_size",
    type=float,
    required=True,
    help="Proportion of instances to sample for validation.",
)
@click.option(
    "--sampled_templates_fraction",
    type=float,
    required=False,
    default=1.0,
    help="Proportion of templates to sample for training.",
)
@click.option(
    "--num_reasoning_clauses",
    type=int,
    required=False,
    default=None,
    help="Number of reasoning clauses to use for training.",
)
@click.option(
    "--dataset_path",
    type=str,
    required=False,
    default=None,
    help="Path to the dataset (string or null).",
)
@click.option(
    "--learned_masks_path",
    type=str,
    required=False,
    default=None,
    help="Path to the learned mask (string or null).",
)
@click.option(
    "--gradients_path",
    type=str,
    required=False,
    default=None,
    help="Path to the gradients (string or null).",
)
@click.option(
    "--train_batch_size", type=int, required=True, help="Batch size for training the mask."
)
@click.option(
    "--test_batch_size", type=int, required=True, help="Batch size for testing the model."
)
@click.option(
    "--mask_learning_rate", type=float, required=True, help="Learning rate for updating the mask."
)
@click.option(
    "--scheduler_gamma",
    type=float,
    required=True,
    help="Decay rate for the learning rate scheduler.",
)
@click.option("--reg_coeff", type=float, required=True, help="Coefficient for updating the mask.")
@click.option(
    "--update_coeff", type=float, required=True, help="Coefficient for updating model weights."
)
@click.option(
    "--update_using_mask",
    type=bool,
    required=True,
    help="Whether to update model weights using the mask.",
)
@click.option("--n_epochs", type=int, required=True, help="Number of epochs for training the mask.")
@click.option(
    "--n_steps", type=int, required=True, help="Number of steps for updating model weights."
)
@click.option(
    "--eval_interval", type=int, required=True, help="Number of gradient steps between evaluations."
)
@click.option(
    "--save_interval",
    type=int,
    required=True,
    help="Number of gradient steps between saving the model.",
)
@click.option(
    "--max_new_tokens", type=int, required=True, help="Maximum number of new tokens to generate."
)
@click.option("--output_dir", type=str, required=True, help="Output directory for results.")
@click.option(
    "--force_dataset_generation",
    is_flag=True,
    default=False,
    help="Force generation of localization dataset and gradients even if they exist.",
)
@click.option(
    "--force_mask_generation",
    is_flag=True,
    default=False,
    help="Force generation of the mask even if it exists.",
)
@click.option(
    "--force_gradients_generation",
    is_flag=True,
    default=False,
    help="Force generation of the gradients even if they exist.",
)
def main(
    seed,
    model_name,
    train_template_idx,
    train_size,
    test_size,
    val_size,
    sampled_templates_fraction,
    num_reasoning_clauses,
    dataset_path,
    learned_masks_path,
    gradients_path,
    train_batch_size,
    test_batch_size,
    mask_learning_rate,
    scheduler_gamma,
    reg_coeff,
    update_coeff,
    update_using_mask,
    n_epochs,
    n_steps,
    eval_interval,
    save_interval,
    max_new_tokens,
    output_dir,
    force_dataset_generation,
    force_mask_generation,
    force_gradients_generation,
):
    # Set environment variable for Python hash seed to ensure deterministic behavior
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
        # Additional deterministic settings for floating-point operations
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Force deterministic algorithms
        torch.use_deterministic_algorithms(mode=True, warn_only=False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    train_template_idx = (
        eval("".join(train_template_idx))
        if train_template_idx not in ["all"]
        else train_template_idx
    )
    job_id = (
        f"seed_{seed}_regcoeff_{reg_coeff}_updatecoeff_{update_coeff}_usemask_{update_using_mask}"
    )
    output_dir = os.path.join(output_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 100)
    logger.info(f"Seed: {seed}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Train template indices: {train_template_idx}")
    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Val size: {val_size}")
    logger.info(f"Sampled templates fraction: {sampled_templates_fraction}")
    logger.info(f"Number of reasoning clauses: {num_reasoning_clauses}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Learned mask path: {learned_masks_path}")
    logger.info(f"Gradients path: {gradients_path}")
    logger.info(f"Train batch size: {train_batch_size}")
    logger.info(f"Test batch size: {test_batch_size}")
    logger.info(f"Mask learning rate: {mask_learning_rate}")
    logger.info(f"Scheduler gamma: {scheduler_gamma}")
    logger.info(f"Regularization coefficient: {reg_coeff}")
    logger.info(f"Update coefficient: {update_coeff}")
    logger.info(f"Update using mask: {update_using_mask}")
    logger.info(f"Number of epochs: {n_epochs}")
    logger.info(f"Number of steps: {n_steps}")
    logger.info(f"Evaluation interval: {eval_interval}")
    logger.info(f"Save interval: {save_interval}")
    logger.info(f"Maximum number of new tokens: {max_new_tokens}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force dataset generation: {force_dataset_generation}")
    logger.info(f"Force mask generation: {force_mask_generation}")
    logger.info(f"Force gradients generation: {force_gradients_generation}")

    logger.info("Loading model...")
    model = LanguageModel(model_name, device_map="auto", torch_dtype=torch.float16, dispatch=True)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.eval()
    logger.info("Model loaded...")

    split_percentages = {"train": train_size, "val": val_size, "test": test_size}
    pipeline_dataset = load_pipeline_dataset(
        localization_data_path=dataset_path,
        train_template_ids=train_template_idx,
        split_percentages=split_percentages,
        seed=seed,
        output_dir=output_dir,
        sampled_templates_fraction=sampled_templates_fraction,
    )
    train_dataset = pipeline_dataset["train"]
    val_dataset = pipeline_dataset["val"]
    test_dataset = pipeline_dataset["test"]
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Learning binary mask
    if update_using_mask:
        unified_learned_mask = learn_binary_mask(
            model=model,
            dataset=train_dataset,
            learned_masks_path=learned_masks_path,
            mask_learning_rate=mask_learning_rate,
            scheduler_gamma=scheduler_gamma,
            reg_coeff=reg_coeff,
            n_epochs=n_epochs,
            force_generation=force_mask_generation,
            seed=seed,
            batch_size=train_batch_size,
            output_dir=output_dir,
        )
    else:
        logger.info("Model need not be updated using the mask. Hence skipping mask learning ...")
        unified_learned_mask = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_results, best_accuracy_step = train_and_evaluate_n_steps(
        model=model,
        train_dataloader=train_dataloader,
        unified_learned_mask=unified_learned_mask,
        max_new_tokens=max_new_tokens,
        val_dataset=val_dataset,
        output_dir=output_dir,
        reg_coeff=reg_coeff,
        update_coeff=update_coeff,
        n_steps=n_steps,
        batch_size=test_batch_size,
        update_using_mask=update_using_mask,
        eval_interval=eval_interval,
        save_interval=save_interval,
    )

    # Load the best model
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Loading the best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
    else:
        logger.info(f"Best model not found at {best_model_path}. Using the base model instead...")

    logger.info("Evaluating the model on the test set...")
    model_output_path = os.path.join(output_dir, "model_outputs_test.json")
    test_results = eval_model(
        model=model,
        tokenizer=model.tokenizer,
        samples=test_dataset,
        max_new_tokens=max_new_tokens,
        batch_size=test_batch_size,
        model_output_path=model_output_path,
    )

    for template_id, result in test_results.items():
        logger.info(f"Template ID: {template_id}, Accuracy: {result['accuracy']}")

    if update_using_mask:
        learned_q_mask, learned_k_mask, learned_v_mask, learned_mlp_mask = get_per_component_mask(
            model, unified_learned_mask
        )
    else:
        learned_q_mask, learned_k_mask, learned_v_mask, learned_mlp_mask = None, None, None, None

    # Log results
    results_path = os.path.join(output_dir, "accuracy_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    log_results = {
        "seed": seed,
        "model_name": model_name,
        "update_using_mask": update_using_mask,
        "train_template_idx": train_template_idx,
        "max_new_tokens": max_new_tokens,
        "mask_learning_rate": mask_learning_rate,
        "reg_coeff": reg_coeff,
        "update_coeff": update_coeff,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "n_k_heads": learned_k_mask.sum().item() if learned_k_mask is not None else -1,
        "n_q_heads": learned_q_mask.sum().item() if learned_q_mask is not None else -1,
        "n_v_heads": learned_v_mask.sum().item() if learned_v_mask is not None else -1,
        "n_mlp_neurons": learned_mlp_mask.sum().item() if learned_mlp_mask is not None else -1,
        "val_results": val_results,
        "best_accuracy_step": best_accuracy_step,
        "test_results": test_results,
        "overall_test_accuracy": sum(result["accuracy"] for result in test_results.values())
        / len(test_results),
    }
    # Save results to file
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing_results = json.load(f)
        existing_results.append(log_results)
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=4)
    else:
        with open(results_path, "w") as f:
            json.dump([log_results], f, indent=4)

    logger.info(f"Results saved to {results_path}...")


if __name__ == "__main__":
    main()
