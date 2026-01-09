#!/usr/bin/env python3
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Main script for running lm-evaluation-harness benchmarks on configured models."""

import gc
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import lm_eval
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_cached_remote_file, logger, report_metrics


def get_device_args() -> str:
    """Get device arguments for model_args based on platform compatibility."""
    if torch.cuda.is_available():
        return "auto"
    elif platform.system() == "Darwin":
        return "mps"
    return "auto"


def cleanup_gpu_memory():
    """Clean up GPU memory between model evaluations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleaned up")


def load_pytorch_checkpoint_as_hf_model(
    checkpoint_path: str, base_model_name: str, model_config: dict = None
) -> tuple[Any, Any]:
    """Load a model from various formats as HuggingFace model instance.

    Extended to handle:
    - .pth PyTorch checkpoint files
    - Local HuggingFace model directories (from trainer.save_model())
    - PEFT/LoRA adapter models

    Args:
        checkpoint_path: Path to the model (.pth file, HF directory, or PEFT adapter directory)
        base_model_name: HuggingFace model name to use as base (required for .pth and PEFT)
        model_config: Optional model configuration dictionary

    Returns:
        Tuple of (model, tokenizer) instances

    """
    # Auto-detect model type and delegate to appropriate loader
    if checkpoint_path.endswith(".pth"):
        logger.info(f"Detected .pth checkpoint file: {os.path.basename(checkpoint_path)}")
        return _load_pth_checkpoint(checkpoint_path, base_model_name)
    elif os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        logger.info(f"Detected PEFT/LoRA adapter model: {os.path.basename(checkpoint_path)}")
        return _load_peft_model(checkpoint_path, base_model_name, model_config)
    elif os.path.exists(os.path.join(checkpoint_path, "config.json")):
        logger.info(f"Detected HuggingFace model directory: {os.path.basename(checkpoint_path)}")
        return _load_hf_directory(checkpoint_path, base_model_name)
    else:
        raise ValueError(f"Unknown model type for path: {checkpoint_path}. "
                        f"Expected .pth file, directory with adapter_config.json (PEFT), "
                        f"or directory with config.json (HuggingFace).")


def _load_pth_checkpoint(checkpoint_path: str, base_model_name: str) -> tuple[Any, Any]:
    """Load a .pth PyTorch checkpoint file
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        base_model_name: HuggingFace model name to use as base
        
    Returns:
        Tuple of (model, tokenizer) instances

    """
    logger.info(f"Loading .pth checkpoint from base model: {base_model_name}")

    try:
        device = get_device_args()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        
        logger.info("Loading checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights into model
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Checkpoint weights loaded successfully (strict mode)")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e):
                logger.info("Strict loading failed, attempting non-strict loading...")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.info(f"Missing keys during loading: {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.info(f"Unexpected keys during loading: {len(unexpected_keys)} keys")
            else:
                raise

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        logger.info("Successfully loaded .pth checkpoint and tokenizer")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load .pth checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load .pth checkpoint: {str(e)}") from e


def _load_peft_model(adapter_path: str, base_model_name: str, model_config: dict = None) -> tuple[Any, Any]:
    """Load PEFT/LoRA adapter model.
    
    Args:
        adapter_path: Path to the PEFT adapter directory
        base_model_name: HuggingFace model name to use as base
        model_config: Optional model configuration
        
    Returns:
        Tuple of (model, tokenizer) instances

    """
    logger.info(f"Loading PEFT adapter with base model: {base_model_name}")
    
    try:
        # Import PEFT with error handling
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "PEFT library is required for loading adapter models. "
                "Install with: pip install peft"
            ) from e
        
        device = get_device_args()
        
        logger.info("Loading base model for PEFT adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        
        logger.info("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        logger.info("Successfully loaded PEFT model and tokenizer")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load PEFT adapter: {str(e)}")
        raise RuntimeError(f"Failed to load PEFT adapter model: {str(e)}") from e


def _load_hf_directory(model_path: str, base_model_name: str = None) -> tuple[Any, Any]:
    """Load local HuggingFace model directory.
    
    Args:
        model_path: Path to the HuggingFace model directory
        base_model_name: Optional base model name for tokenizer fallback
        
    Returns:
        Tuple of (model, tokenizer) instances

    """
    logger.info("Loading HuggingFace model from local directory")
    
    try:
        device = get_device_args()
        
        logger.info("Loading model from HuggingFace directory...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        
        # Try to load tokenizer from local directory first, fallback to base model
        tokenizer = None
        logger.info("Loading tokenizer from HuggingFace directory...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            logger.info("Successfully loaded tokenizer from local directory")
        except Exception as tokenizer_error:
            logger.info(f"Failed to load tokenizer from local directory: {tokenizer_error}")
            
            if base_model_name:
                logger.info(f"Falling back to base model tokenizer: {base_model_name}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name,
                        trust_remote_code=True,
                    )
                    logger.info("Successfully loaded tokenizer from base model")
                except Exception as base_tokenizer_error:
                    logger.error(f"Failed to load tokenizer from base model: {base_tokenizer_error}")
                    raise
            else:
                logger.error("No base model provided for tokenizer fallback")
                raise tokenizer_error
        
        logger.info("Successfully loaded HuggingFace model and tokenizer")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load HuggingFace directory: {str(e)}")
        raise RuntimeError(f"Failed to load HuggingFace directory: {str(e)}") from e


def prepare_local_model(
    model_path: str, model_config: dict[str, Any], cache_dir: str | None = None
) -> str:
    """Prepare local model for evaluation (stub for customization).

    This function can be customized to handle local model preparation tasks such as:
    - Downloading model files from remote storage

    Args:
        model_path: Original path from model configuration
        model_config: Full model configuration dictionary

    Returns:
        Final model path to use for loading

    """
    logger.debug(f"Preparing local model: {model_path}")
    logger.debug(f"Cache dir: {cache_dir}")
    try:
        local_path = get_cached_remote_file(model_path, cache_dir=cache_dir)
        logger.debug(f"Local path resolved to: {local_path}")
        return local_path
    except Exception as e:
        logger.debug(f"Exception in prepare_local_model: {e}")
        raise


def load_model_config(model_key: str) -> dict[str, Any]:
    """Load model configuration by key."""
    config_path = Path("math_reasoning/models/lm_eval_models.json")
    with open(config_path) as f:
        models = json.load(f)

    if model_key not in models:
        available_keys = list(models.keys())
        raise ValueError(f"Model key '{model_key}' not found. Available keys: {available_keys}")

    return models[model_key]


def load_benchmark_config(benchmark_key: str) -> dict[str, Any]:
    """Load benchmark configuration by key."""
    config_path = Path("math_reasoning/configs/lm_eval_benchmarks.json")
    with open(config_path) as f:
        benchmarks = json.load(f)

    if benchmark_key not in benchmarks:
        available_keys = list(benchmarks.keys())
        raise ValueError(
            f"Benchmark key '{benchmark_key}' not found. Available keys: {available_keys}"
        )

    return benchmarks[benchmark_key]


def setup_output_directory(model_key: str, benchmark_key: str) -> Path:
    """Create and return output directory for results."""
    # Use ARTIFACT_DIR if available, otherwise create local results dir
    base_dir = os.environ.get("ARTIFACT_DIR", "math_reasoning/results")
    output_dir = Path(base_dir) / "lm_eval" / f"{model_key}_{benchmark_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_evaluation(
    model_key: str,
    benchmark_key: str,
    limit: int | None = None,
    batch_size: int | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Run lm-evaluation-harness evaluation.

    Args:
        model_key: Key from lm_eval_models.json
        benchmark_key: Key from lm_eval_benchmarks.json
        limit: Limit number of samples (for testing)
        batch_size: Override batch size (uses benchmark default if None)
        output_dir: Directory to save results
        verbose: Enable verbose logging

    Returns:
        Results dictionary from lm-eval

    """
    # Load configurations
    model_config = load_model_config(model_key)
    benchmark_config = load_benchmark_config(benchmark_key)

    # Setup output directory
    if output_dir is None:
        output_dir = setup_output_directory(model_key, benchmark_key)

    # Determine model identifier and prepare local models if needed
    model_instance = None
    if "path" in model_config:
        # Local model - prepare it first
        model_type = "Local"
        model_identifier = prepare_local_model(model_config["path"], model_config, cache_dir)

        # Handle all local model types: .pth, HF directories, and PEFT adapters
        if (model_identifier.endswith(".pth") or
            os.path.exists(os.path.join(model_identifier, "adapter_config.json")) or
            os.path.exists(os.path.join(model_identifier, "config.json"))):
            
            # Validate base_model for .pth and PEFT models
            requires_base_model = (model_identifier.endswith(".pth") or
                                 os.path.exists(os.path.join(model_identifier, "adapter_config.json")))
            
            if requires_base_model and "base_model" not in model_config:
                raise ValueError(
                    "For .pth checkpoints and PEFT adapters, 'base_model' must be specified in model config. "
                    "Example: 'base_model': 'microsoft/phi-2'"
                )

            # Load using unified model loading function
            base_model_name = model_config.get("base_model")
            model_instance, tokenizer = load_pytorch_checkpoint_as_hf_model(
                model_identifier, base_model_name, model_config
            )
            
            # Update identifier for logging
            if base_model_name:
                model_identifier = f"{base_model_name} (from {os.path.basename(model_identifier)})"
            else:
                model_identifier = f"Local model ({os.path.basename(model_identifier)})"
        else:
            raise ValueError(f"Unknown local model format: {model_identifier}")
    else:
        # HuggingFace model
        model_type = "HuggingFace"
        model_identifier = model_config["name"]

    logger.info("Starting evaluation:")
    logger.info(f"  Model: {model_identifier} ({model_type})")
    logger.info(f"  Benchmark: {benchmark_config['description']}")
    logger.info(f"  Tasks: {benchmark_config['tasks']}")
    logger.info(f"  Output: {output_dir}")

    # Override batch size if provided, otherwise use benchmark default
    if batch_size is None:
        batch_size = benchmark_config["batch_size"]

    # Prepare evaluation arguments
    if model_instance is not None:
        # Wrap model instance in lm_eval HFLM wrapper (for .pth checkpoints)
        logger.debug("Wrapping model instance in HFLM wrapper...")
        try:
            # Create HFLM wrapper with the loaded model instance
            lm_eval_model = HFLM(
                pretrained=model_instance,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=get_device_args(),
            )
            logger.debug("HFLM wrapper created successfully")
        except Exception as e:
            logger.debug(f"Failed to create HFLM wrapper: {e}")
            raise

        eval_args = {
            "model": lm_eval_model,
            "tasks": benchmark_config["tasks"],
            "num_fewshot": benchmark_config["num_fewshot"],
            "batch_size": batch_size,
            "log_samples": True,
        }
        logger.info("Using wrapped model instance for evaluation")
    else:
        # Use HuggingFace model path approach
        # Build model arguments with platform-specific device handling
        model_args_parts = [
            f"pretrained={model_identifier}",
            f"dtype={model_config['precision']}",
        ]

        # Add custom loading arguments if present
        if "loading_args" in model_config:
            for key, value in model_config["loading_args"].items():
                if isinstance(value, bool):
                    model_args_parts.append(f"{key}={str(value).lower()}")
                elif isinstance(value, str):
                    model_args_parts.append(f"{key}={value}")
                else:
                    model_args_parts.append(f"{key}={value}")
            logger.info(f"  Custom loading args: {model_config['loading_args']}")

        device = get_device_args()
        model_args_parts.append(f"device={device}")
        logger.info(f"  Device: {device}")

        eval_args = {
            "model": "hf",
            "model_args": ",".join(model_args_parts),
            "tasks": benchmark_config["tasks"],
            "num_fewshot": benchmark_config["num_fewshot"],
            "batch_size": batch_size,
            "log_samples": True,
        }
        logger.info("Using HuggingFace model path approach")

    # Add limit if specified
    if limit is not None:
        eval_args["limit"] = limit
    elif benchmark_config["limit"] is not None:
        eval_args["limit"] = benchmark_config["limit"]

    logger.info(f"Evaluation arguments: {eval_args}")

    # Add detailed debugging before lm_eval call
    logger.debug("=== About to call lm_eval.simple_evaluate ===")
    logger.debug(f"Model instance type: {type(model_instance) if model_instance else 'None'}")
    if model_instance is not None:
        logger.debug(f"Model device: {next(model_instance.parameters()).device}")
        logger.debug(f"Model dtype: {next(model_instance.parameters()).dtype}")
    logger.debug("=== END DEBUG ===")

    try:
        logger.debug("Calling lm_eval.simple_evaluate...")
        # Run evaluation using lm_eval.simple_evaluate
        results = lm_eval.simple_evaluate(**eval_args)
        logger.debug("lm_eval.simple_evaluate completed successfully")

        # Extract clean results (just the metrics)
        clean_results: dict[str, Any] = {}
        if "results" in results:
            for task_name, task_results in results["results"].items():
                clean_results[task_name] = {}
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float, str, bool)):
                        clean_results[task_name][metric_name] = metric_value
                    elif hasattr(metric_value, "item"):  # numpy scalar
                        clean_results[task_name][metric_name] = metric_value.item()
                    else:
                        clean_results[task_name][metric_name] = str(metric_value)

        # Save summary with clean results and evaluation parameters
        summary = {
            "model_key": model_key,
            "model_name": model_identifier,
            "model_type": model_type,
            "benchmark_key": benchmark_key,
            "benchmark_description": benchmark_config["description"],
            "tasks": benchmark_config["tasks"],
            "timestamp": datetime.now().isoformat(),
            "evaluation_params": {
                "num_fewshot": benchmark_config["num_fewshot"],
                "batch_size": batch_size,
                "limit": limit,
                "precision": model_config["precision"],
                "device": get_device_args(),
                "loading_args": model_config.get("loading_args", {}),
            },
            "results": clean_results,
            "version": results.get("version", "unknown"),
        }

        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save full results as JSON with numpy conversion
        def convert_numpy(obj):
            """Convert numpy and torch objects to JSON-serializable format."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int_, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return convert_numpy(obj.detach().cpu().numpy())
            elif isinstance(obj, torch.dtype):
                return str(obj)
            elif hasattr(obj, "dtype") and hasattr(obj, "item"):
                return obj.item()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        # Save full results
        full_results_file = output_dir / "full_results.json"
        try:
            serializable_results = convert_numpy(results)
            with open(full_results_file, "w") as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Full results saved to: {full_results_file}")
        except Exception as json_error:
            # Fallback to text if JSON serialization fails
            fallback_file = output_dir / "raw_results.txt"
            with open(fallback_file, "w") as f:
                f.write(str(results))
            logger.warning(
                f"Full results saved as text (JSON failed: {json_error}): {fallback_file}"
            )

        if results.get("results"):
            metrics = {}
            for task_name, task_results in results["results"].items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        metrics[f"{task_name}_{metric_name}"] = metric_value

            if metrics:
                report_metrics(metrics)

        logger.info("Evaluation completed successfully!")
        logger.info(f"Full results saved to: {full_results_file}")
        logger.info(f"Summary saved to: {summary_file}")

        return results

    except Exception as e:
        logger.error(f"Exception in run_evaluation: {e}", exc_info=True)
        raise


@click.command()
@click.option("--models", required=True, help="Comma-separated model keys from lm_eval_models.json")
@click.option(
    "--benchmarks",
    required=True,
    help="Comma-separated benchmark keys from lm_eval_benchmarks.json",
)
@click.option("--limit", type=int, help="Limit number of samples for testing")
@click.option(
    "--batch-size", type=int, help="Override batch size (uses benchmark default if not specified)"
)
@click.option("--output-dir", help="Output directory for results")
@click.option("--quiet", is_flag=True, help="Reduce output verbosity")
@click.option("--cache-dir", help="Cache dir for downloading models", default="/tmp")
def main(
    models: str,
    benchmarks: str,
    limit: int,
    batch_size: int,
    output_dir: str,
    quiet: bool,
    cache_dir: str,
):
    """Run lm-evaluation-harness benchmarks."""
    # Parse models and benchmarks from comma-separated strings
    model_list = [model.strip() for model in models.split(",")]
    benchmark_list = [benchmark.strip() for benchmark in benchmarks.split(",")]

    logger.info(f"Running evaluation for {len(model_list)} model(s): {model_list}")
    logger.info(f"Running evaluation for {len(benchmark_list)} benchmark(s): {benchmark_list}")

    try:
        for i, model_key in enumerate(model_list):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"EVALUATING MODEL {i + 1}/{len(model_list)}: {model_key}")
            logger.info(f"{'=' * 60}")

            for j, benchmark_key in enumerate(benchmark_list):
                logger.info(f"\n{'-' * 50}")
                logger.info(f"  Running benchmark {j + 1}/{len(benchmark_list)}: {benchmark_key}")
                logger.info(f"{'-' * 50}")

                try:
                    logger.debug(f"Setting up output directory for {model_key} and {benchmark_key}")
                    # Setup model and benchmark specific output directory
                    if output_dir:
                        model_benchmark_output_dir = (
                            Path(output_dir) / f"{model_key}_{benchmark_key}"
                        )
                    else:
                        model_benchmark_output_dir = None  # Use default from run_evaluation

                    logger.debug(f"About to call run_evaluation for {model_key} on {benchmark_key}")
                    results = run_evaluation(
                        model_key=model_key,
                        benchmark_key=benchmark_key,
                        limit=limit,
                        batch_size=batch_size,
                        output_dir=model_benchmark_output_dir,
                        verbose=not quiet,
                        cache_dir=cache_dir,
                    )
                    logger.debug(
                        f"run_evaluation completed successfully for {model_key} on {benchmark_key}"
                    )

                    # Print individual model and benchmark summary
                    if results.get("results"):
                        logger.info(
                            f"\n=== {model_key.upper()} - {benchmark_key.upper()} SUMMARY ==="
                        )
                        for task_name, task_results in results["results"].items():
                            logger.info(f"\nTask: {task_name}")
                            for metric_name, metric_value in task_results.items():
                                if isinstance(metric_value, (int, float)):
                                    logger.info(f"  {metric_name}: {metric_value:.4f}")

                except Exception as e:
                    logger.error(
                        f"Exception caught in inner loop for {model_key} on {benchmark_key}: {e}",
                        exc_info=True,
                    )
                    logger.error(f"Error evaluating {model_key} on {benchmark_key}: {e}")
                    continue

                cleanup_gpu_memory()

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
