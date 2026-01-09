# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import gc
import json
import os
import random
import time
from pathlib import Path

import click
import numpy as np
import torch
import wandb
from gsm_utils import (
    extract_final_answer,
    get_device,
    get_target_modules_for_model,
    load_pipeline_dataset,
    logger,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from pipeline_utils import save_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# W&B Configuration - use public wandb by default or environment variable
WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "math-reasoning-circuit-tuning")

if os.environ.get("ARTIFACT_DIR", None) is not None:
    OUTPUT_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", ""), "finetuned_models")
else:
    OUTPUT_DIR = os.path.join(os.getcwd(), "finetuned_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def configure_wandb():
    """Configure W&B by setting environment variables.
    
    The HuggingFace Trainer integration will automatically use these.
    """
    # If WANDB_API_KEY is not set, disable W&B to prevent errors
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set. Disabling W&B integration.")
        os.environ["WANDB_DISABLED"] = "true"
        return

    # Set W&B environment variables for the Trainer to use
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    if "WANDB_BASE_URL" not in os.environ:
        os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
    
    logger.info(f"W&B integration enabled for project '{os.environ['WANDB_PROJECT']}'")
    logger.info(f"W&B URL: {os.environ['WANDB_BASE_URL']}")
    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        logger.info(f"W&B KEY info: {len(api_key)} chars, starts with: {api_key[:7] if len(api_key) >= 7 else '***'}")


def _canon(a: str | None) -> str | None:
    if a is None: 
        return None
    
    if not isinstance(a, str):
        return a
    a = a.strip()
    # try numeric
    try:
        from decimal import Decimal
        return str(Decimal(a))
    except Exception:
        pass
    # fallback: strip common wrappers
    return a.replace(",", "").strip().lower()


class EMEvaluator:
    """Class for fast generation-based EM evaluation."""
    
    def __init__(
        self,
        tokenizer,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        early_stop_markers: list[str] = None,
    ):
        """Initialize EM evaluator.
        
        Args:
            tokenizer: Model tokenizer
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for generation
            early_stop_markers: List of markers to stop generation at

        """
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        # Default early stopping markers - both formats supported
        if early_stop_markers is None:
            self.early_stop_markers = ["#### ", "The final answer is "]
        else:
            self.early_stop_markers = early_stop_markers
            
        logger.info(f"EM Evaluator initialized: max_tokens={max_new_tokens}, batch_size={batch_size}")
        logger.info(f"Early stop markers: {self.early_stop_markers}")
    
    def compute_em_for_dataset(self, model, eval_dataset, save_generations=False, save_dir=None, step=None):
        """Compute EM score for a dataset.
        
        Args:
            model: The model to evaluate
            eval_dataset: Dataset to evaluate on
            save_generations: Whether to save generations to a file
            save_dir: Directory to save generations (required if save_generations=True)
            step: Current training step (used in filename)
            
        Returns:
            EM score (float between 0 and 1)

        """
        # Save original training state to restore after evaluation
        was_training = model.training
        
        try:
            model.eval()
            correct = 0
            total = len(eval_dataset)
            
            logger.info(f"Computing EM score for {total} samples...")
            
            # Collect all generations if requested
            all_generations = [] if save_generations else None
            
            # Calculate number of batches for progress bar
            num_batches = (total + self.batch_size - 1) // self.batch_size
            
            # Process in batches with progress bar
            batch_iterator = tqdm(
                range(0, total, self.batch_size),
                desc="EM Evaluation",
                unit="batch",
                total=num_batches,
                leave=False
            )
            
            for i in batch_iterator:
                batch_end = min(i + self.batch_size, total)
                batch_samples = [eval_dataset[j] for j in range(i, batch_end)]
                
                batch_prompts = [sample["prompt"] for sample in batch_samples]
                batch_targets = [sample["target"] for sample in batch_samples]
                
                # Fast greedy generation
                generated_texts = self._generate_with_early_stop(model, batch_prompts)
                
                # Calculate EM for this batch using existing function
                for idx, (generated, target, prompt) in enumerate(zip(generated_texts, batch_targets, batch_prompts, strict=False)):
                    sample_idx = i + idx
                    
                    if generated is not None:
                        pred_answer = extract_final_answer(generated)
                        true_answer = extract_final_answer(target)
                        
                        is_correct = False
                        if pred_answer is not None and true_answer is not None:
                            if _canon(pred_answer) == _canon(true_answer):
                                correct += 1
                                is_correct = True
                        
                        # Save generation data if requested
                        if save_generations:
                            generation_data = {
                                "sample_idx": sample_idx,
                                "prompt": prompt,
                                "target": target,
                                "generated": generated,
                                "pred_answer": pred_answer,
                                "true_answer": true_answer,
                                "is_correct": is_correct
                            }
                            all_generations.append(generation_data)
                
                # Update progress bar with current metrics
                samples_processed = min(i + self.batch_size, total)
                current_em = correct / samples_processed if samples_processed > 0 else 0.0
                batch_iterator.set_postfix({
                    'EM': f"{current_em:.3f}",
                    'Correct': f"{correct}/{samples_processed}"
                })
            
            em_score = correct / total if total > 0 else 0.0
            logger.info(f"EM Score: {correct}/{total} = {em_score:.4f}")
            
            # Save generations to file if requested
            if save_generations and save_dir and all_generations:
                import json
                import os
                os.makedirs(save_dir, exist_ok=True)
                
                step_str = f"step_{step}" if step is not None else "eval"
                filename = f"eval_generations_{step_str}.json"
                save_path = os.path.join(save_dir, filename)
                
                with open(save_path, 'w') as f:
                    json.dump({
                        "step": step,
                        "em_score": em_score,
                        "correct": correct,
                        "total": total,
                        "generations": all_generations
                    }, f, indent=2)
                logger.info(f"Saved {len(all_generations)} generations to {save_path}")
            
            # Aggressive memory cleanup after EM evaluation
            logger.info("Starting aggressive memory cleanup...")
            
            # 1. Clear model internal caches and states
            if hasattr(model, 'clear_cache'):
                model.clear_cache()
            
            # Clear past key values if model uses attention caching
            if hasattr(model, 'past_key_values'):
                model.past_key_values = None
                
            # Clear any transformer cache
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'past_key_values'):
                model.transformer.past_key_values = None
                
            # 2. Clear tokenizer internal caches
            if hasattr(self.tokenizer, 'clear_cache'):
                self.tokenizer.clear_cache()
                
            # 3. Clear references to large objects first
            if all_generations:
                del all_generations
            
            # 4. Aggressively clear ALL local variables that might reference tensors
            import sys
            frame = sys._getframe()
            locals_copy = dict(frame.f_locals)
            
            # Clear specific known variables
            vars_to_clear = [
                'batch_iterator', 'batch_samples', 'batch_prompts', 'batch_targets',
                'generated_texts', 'toks', 'out', 'attn', 'cont_texts', 'trimmed',
                'batch_end', 'prompt_len_i', 'continuation_i', 't', 'tt', 'pos',
                'generated', 'target', 'prompt', 'pred_answer', 'true_answer',
                'generation_data', 'current_em', 'samples_processed'
            ]
            
            for var_name in vars_to_clear:
                if var_name in locals_copy:
                    try:
                        del frame.f_locals[var_name]
                    except:
                        pass
            
            # Clear any remaining tensor-like objects
            for name, value in list(locals_copy.items()):
                if hasattr(value, 'device') or (hasattr(value, '__len__') and isinstance(value, (list, dict, tuple))):
                    try:
                        del frame.f_locals[name]
                    except:
                        pass
            
            # 5. Multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()
                
            # 6. Aggressive CUDA memory management
            if torch.cuda.is_available():
                # Synchronize first
                torch.cuda.synchronize()
                
                # Clear cache multiple times
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Reset memory stats to help with fragmentation tracking
                torch.cuda.reset_peak_memory_stats()
                
            logger.info("Aggressive memory cleanup completed after EM evaluation")
            return em_score
            
        finally:
            # Restore original training mode after evaluation and cleanup
            if was_training:
                model.train()
                logger.info("Restored model to training mode after EM evaluation")
            else:
                model.eval()
                logger.info("Kept model in eval mode after EM evaluation")
    
    def _generate_with_early_stop(self, model, prompts):
        """Fast greedy generation via model.generate, with per-sample post-trim on early-stop markers.
        Padding is switched to LEFT only inside this function.
        """
        # --- temporarily switch tokenizer to left padding ---
        prev_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"

        # ensure we have a pad token id for batched generation
        pad_id_was_none = False
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            pad_id_was_none = True
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        try:
            # tokenize on CPU; move to device if model is fully on one device
            toks = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                out = model.generate(
                    **toks,
                    do_sample=False,                  # greedy
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            # --- slice continuations per-row using true prompt lengths ---
            # attention_mask has 1s for real tokens (left-padded), so sum = prompt length
            attn = toks["attention_mask"]
            cont_texts = []
            for i in range(out.size(0)):
                prompt_len_i = int(attn[i].sum().item())
                continuation_i = out[i, prompt_len_i:]
                cont_texts.append(self.tokenizer.decode(continuation_i, skip_special_tokens=True))

            # --- per-sample trim at first early-stop marker (doesn't affect others) ---
            trimmed = []
            for t in cont_texts:
                tt = t
                for m in self.early_stop_markers:
                    pos = tt.find(m)
                    if pos != -1:
                        # keep marker + small window to capture the numeric answer
                        tt = tt[: pos + len(m) + 20]
                        break
                trimmed.append(tt)

            return trimmed

        finally:
            # restore tokenizer padding side (and pad token if we set it)
            self.tokenizer.padding_side = prev_side
            if pad_id_was_none:
                # revert to "no pad token" state if that’s how it was
                self.tokenizer.pad_token = None



class ModelParallelTrainer(Trainer):
    """Custom Trainer class that handles device placement for model parallelism and EM evaluation."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to append EM metrics to standard evaluation.
        
        This does NOT affect loss calculation - it just adds EM as an additional metric
        alongside the standard loss-based evaluation.
        """
        # Get standard metrics (loss, perplexity, etc.) - loss calculation unchanged
        #metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        metrics = {}
        # Append EM evaluation if evaluator is available
        if hasattr(self, 'em_evaluator') and self.em_evaluator is not None:
            
            start_time = time.time()
            
            dataset_to_use = eval_dataset if eval_dataset is not None else self.eval_dataset
            
            # Create save directory for generations
            em_score = self.em_evaluator.compute_em_for_dataset(
                self.model,
                dataset_to_use,
                save_generations=True,
                save_dir=OUTPUT_DIR,
                step=self.state.global_step
            )
            
            eval_runtime = time.time() - start_time
            num_samples = len(dataset_to_use)
            samples_per_second = num_samples / eval_runtime if eval_runtime > 0 else 0
            
            metrics[f"{metric_key_prefix}_em"] = em_score
            metrics[f"{metric_key_prefix}_runtime"] = eval_runtime
            metrics[f"{metric_key_prefix}_samples_per_second"] = samples_per_second
            
            # Ensure metrics get logged to wandb using Trainer's logging infrastructure
            self.log(metrics)
            
            loss_val = metrics.get(f"{metric_key_prefix}_loss", None)
            loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else "N/A"
            logger.info(f"Evaluation: loss={loss_str}, EM={em_score:.4f}, runtime={eval_runtime:.2f}s, samples/s={samples_per_second:.2f}")
        
        return metrics
    
    def _get_first_param_device(self, model):
        """Get the target device for model parallelism by getting the device of the first parameter.

        Args:
            model: The model to get device information from

        Returns:
            torch.device: The target device for the model

        """
        # Get the device where the model's first layer is located
        try:
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                # For PEFT models, get the base model's first parameter device
                first_param_device = next(model.base_model.model.parameters()).device
            else:
                # For non-PEFT models
                first_param_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            # Fallback to default device
            first_param_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set the default CUDA device to match the model's device
        if torch.cuda.is_available() and str(first_param_device).startswith("cuda"):
            torch.cuda.set_device(first_param_device)

        return first_param_device

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to handle device placement properly for model parallelism."""
        # Get the target device
        target_device = self._get_first_param_device(model)

        # Move all input tensors to the target device
        prepared_inputs = {
            k: v.to(target_device) if torch.is_tensor(v) else v for k, v in inputs.items()
        }

        # Call the parent method without num_items_in_batch to avoid device issues
        return super().training_step(model, prepared_inputs, None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to handle device placement properly for model parallelism."""
        # Get the target device
        target_device = self._get_first_param_device(model)

        # Move all input tensors to the target device
        prepared_inputs = {
            k: v.to(target_device) if torch.is_tensor(v) else v for k, v in inputs.items()
        }

        # Call the parent method without num_items_in_batch to avoid device issues
        return super().compute_loss(model, prepared_inputs, return_outputs, None)




class GSMFinetuner:
    """Class for finetuning language models on GSM-Symbolic dataset with model parallelism support."""

    def __init__(
        self,
        model_name: str,
        precision: str,
        finetuning_method: str,
        train_template_ids: str | list[int],
        pipeline_dataset_path: str,
        output_dir: str,
        train_split: float = 0.52,
        val_split: float = 0.08,
        test_split: float = 0.4,
        train_batch_size: int = 4,
        eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        seed: int = 42,
        logging_steps: int = 10,
        save_total_limit: int = 10,
        train_on_inp_prompt: bool = False,
        dataloader_num_workers: int = 0,
        model_parallel_size: int = 1,
        fp16: bool = True,
        # EM Evaluation parameters
        em_eval_steps: int = 10,
        em_max_new_tokens: int = 256,
        em_early_stop_markers: list[str] = None,
        val_sample_size: int = None,
    ):
        """Initialize the GSM finetuner with model parallelism.

        Args:
            model_name: Name of the pretrained model to finetune
            precision: Precision to load the model in
            finetuning_method: Either "full" or "lora"
            train_template_ids: Template IDs to use for training
            pipeline_dataset_path: Path to the pipeline dataset
            output_dir: Directory to save the finetuned model
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            train_batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            gradient_accumulation_steps: Number of steps for gradient accumulation
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimization
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            seed: Random seed for reproducibility
            logging_steps: Log every N steps
            save_total_limit: Maximum number of checkpoints to save
            train_on_inp_prompt: Whether to train on the input prompt
            dataloader_num_workers: Number of workers for data loading
            model_parallel_size: Number of GPUs to split the model across
            fp16: Whether to use mixed precision training
            em_eval_steps: Evaluate EM every N steps (default: 50)
            em_max_new_tokens: Maximum tokens to generate for EM eval (default: 256)
            em_early_stop_markers: List of markers to stop generation at
            val_sample_size: Limit validation dataset to N samples for faster evals (default: None = use full dataset)

        """
        self.model_name = model_name
        self.precision = precision
        self.finetuning_method = finetuning_method.lower()
        self.train_template_ids = train_template_ids
        self.pipeline_dataset_path = pipeline_dataset_path
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.seed = seed
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.train_on_inp_prompt = train_on_inp_prompt
        self.dataloader_num_workers = dataloader_num_workers
        self.model_parallel_size = model_parallel_size
        self.fp16 = fp16
        
        # EM Evaluation parameters
        self.em_eval_steps = em_eval_steps
        self.em_max_new_tokens = em_max_new_tokens
        self.em_early_stop_markers = em_early_stop_markers
        self.val_sample_size = val_sample_size

        # Check GPU availability for model parallelism
        self.num_gpus = torch.cuda.device_count()
        if self.model_parallel_size > self.num_gpus:
            raise ValueError(
                f"Model parallel size ({self.model_parallel_size}) exceeds available GPUs ({self.num_gpus})"
            )

        self._set_seeds()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        set_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Additional deterministic settings for floating-point operations
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # Force deterministic algorithms
            torch.use_deterministic_algorithms(mode=True, warn_only=False)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


    def load_model_and_tokenizer(self):
        """Load the pretrained model and tokenizer with model parallelism."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.precision == "fp16" else torch.float32,
        }

        # Configure model parallelism
        if self.model_parallel_size > 1:
            # For model parallelism, use a more explicit device map to avoid device conflicts
            model_kwargs["device_map"] = "balanced"
            logger.info(
                f"Model will be split across {self.model_parallel_size} GPUs using balanced device map"
            )
        else:
            # Single GPU case - ensure consistent device placement
            self.device = get_device()
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device)
            model_kwargs["device_map"] = {"": self.device}

        if "gemma" in self.model_name:
            model_kwargs["attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        if self.finetuning_method == "lora":
            self._prepare_model_for_lora()
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        # Print model device mapping info
        if hasattr(self.model, "hf_device_map"):
            logger.info(f"Model device mapping: {self.model.hf_device_map}")

        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")

    def _prepare_model_for_lora(self):
        """Prepare the model for LoRA finetuning."""
        logger.info("Preparing model for LoRA finetuning...")

        # self.model = prepare_model_for_kbit_training(self.model)

        # Get architecture-specific target modules
        target_modules = get_target_modules_for_model(self.model)
        logger.info(f"Using target modules for {self.model.config.model_type}: {target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
        )

        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()
        logger.info("Model prepared for LoRA finetuning")

    def load_and_preprocess_dataset(self):
        """Load and preprocess the GSM-Symbolic dataset."""
        logger.info("Loading GSM-Symbolic dataset...")

        pipeline_dataset = load_pipeline_dataset(
            train_template_ids=self.train_template_ids,
            split_percentages={
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split,
            },
            seed=self.seed,
            output_dir=str(self.output_dir),
            pipeline_dataset_path=None,
            localization_data_path=None
        )
        self.train_dataset = pipeline_dataset["train"]
        self.val_dataset = pipeline_dataset["val"]
        self.test_dataset = pipeline_dataset["test"]

        # Apply validation dataset sampling if specified
        if self.val_sample_size is not None and self.val_sample_size < len(self.val_dataset):
            logger.info(f"Sampling validation dataset from {len(self.val_dataset)} to {self.val_sample_size} samples")
            
            # Create a reproducible random sample using the same seed
            random.seed(self.seed)
            indices = random.sample(range(len(self.val_dataset)), self.val_sample_size)
            indices.sort()  # Keep in order for reproducibility
            
            # Create sampled dataset
            sampled_val_dataset = [self.val_dataset[i] for i in indices]
            self.val_dataset = sampled_val_dataset
            logger.info(f"Validation dataset sampled to {len(self.val_dataset)} samples")

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def create_data_collator(self):
        """Create a custom data collator for GSM-Symbolic dataset."""

        def collate_fn(batch):
            prompts = [item["prompt"] for item in batch]
            targets = [item["target"] for item in batch]

            # Tokenize prompts to get their lengths
            prompt_tokens = self.tokenizer(prompts, add_special_tokens=False)
            prompt_lengths = [len(tokens) for tokens in prompt_tokens["input_ids"]]

            # Combine prompt and target for causal language modeling task
            texts = [prompt + target for prompt, target in zip(prompts, targets, strict=False)]

            tokenized = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
            )

            # Set labels to be the same as input_ids for causal language modeling
            labels = tokenized["input_ids"].clone()

            # Mask out padding tokens in labels (set to -100)
            labels[tokenized["attention_mask"] == 0] = -100

            # Also mask prompt tokens (only train on target)
            if not self.train_on_inp_prompt:
                for i, prompt_len in enumerate(prompt_lengths):
                    labels[i, :prompt_len] = -100

            tokenized["labels"] = labels

            # Move all tensors to a consistent device to prevent device mismatch
            # if torch.cuda.is_available():
            #     # The Trainer will handle moving data to the correct device, so we don't need to do it here.
            #     # This was causing issues with model parallelism where parts of the model are on different GPUs.
            #     target_device = torch.device("cuda:0")
            #     tokenized = {
            #         k: v.to(target_device) if torch.is_tensor(v) else v
            #         for k, v in tokenized.items()
            #     }

            return tokenized

        return collate_fn

    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments for the Trainer with model parallelism support."""
        run_name = f"gsm_symbolic_{self.finetuning_method}_{self.model_name.split('/')[-1]}_lr{self.learning_rate}_bs{self.train_batch_size}x{self.gradient_accumulation_steps}_ep{self.num_epochs}_r{self.lora_r if self.finetuning_method == 'lora' else 'full'}_mp{self.model_parallel_size}"

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.em_eval_steps,
            save_strategy="steps",
            save_steps=self.em_eval_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_em",
            greater_is_better=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=False,
            logging_dir=str(self.output_dir / "logs"),
            run_name=run_name,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            fp16=self.fp16,
            group_by_length=False,
            save_safetensors=True,
            report_to="wandb",
            # Model parallelism specific settings
            logging_first_step=True,
            log_level="info",
            # Disable data parallel operations
            dataloader_drop_last=False,
        )

        return training_args

    def create_trainer(self) -> ModelParallelTrainer:
        """Create the custom ModelParallelTrainer instance."""
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        data_collator = self.create_data_collator()

        training_args = self.create_training_arguments()

        # Create callbacks list
        callbacks = []
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=3,      # tune
            early_stopping_threshold=0.0,   # optional min improvement
        ))
        
        # Create EM evaluator (always enabled)
        em_evaluator = EMEvaluator(
            tokenizer=self.tokenizer,
            max_new_tokens=self.em_max_new_tokens,
            batch_size=self.eval_batch_size,
            early_stop_markers=self.em_early_stop_markers,
        )
        logger.info("EM evaluation via evaluate() override enabled")

        trainer = ModelParallelTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Attach EM evaluator to trainer for use in evaluate() override
        trainer.em_evaluator = em_evaluator

        return trainer

    def train(self) -> dict:
        """Train the model with model parallelism.

        Returns:
            Dictionary containing training results

        """
        logger.info("Starting training with model parallelism...")

        trainer = self.create_trainer()
        train_result = trainer.train()

        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        logger.info(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")

        logger.info("Saving the best model ...")
        best_model_path = os.path.join(self.output_dir, "best_model")
        # save_model(trainer.model, best_model_path)
        trainer.save_model(output_dir=best_model_path)
        # self.tokenizer.save_pretrained(str(best_model_path))
        logger.info(f"Best model saved to: {best_model_path}")

        return train_result.metrics

    def save_model_info(self):
        """Save model information and configuration."""
        model_info = {
            "model_name": self.model_name,
            "finetuning_method": self.finetuning_method,
            "dataset_splits": {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
            "training_config": {
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.weight_decay,
                "effective_batch_size": self.train_batch_size * self.gradient_accumulation_steps,
            },
            "model_parallel_config": {
                "model_parallel_size": self.model_parallel_size,
                "num_gpus_available": self.num_gpus,
            },
            "lora_config": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
            }
            if self.finetuning_method == "lora"
            else None,
            "seed": self.seed,
        }

        with open(self.output_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model information saved to {self.output_dir / 'model_info.json'}")

    def run_full_pipeline(self) -> dict:
        """Run the complete finetuning pipeline with model parallelism.

        Returns:
            Dictionary containing all results

        """
        logger.info("Starting GSM-Symbolic finetuning pipeline with model parallelism...")
        logger.info(f"Using model parallelism across {self.model_parallel_size} GPU(s)")

        self.load_model_and_tokenizer()

        self.load_and_preprocess_dataset()

        train_results = self.train()

        self.save_model_info()

        results = {
            "train_results": train_results,
            "model_path": str(self.output_dir),
        }

        logger.info("Finetuning pipeline completed successfully!")
        logger.info(f"Model saved to: {self.output_dir}")

        return results


@click.command()
@click.option(
    "--model_name",
    type=str,
    required=True,
    help="Name of the pretrained model to finetune (e.g., 'google/gemma-2-9b-it')",
)
@click.option("--precision", type=str, default="fp32", help="Precision to load the model in")
@click.option(
    "--finetuning_method",
    type=click.Choice(["full", "lora"]),
    required=True,
    help="Finetuning method: 'full' for full finetuning, 'lora' for LoRA finetuning",
)
@click.option(
    "--train_template_ids", type=str, required=True, help="Template IDs to use for training"
)
@click.option(
    "--pipeline_dataset_path", type=str, required=True, help="Path to the pipeline dataset"
)
@click.option("--train_split", type=float, default=0.52, help="Proportion of data for training")
@click.option("--val_split", type=float, default=0.08, help="Proportion of data for validation")
@click.option("--test_split", type=float, default=0.4, help="Proportion of data for testing")
@click.option("--train_batch_size", type=int, default=4, help="Training batch size")
@click.option("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
@click.option(
    "--gradient_accumulation_steps",
    type=int,
    default=4,
    help="Number of steps for gradient accumulation",
)
@click.option("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
@click.option("--num_epochs", type=int, default=3, help="Number of training epochs")
@click.option("--warmup_steps", type=int, default=10, help="Number of warmup steps")
@click.option("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
@click.option("--lora_r", type=int, default=16, help="LoRA rank")
@click.option("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
@click.option("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--logging_steps", type=int, default=2, help="Log every N steps")
@click.option(
    "--save_total_limit", type=int, default=10, help="Maximum number of checkpoints to save"
)
@click.option(
    "--train_on_inp_prompt", type=bool, default=False, help="Whether to train on the prompt"
)
@click.option(
    "--dataloader_num_workers", type=int, default=0, help="Number of workers for data loading"
)
@click.option(
    "--model_parallel_size",
    type=int,
    default=1,
    help="Number of GPUs to split the model across for model parallelism",
)
@click.option("--fp16", type=bool, default=True, help="Whether to use mixed precision training")
@click.option("--em_eval_steps", type=int, default=10, help="Evaluate EM every N steps")
@click.option("--em_max_new_tokens", type=int, default=512, help="Maximum tokens to generate for EM eval")
@click.option("--val_sample_size", type=int, default=None, help="Limit validation dataset to N samples for faster evals")
def main(
    model_name: str,
    precision: str,
    finetuning_method: str,
    train_template_ids: str,
    pipeline_dataset_path: str,
    train_split: float,
    val_split: float,
    test_split: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: int,
    warmup_steps: int,
    weight_decay: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
    logging_steps: int,
    save_total_limit: int,
    train_on_inp_prompt: bool,
    dataloader_num_workers: int,
    model_parallel_size: int,
    fp16: bool,
    em_eval_steps: int,
    em_max_new_tokens: int,
    val_sample_size: int,
):
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    # assert effective_batch_size == 16, f"Effective batch size should be 16, got {effective_batch_size} (train_batch_size: {train_batch_size}, gradient_accumulation_steps: {gradient_accumulation_steps})"

    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Train, validation, and test splits must sum to 1.0, got {total_split}")

    train_template_ids = (
        eval("".join(train_template_ids))
        if train_template_ids not in ["all"]
        else train_template_ids
    )

    # Configure W&B at the start of the main execution
    configure_wandb()

    logger.info(f"Model name: {model_name}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Finetuning method: {finetuning_method}")
    logger.info(f"Train template IDs: {train_template_ids}")
    logger.info(f"Pipeline dataset path: {pipeline_dataset_path}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Train split: {train_split}")
    logger.info(f"Val split: {val_split}")
    logger.info(f"Test split: {test_split}")
    logger.info(f"Train batch size: {train_batch_size}")
    logger.info(f"Eval batch size: {eval_batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Model parallel size: {model_parallel_size}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"LoRA rank: {lora_r}")
    logger.info(f"LoRA alpha: {lora_alpha}")
    logger.info(f"LoRA dropout: {lora_dropout}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Logging steps: {logging_steps}")
    logger.info(f"Save total limit: {save_total_limit}")
    logger.info(f"Train on prompt: {train_on_inp_prompt}")
    logger.info(f"Dataloader num workers: {dataloader_num_workers}")
    logger.info(f"FP16: {fp16}")
    logger.info(f"EM eval steps: {em_eval_steps}")
    logger.info(f"EM max new tokens: {em_max_new_tokens}")
    logger.info(f"Val sample size: {val_sample_size}")

    finetuner = GSMFinetuner(
        model_name=model_name,
        precision=precision,
        finetuning_method=finetuning_method,
        output_dir=OUTPUT_DIR,
        train_template_ids=train_template_ids,
        pipeline_dataset_path=pipeline_dataset_path,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        seed=seed,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        train_on_inp_prompt=train_on_inp_prompt,
        dataloader_num_workers=dataloader_num_workers,
        model_parallel_size=model_parallel_size,
        fp16=fp16,
        em_eval_steps=em_eval_steps,
        em_max_new_tokens=em_max_new_tokens,
        val_sample_size=val_sample_size,
    )

    results = finetuner.run_full_pipeline()

    logger.info("\n" + "=" * 50)
    logger.info("FINETUNING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {finetuning_method}")
    logger.info(f"Training loss: {results['train_results']['train_loss']:.4f}")
    logger.info(f"Model parallelism across {model_parallel_size} GPU(s)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
