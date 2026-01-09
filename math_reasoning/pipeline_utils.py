# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import gc
import json
import os
import pickle
from collections import defaultdict

import torch
from datasets import Dataset
from gsm_utils import eval_model, extract_final_answer, extract_until_first_answer, logger
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from utils import load_json, report_metrics


def save_model(model: LanguageModel | AutoModelForCausalLM, path: str) -> None:
    """Save the model to disk.

    Args:
        model: The model to save.
        path: The path to save the model to.

    Returns:
        None

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def train_and_evaluate_n_steps(
    model: LanguageModel,
    train_dataloader: torch.utils.data.DataLoader,
    unified_learned_mask: torch.Tensor | None,
    max_new_tokens: int,
    val_dataset: Dataset,
    output_dir: str,
    reg_coeff: float,
    update_coeff: float,
    n_steps: int,
    batch_size: int,
    update_using_mask: bool,
    eval_interval: int,
    save_interval: int,
) -> tuple[dict[str, dict[int, float]], int]:
    """Perform n gradient steps on the model and evaluates after each step.

    Args:
        model: The language model to compute gradients for.
        train_dataloader: DataLoader containing the localization dataset.
        unified_learned_mask: Unified learned mask for the model.
        max_new_tokens: Maximum number of new tokens to generate.
        val_dataset: Samples to evaluate on.
        output_dir: Output directory for results.
        reg_coeff: Regularization coefficient for the mask.
        update_coeff: Coefficient for updating the model weights.
        n_steps: Number of gradient steps to compute.
        batch_size: Batch size for testing the model.
        update_using_mask: Whether to update model weights using the mask.
        eval_interval: Number of gradient steps between evaluations.
        save_interval: Number of gradient steps between saving the model.

    Returns:
        A dictionary containing validation metrics after each gradient step.

    """
    if update_using_mask:
        assert unified_learned_mask is not None, (
            "unified_learned_mask must be provided when update_using_mask is True"
        )
        learned_q_mask, learned_k_mask, learned_v_mask, learned_mlp_mask = get_per_component_mask(
            model, unified_learned_mask
        )

        assert learned_q_mask.shape == (
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
        ), (
            f"Expected learned_q_mask shape: (num_hidden_layers, num_attention_heads): ({model.config.num_hidden_layers}, {model.config.num_attention_heads}), but got {learned_q_mask.shape}"
        )
        assert learned_k_mask.shape == (
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
        ), (
            f"Expected learned_k_mask shape: (num_hidden_layers, num_key_value_heads): ({model.config.num_hidden_layers}, {model.config.num_key_value_heads}), but got {learned_k_mask.shape}"
        )
        assert learned_v_mask.shape == (
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
        ), (
            f"Expected learned_v_mask shape: (num_hidden_layers, num_key_value_heads): ({model.config.num_hidden_layers}, {model.config.num_key_value_heads}), but got {learned_v_mask.shape}"
        )
        assert learned_mlp_mask.shape == (
            model.config.num_hidden_layers,
            model.config.intermediate_size,
        ), (
            f"Expected learned_mlp_shape: (num_hidden_layers, intermediate_size): ({model.config.num_hidden_layers}, {model.config.intermediate_size}), but got {learned_mlp_mask.shape}"
        )
    else:
        learned_q_mask, learned_k_mask, learned_v_mask, learned_mlp_mask = None, None, None, None

    logger.info("Computing base model performance...")
    model_output_path = os.path.join(output_dir, "model_outputs_base.json")
    base_metrics = eval_model(
        model,
        model.tokenizer,
        val_dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        model_output_path=model_output_path,
    )
    for template_id in base_metrics:
        logger.info(
            f"Template: {template_id} | Correct: {base_metrics[template_id]['correct']} | Total: {base_metrics[template_id]['total']} | Accuracy: {base_metrics[template_id]['accuracy']:.2f}"
        )
    overall_accuracy = sum(metric["accuracy"] for metric in base_metrics.values()) / len(
        base_metrics
    )
    logger.info(f"Overall accuracy: {overall_accuracy:.2f}")
    logger.info("=" * 60)
    results = defaultdict(dict)
    results["step_0"] = base_metrics

    best_accuracy = float("-inf")
    best_accuracy_step = 0
    logger.info(f"Starting {n_steps} gradient updates...")
    for i in range(n_steps):
        logger.info(f"Gradient computation at step {i + 1} started...")
        grads = compute_gradients(
            model=model, dataloader=train_dataloader, update_using_mask=update_using_mask
        )
        logger.info(f"Gradient computation at step {i + 1} completed.")
        logger.info(f"Updating model weights at step {i + 1}...")
        model = update_model(
            model=model,
            learned_k_mask=learned_k_mask,
            learned_q_mask=learned_q_mask,
            learned_v_mask=learned_v_mask,
            learned_mlp_mask=learned_mlp_mask,
            update_using_mask=update_using_mask,
            coeff=update_coeff,
            grads=grads,
        )
        logger.info(f"Model weights updated at step {i + 1}.")

        if (i + 1) % eval_interval == 0 or (i < 10 and (i + 1) % 2 == 0):
            logger.info(f"Evaluating model at step {i + 1}...")
            model_output_path = os.path.join(output_dir, f"model_outputs_step_{i + 1}.json")
            metrics = eval_model(
                model=model,
                tokenizer=model.tokenizer,
                samples=val_dataset,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                model_output_path=model_output_path,
            )

            logger.info(f"Step {i + 1} results:")
            for template_id in metrics:
                logger.info(
                    f"Template: {template_id} | Correct: {metrics[template_id]['correct']} | Total: {metrics[template_id]['total']} | Accuracy: {metrics[template_id]['accuracy']:.2f}"
                )
            overall_accuracy = sum(metric["accuracy"] for metric in metrics.values()) / len(metrics)
            logger.info(f"Overall accuracy: {overall_accuracy:.2f}")
            results[f"step_{i + 1}"] = metrics

            # Plot validation accuracy over steps
            report_metrics(
                {
                    f"val_acc_reg_coeff_{reg_coeff}": overall_accuracy,
                }
            )

            if overall_accuracy > best_accuracy:
                logger.info(
                    f"New model accuracy ({overall_accuracy:.2f}) is better than best accuracy ({best_accuracy:.2f}). Saving new best model..."
                )
                save_model(model, os.path.join(output_dir, "best_model.pth"))
                best_accuracy = overall_accuracy
                best_accuracy_step = i + 1

        if (i + 1) % save_interval == 0:
            logger.info(f"Saving model at step {i + 1}...")
            save_model(model, os.path.join(output_dir, f"model_step_{i + 1}.pth"))

        logger.info("=" * 60)

    return results, best_accuracy_step


@torch.no_grad()
def update_model(
    model: LanguageModel,
    learned_k_mask: torch.Tensor | None,
    learned_q_mask: torch.Tensor | None,
    learned_v_mask: torch.Tensor | None,
    learned_mlp_mask: torch.Tensor | None,
    update_using_mask: bool,
    coeff: float,
    grads: dict[str, dict[int, torch.Tensor] | torch.Tensor],
) -> LanguageModel:
    """Update the model's attention heads based on the learned mask and gradients.

    Args:
        model (LanguageModel): The language model to update.
        learned_k_mask (torch.Tensor): The mask indicating which K heads to update.
        learned_q_mask (torch.Tensor): The mask indicating which Q heads to update.
        learned_v_mask (torch.Tensor): The mask indicating which V heads to update.
        learned_mlp_mask (torch.Tensor): The mask indicating which MLP neurons to update.
        update_using_mask (bool): Whether to update model weights using the mask.
        coeff (float): The coefficient to scale the gradients.
        grads (dict): The gradients to update the model.

    Returns:
        LanguageModel: The updated model with modified attention heads.

    """
    d_head = getattr(
        model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
    )
    logger.info(f"d_head: {d_head}")

    if update_using_mask:
        for l in range(model.config.num_hidden_layers):
            layer_device = get_layer_device(model, l)

            if learned_mlp_mask[l].sum() > 0:
                model.model.layers[l].mlp.up_proj.weight[learned_mlp_mask[l] == 1.0] += (
                    coeff * grads["mlp"][l][learned_mlp_mask[l] == 1.0].to(layer_device)
                )

            for h in range(model.config.num_attention_heads):
                if learned_q_mask[l, h] == 1.0:
                    q_start_idx, q_end_idx = h * d_head, (h + 1) * d_head
                    model.model.layers[l].self_attn.q_proj.weight[q_start_idx:q_end_idx, :] += (
                        coeff * grads["q_proj"][l][q_start_idx:q_end_idx, :].to(layer_device)
                    )

            for h in range(model.config.num_key_value_heads):
                if learned_k_mask[l, h] == 1.0:
                    k_start_idx, k_end_idx = h * d_head, (h + 1) * d_head
                    model.model.layers[l].self_attn.k_proj.weight[k_start_idx:k_end_idx, :] += (
                        coeff * grads["k_proj"][l][k_start_idx:k_end_idx, :].to(layer_device)
                    )
                if learned_v_mask[l, h] == 1.0:
                    v_start_idx, v_end_idx = h * d_head, (h + 1) * d_head
                    model.model.layers[l].self_attn.v_proj.weight[v_start_idx:v_end_idx, :] += (
                        coeff * grads["v_proj"][l][v_start_idx:v_end_idx, :].to(layer_device)
                    )
    else:
        for name, param in model.named_parameters():
            param += coeff * grads[name].to(param.device)

    return model


def compute_gradients(
    model: LanguageModel,
    dataloader: torch.utils.data.DataLoader,
    update_using_mask: bool,
) -> dict[str, dict[int, torch.Tensor] | torch.Tensor]:
    """Compute gradients for the model based on the localization dataset.

    Args:
        model: The language model to compute gradients for.
        dataloader: DataLoader containing the localization dataset.
        update_using_mask: Whether to update model weights using the mask.

    Returns:
        A dictionary containing the computed gradients.

    """
    grads = defaultdict(dict)
    dataset_size = len(dataloader.dataset)
    model.eval()
    model.zero_grad()
    for batch in tqdm(dataloader, total=len(dataloader), desc="Computing gradients"):
        batch_size = len(batch["prefix"])
        prefix = batch["prefix"]
        desired_token = batch["desired_token"]
        undesired_token = batch["undesired_token"]

        if not torch.is_tensor(desired_token):
            desired_token = torch.tensor(desired_token, dtype=torch.long)
        else:
            desired_token = desired_token.to(dtype=torch.long)

        if not torch.is_tensor(undesired_token):
            undesired_token = torch.tensor(undesired_token, dtype=torch.long)
        else:
            undesired_token = undesired_token.to(dtype=torch.long)

        vocab_size = model.config.vocab_size
        for i in range(batch_size):
            assert desired_token[i].item() < vocab_size, (
                f"Desired token {desired_token[i].item()} is not in the vocab of size {vocab_size}"
            )
            assert undesired_token[i].item() < vocab_size, (
                f"Undesired token {undesired_token[i].item()} is not in the vocab of size {vocab_size}"
            )

        prefix_tokens = model.tokenizer(
            prefix, padding=True, padding_side="left", return_tensors="pt"
        )
        prefix_tokens = {k: v.to(model.device) for k, v in prefix_tokens.items()}
        last_token_logits = model(**prefix_tokens).logits[:, -1]

        assert last_token_logits.shape == (batch_size, model.config.vocab_size), (
            f"Expected last_token_logits shape: (batch_size, vocab_size): ({batch_size}, {model.config.vocab_size}), but got {last_token_logits.shape}"
        )

        index = torch.arange(batch_size, device=last_token_logits.device)
        desired_token = desired_token.to(last_token_logits.device)
        undesired_token = undesired_token.to(last_token_logits.device)
        desired_token_logits = last_token_logits[index, desired_token]
        undesired_token_logits = last_token_logits[index, undesired_token]

        assert desired_token_logits.shape == (batch_size,), (
            f"Expected desired_token_logits shape: (batch_size,): ({batch_size}), but got {desired_token_logits.shape}"
        )
        assert undesired_token_logits.shape == (batch_size,), (
            f"Expected undesired_token_logits shape: (batch_size,): ({batch_size}), but got {undesired_token_logits.shape}"
        )

        loss = desired_token_logits - undesired_token_logits
        loss = -loss.sum()
        loss = loss / dataset_size
        loss.backward()

    del last_token_logits, prefix, desired_token, undesired_token, loss
    gc.collect()
    torch.cuda.empty_cache()

    if update_using_mask:
        for l in range(model.config.num_hidden_layers):
            grads["k_proj"][l] = model.model.layers[l].self_attn.k_proj.weight.grad.cpu()
            grads["q_proj"][l] = model.model.layers[l].self_attn.q_proj.weight.grad.cpu()
            grads["v_proj"][l] = model.model.layers[l].self_attn.v_proj.weight.grad.cpu()
            grads["mlp"][l] = model.model.layers[l].mlp.up_proj.weight.grad.cpu()
    else:
        for name, param in model.named_parameters():
            grads[name] = param.grad.cpu()

    return grads


def get_layer_device(model, layer_idx):
    """Get the device of a specific layer"""
    return next(model.model.layers[layer_idx].parameters()).device


def get_per_component_mask(
    model: LanguageModel,
    unified_mask: torch.Tensor,
    layer_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the per-component mask for the model.

    Args:
        model: The language model to use for learning the mask.
        unified_mask: The unified mask to split into per-component masks.
        layer_idx: The index of the layer to get the per-component mask for.

    Returns:
        A tuple (Q, K, V, MLP) of per-component masks for the model.

    """
    assert unified_mask.shape[0] == model.config.num_hidden_layers
    assert (
        unified_mask.shape[1]
        == 2 * model.config.num_key_value_heads
        + model.config.num_attention_heads
        + model.config.intermediate_size
    )

    q_mask_start = 0
    k_mask_start = model.config.num_attention_heads
    v_mask_start = model.config.num_attention_heads + model.config.num_key_value_heads
    mlp_mask_start = model.config.num_attention_heads + 2 * model.config.num_key_value_heads

    if layer_idx == -1:
        q_masks = unified_mask[:, q_mask_start:k_mask_start]
        k_masks = unified_mask[:, k_mask_start:v_mask_start]
        v_masks = unified_mask[:, v_mask_start:mlp_mask_start]
        mlp_masks = unified_mask[:, mlp_mask_start:]
    else:
        q_masks = unified_mask[layer_idx, q_mask_start:k_mask_start]
        k_masks = unified_mask[layer_idx, k_mask_start:v_mask_start]
        v_masks = unified_mask[layer_idx, v_mask_start:mlp_mask_start]
        mlp_masks = unified_mask[layer_idx, mlp_mask_start:]

    return q_masks, k_masks, v_masks, mlp_masks


def learn_binary_mask(
    model: LanguageModel,
    dataset: Dataset,
    learned_masks_path: str,
    mask_learning_rate: float = 1e-3,
    reg_coeff: float = 0.1,
    scheduler_gamma: float = 0.9,
    n_epochs: int = 50,
    force_generation: bool = True,
    seed: int = 42,
    batch_size: int = 8,
    output_dir: str | None = None,
) -> torch.Tensor:
    """Learn a binary mask for attention heads based on the localization dataset.

    More details on the algorithm can be found in the following link: https://dcm.baulab.info.
    The algorithm is based on the following paper: https://arxiv.org/abs/2307.03637

    Args:
        model: The language model to use for learning the mask.
        dataset: Dataset of samples from the localization dataset.
        learned_masks_path: Path to save/load the learned attention mask.
        mask_learning_rate: Learning rate for updating the mask.
        reg_coeff: Coefficient for updating the mask.
        scheduler_gamma: Decay rate for the learning rate scheduler.
        n_epochs: Number of epochs to train the mask.
        force_generation: If True, forces generation of the mask even if it exists.
        seed: Seed for deterministic training.
        batch_size: Batch size for training.
        output_dir: Directory to save the learned masks.

    Returns:
        A binary mask indicating which attention heads to update.

    """
    # Return mask from disk if it exists and force_generation is False
    if not force_generation and learned_masks_path is not None:
        logger.info(f"Learned masks already exists. Loading from {learned_masks_path}...")
        if learned_masks_path.startswith(("http://", "https://")):
            raise ValueError(f"Remote URLs not supported: {learned_masks_path}")
        
        with open(learned_masks_path, "rb") as f:
            learned_masks = pickle.load(f)

        is_binary = torch.all(
            (learned_masks["unified_mask"] == 0.0) | (learned_masks["unified_mask"] == 1.0)
        )
        if not is_binary:
            logger.info("Loaded learned masks are not binary. Deleting them...")
            del learned_masks
            gc.collect()
            torch.cuda.empty_cache()
        else:
            return learned_masks["unified_mask"]

    logger.info("Learning new binary masks...")
    learned_masks_dir = os.path.join(output_dir, "learned_masks", f"reg_coeff_{reg_coeff}")
    os.makedirs(learned_masks_dir, exist_ok=True)

    d_head = getattr(
        model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
    )
    # logger.info(f"d_head: {d_head}")

    components_per_layer = (
        model.config.num_attention_heads  # Q heads
        + model.config.num_key_value_heads * 2  # K, V heads
        + model.config.intermediate_size  # MLP neurons
    )
    total_components = model.config.num_hidden_layers * components_per_layer
    initial_components = total_components

    logger.info(f"Total components to learn: {total_components}")
    logger.info(
        f"Components per layer: {components_per_layer} ({model.config.num_attention_heads} Q heads + {model.config.num_key_value_heads} K, V heads + {model.config.intermediate_size} MLP neurons)"
    )
    logger.info("=" * 60)

    # Run early stopping if no change in mask components for 20% of total batches
    patience = len(dataset) // batch_size
    if len(dataset) % batch_size != 0:
        patience += 1
    patience = int(patience * 0.2)  # 20% of total batches

    # Create masks on the same device as each layer
    unified_mask_list = []
    for layer_idx in range(model.config.num_hidden_layers):
        layer_device = get_layer_device(model, layer_idx)
        layer_mask = torch.ones(
            components_per_layer, requires_grad=True, dtype=torch.float32, device=layer_device
        )
        unified_mask_list.append(layer_mask)

    optimizer = torch.optim.Adam(unified_mask_list, lr=mask_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=scheduler_gamma)

    # Reset early stopping variables for this iteration
    n_prev_components = initial_components
    early_stopping_triggered = False

    for epoch in range(n_epochs):
        no_change_count = 0
        epoch_target_loss = 0
        print(
            f"Epoch {epoch + 1} started with learning rate {scheduler.get_last_lr()[0]:.3e} and reg_coeff {reg_coeff}"
        )

        # Create a deterministic generator for this epoch to ensure consistent shuffling
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(seed + epoch)

        # Create a new DataLoader for this epoch with deterministic shuffling
        epoch_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=epoch_generator,
        )

        for batch_index, batch in enumerate(epoch_dataloader):
            prefix = batch["prefix"]

            desired_token = batch["desired_token"]
            undesired_token = batch["undesired_token"]

            with model.trace() as tracer:
                with tracer.invoke(prefix):
                    for layer_idx in range(model.config.num_hidden_layers):
                        # Extract masks for this layer (already on correct device)
                        layer_mask = unified_mask_list[layer_idx]

                        # Split the unified mask into component-specific masks
                        q_mask_start = 0
                        k_mask_start = model.config.num_attention_heads
                        v_mask_start = (
                            model.config.num_attention_heads + model.config.num_key_value_heads
                        )
                        mlp_mask_start = (
                            model.config.num_attention_heads + 2 * model.config.num_key_value_heads
                        )
                        q_masks = layer_mask[q_mask_start:k_mask_start]
                        k_masks = layer_mask[k_mask_start:v_mask_start]
                        v_masks = layer_mask[v_mask_start:mlp_mask_start]
                        mlp_masks = layer_mask[mlp_mask_start:]

                        for head_idx in range(model.config.num_attention_heads):
                            # Apply q_mask to q_proj output
                            q_dims = torch.arange(head_idx * d_head, (head_idx + 1) * d_head)
                            q_output = (
                                model.model.layers[layer_idx]
                                .self_attn.q_proj.output[:, -1, q_dims]
                                .clone()
                            )  # (batch_size, d_head)
                            q_masked = q_output * (2 * q_masks[head_idx])
                            model.model.layers[layer_idx].self_attn.q_proj.output[:, -1, q_dims] = (
                                q_masked
                            )

                        for head_idx in range(model.config.num_key_value_heads):
                            # Apply k_mask to k_proj output
                            k_dims = torch.arange(head_idx * d_head, (head_idx + 1) * d_head)
                            k_output = (
                                model.model.layers[layer_idx]
                                .self_attn.k_proj.output[:, -1, k_dims]
                                .clone()
                            )  # (batch_size, d_head)
                            k_masked = k_output * (2 * k_masks[head_idx])
                            model.model.layers[layer_idx].self_attn.k_proj.output[:, -1, k_dims] = (
                                k_masked
                            )

                            # Apply v_mask to v_proj output
                            v_dims = torch.arange(head_idx * d_head, (head_idx + 1) * d_head)
                            v_output = (
                                model.model.layers[layer_idx]
                                .self_attn.v_proj.output[:, -1, v_dims]
                                .clone()
                            )  # (batch_size, d_head)
                            v_masked = v_output * (2 * v_masks[head_idx])
                            model.model.layers[layer_idx].self_attn.v_proj.output[:, -1, v_dims] = (
                                v_masked
                            )

                        mlp_intermediate = (
                            model.model.layers[layer_idx].mlp.down_proj.input[:, -1].clone()
                        )
                        masked_intermediate = 2 * mlp_masks * mlp_intermediate
                        model.model.layers[layer_idx].mlp.down_proj.input[:, -1] = (
                            masked_intermediate
                        )

                    out = model.lm_head.output[:, -1].save()

            # Calculate target loss
            desired_token_logit = out[:, desired_token]
            undesired_token_logit = out[:, undesired_token]
            target_loss = desired_token_logit - undesired_token_logit
            target_loss = -target_loss.mean()

            # Calculate L1 regularization term for each layer mask on its device
            l1_loss = 0
            reference_device = target_loss.device
            for layer_idx, layer_mask in enumerate(unified_mask_list):
                layer_l1 = reg_coeff * torch.norm(layer_mask, p=1)
                # Ensure deterministic device transfer
                if layer_l1.device != reference_device:
                    layer_l1 = layer_l1.to(reference_device)
                l1_loss += layer_l1

            loss = target_loss + l1_loss
            epoch_target_loss += target_loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Get the current learned mask for logging and early stopping
            # Collect masks from all devices to CPU for counting
            learned_masks = []
            for layer_mask in unified_mask_list:
                learned_mask = layer_mask.clamp(0, 1).round().cpu()
                learned_masks.append(learned_mask)
            unified_learned_mask = torch.stack(learned_masks, dim=0)

            # Count total current components
            n_current_components = unified_learned_mask.sum().item()
            if n_current_components == n_prev_components:
                no_change_count += 1
            else:
                no_change_count = 0
            n_prev_components = n_current_components

            # Count individual current components
            q_masks, k_masks, v_masks, mlp_masks = get_per_component_mask(
                model, unified_learned_mask
            )
            q_components = q_masks.sum().item()
            k_components = k_masks.sum().item()
            v_components = v_masks.sum().item()
            mlp_components = mlp_masks.sum().item()

            logger.info(
                f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_index + 1}/{len(epoch_dataloader)}, "
                f"Loss: {loss:.3f}, Target Loss: {target_loss:.3f}, "
                f"L1 Loss: {l1_loss:.3f}, Q: {q_components}, K: {k_components}, V: {v_components}, MLP: {mlp_components}, no_change_count: {no_change_count}"
            )

            report_metrics(
                {
                    f"loss_{reg_coeff}": loss.item(),
                    f"target_loss_{reg_coeff}": target_loss.item(),
                    f"l1_loss_{reg_coeff}": l1_loss.item(),
                    f"q_components_{reg_coeff}": q_components,
                    f"k_components_{reg_coeff}": k_components,
                    f"v_components_{reg_coeff}": v_components,
                    f"mlp_components_{reg_coeff}": mlp_components,
                }
            )

            if no_change_count >= patience:
                at_initial_state = n_current_components == initial_components

                if not at_initial_state:
                    print(
                        f"Early stopping triggered after {no_change_count} steps with no change in mask components"
                    )
                    early_stopping_triggered = True

                    # Clean up
                    del out, desired_token, undesired_token, target_loss, l1_loss, loss
                    gc.collect()
                    torch.cuda.empty_cache()

                    break

            # Clamp mask values to [0, 1] after every gradient step
            with torch.no_grad():
                for layer_mask in unified_mask_list:
                    layer_mask.data.clamp_(0, 1)

            del out, desired_token, undesired_token, target_loss, l1_loss, loss
            gc.collect()
            torch.cuda.empty_cache()

        scheduler.step()

        epoch_target_loss /= len(epoch_dataloader)
        logger.info(f"Epoch {epoch + 1} target loss: {epoch_target_loss:.3f}")
        report_metrics(
            {
                f"epoch_loss_{reg_coeff}": epoch_target_loss,
            }
        )

        # Save mask every 5 epochs or when early stopping is triggered
        if (epoch + 1) % 5 == 0 or early_stopping_triggered:
            logger.info(f"Saving unified mask at epoch {epoch + 1}")
            current_unified_mask_cpu = []
            for layer_mask in unified_mask_list:
                current_mask = layer_mask.clamp(0, 1).round().cpu()
                current_unified_mask_cpu.append(current_mask)

            current_unified_mask = torch.stack(current_unified_mask_cpu, dim=0)
            q_masks, k_masks, v_masks, mlp_masks = get_per_component_mask(
                model, current_unified_mask
            )
            q_components = q_masks.sum().item()
            k_components = k_masks.sum().item()
            v_components = v_masks.sum().item()
            mlp_components = mlp_masks.sum().item()

            mask_filename = f"epoch_{epoch + 1}.pkl"
            mask_filepath = os.path.join(learned_masks_dir, mask_filename)

            mask_data = {
                "epoch": epoch + 1,
                "unified_mask": current_unified_mask,
                "components": current_unified_mask.sum().item(),
                "q_components": q_components,
                "k_components": k_components,
                "v_components": v_components,
                "mlp_components": mlp_components,
                "learning_rate": mask_learning_rate,
            }

            with open(mask_filepath, "wb") as f:
                pickle.dump(mask_data, f)

            logger.info(
                f"Saved unified mask at epoch {epoch + 1} with {current_unified_mask.sum().item()} components to {mask_filepath}"
            )

            if early_stopping_triggered:
                break

    final_unified_mask_cpu = []
    for layer_mask in unified_mask_list:
        final_mask = layer_mask.clamp(0, 1).round().cpu()
        final_unified_mask_cpu.append(final_mask)

    final_unified_mask = torch.stack(final_unified_mask_cpu, dim=0)
    final_q_masks, final_k_masks, final_v_masks, final_mlp_masks = get_per_component_mask(
        model, final_unified_mask
    )

    logger.info(f"Final unified mask: {final_unified_mask.sum().item()} components")
    logger.info(f"Final q masks: {final_q_masks.sum().item()} components")
    logger.info(f"Final k masks: {final_k_masks.sum().item()} components")
    logger.info(f"Final v masks: {final_v_masks.sum().item()} components")
    logger.info(f"Final mlp masks: {final_mlp_masks.sum().item()} components")

    return final_unified_mask


@torch.no_grad()
def generate_non_greedy_reasoning(
    model: LanguageModel,
    prompt: str,
    temperature: float,
    target: str,
    get_correct_sample: bool,
    max_new_tokens: int = 200,
) -> "tuple[str, torch.Tensor] | tuple[None, None]":
    """Generate non-greedy reasoning output from the model using batch generation.

    Args:
        model: The language model to use for generation.
        prompt: The input prompt for the model.
        temperature: The temperature for sampling.
        target: The target answer for comparison.
        get_correct_sample: Whether the result should match the ground truth or not.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        A tuple containing the generated output text and the tokens, or (None, None) if not found.

    """
    ground_truth = extract_final_answer(target)
    batch_size = 10
    max_attempts = 1

    for batch_attempt in range(max_attempts):
        logger.info("Non greedy batch attempt: %d (batch_size=%d)", batch_attempt + 1, batch_size)

        batch_prompts = [prompt] * batch_size
        tokens = model.tokenizer(batch_prompts, return_tensors="pt", padding=True)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=model.tokenizer.eos_token_id,
        ):
            batch_out = model.generator.output.save()

        for i in range(batch_size):
            generated_tokens = batch_out[i]

            # Remove padded tokens from the generated tokens
            n_padded_tokens = (tokens["attention_mask"][i] == 0).sum().item()
            generated_tokens = generated_tokens[n_padded_tokens:]

            # Extract reasoning trace of the final question
            prompt_len = tokens["attention_mask"][i].sum().item()
            reasoning_trace = model.tokenizer.decode(
                generated_tokens[prompt_len:], skip_special_tokens=True
            )
            reasoning_trace = extract_until_first_answer(reasoning_trace)
            reasoning_trace = prompt + " " + reasoning_trace
            final_answer = extract_final_answer(reasoning_trace)

            if (final_answer != ground_truth and not get_correct_sample) or (
                final_answer == ground_truth and get_correct_sample
            ):
                del batch_out
                torch.cuda.empty_cache()

                filtered_generated_tokens = model.tokenizer.encode(
                    reasoning_trace, return_tensors="pt"
                )
                filtered_generated_tokens = filtered_generated_tokens[0]

                return reasoning_trace, filtered_generated_tokens

        del batch_out
        torch.cuda.empty_cache()

    return None, None


def generate_greedy_reasoning_traces(
    model: LanguageModel, samples: Dataset, batch_size: int, max_new_tokens: int
) -> list[dict]:
    """Generate greedy reasoning outputs for a dataset.

    This function generates greedy reasoning outputs for a dataset.

    Args:
        model: The language model to use for generation.
        samples: Dataset of samples from the GSM-Symbolic dataset.
        batch_size: Batch size for generation.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        List of reasoning traces generated greedily.

    """
    logger.info("Generating greedy reasoning outputs...")
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=False)
    generation_results = []
    for batch in tqdm(dataloader, desc="Generating greedy reasoning outputs"):
        prompts = batch["prompt"]
        targets = batch["target"]
        template_ids = batch["template_id"]
        instance_ids = batch["instance_id"]

        tokens = model.tokenizer(prompts, return_tensors="pt", padding=True)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.no_grad():
            with model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            ):
                out = model.generator.output.save()

        for i in range(len(prompts)):
            generated_tokens = out[i]

            # Remove padded tokens from the generated tokens
            n_padded_tokens = (tokens["attention_mask"][i] == 0).sum().item()
            generated_tokens = generated_tokens[n_padded_tokens:]

            # Extract reasoning trace of the final question
            prompt_len = tokens["attention_mask"][i].sum().item()
            reasoning_trace = model.tokenizer.decode(
                generated_tokens[prompt_len:], skip_special_tokens=True
            )
            reasoning_trace = extract_until_first_answer(reasoning_trace)
            reasoning_trace = prompts[i] + " " + reasoning_trace

            filtered_generated_tokens = model.tokenizer.encode(reasoning_trace)

            generation_results.append(
                {
                    "prompt": prompts[i],
                    "prompt_len": prompt_len,
                    "target": targets[i],
                    "generated_tokens": filtered_generated_tokens,
                    "reasoning_trace": reasoning_trace,
                    "correct_answer": extract_final_answer(reasoning_trace)
                    == extract_final_answer(targets[i]),
                    "template_id": template_ids[i].item()
                    if isinstance(template_ids[i], torch.Tensor)
                    else template_ids[i],
                    "instance_id": instance_ids[i].item()
                    if isinstance(instance_ids[i], torch.Tensor)
                    else instance_ids[i],
                }
            )

        del out
        torch.cuda.empty_cache()

    return generation_results


def generate_non_greedy_reasoning_traces(
    model: LanguageModel,
    greedy_traces: list[dict],
    max_new_tokens: int,
    temperature_correct: float | None = 1.2,
    temperature_incorrect: float | None = 0.8,
) -> list[dict]:
    """Generate non-greedy reasoning outputs for a dataset.

    This function generates non-greedy reasoning outputs for a dataset.

    Args:
        model: The language model to use for generation.
        greedy_traces: List of reasoning traces generated greedily.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature_correct: Temperature for correct greedy generation.
        temperature_incorrect: Temperature for incorrect greedy generation.

    Returns:
        List of reasoning traces generated non-greedily.

    """
    logger.info("Generating non-greedy reasoning outputs...")
    non_greedy_generation_results = []
    for sample in tqdm(greedy_traces, desc="Generating non-greedy reasoning outputs"):
        prompt = sample["prompt"]
        target = sample["target"]
        prompt_len = sample["prompt_len"]
        is_greedy_correct = sample["correct_answer"]
        template_id = sample["template_id"]
        instance_id = sample["instance_id"]

        non_greedy_result = generate_non_greedy_reasoning(
            model=model,
            prompt=prompt,
            temperature=temperature_correct if is_greedy_correct else temperature_incorrect,
            target=target,
            get_correct_sample=not is_greedy_correct,
            max_new_tokens=max_new_tokens,
        )
        non_greedy_trace, non_greedy_generated_tokens = non_greedy_result

        non_greedy_generation_results.append(
            {
                "prompt": prompt,
                "prompt_len": prompt_len,
                "target": target,
                "generated_tokens": non_greedy_generated_tokens.tolist()
                if non_greedy_generated_tokens is not None
                else None,
                "reasoning_trace": non_greedy_trace,
                "correct_answer": (
                    extract_final_answer(non_greedy_trace) == extract_final_answer(target)
                )
                if non_greedy_trace is not None
                else None,
                "template_id": template_id.item()
                if isinstance(template_id, torch.Tensor)
                else template_id,
                "instance_id": instance_id.item()
                if isinstance(instance_id, torch.Tensor)
                else instance_id,
            }
        )

    return non_greedy_generation_results


def find_branching_info(
    model: LanguageModel,
    non_greedy_generated_tokens: torch.Tensor,
    question_len: int,
    target: str,
    max_new_tokens: int,
    greedy_answer_correct: bool,
) -> "tuple[int, int, list[int]] | None":
    """Find the branching token in the reasoning output as well as the desired and undesired tokens.

    This function identifies the critical token in the non-greedy reasoning sequence
    where the model's behavior diverges from the greedy approach, leading to
    different answer correctness.

    Args:
        model: The language model to use for generation.
        non_greedy_generated_tokens: The non-greedy reasoning output tokens.
        question_len: The length of the question in the prompt.
        target: The target answer for comparison.
        max_new_tokens: The maximum number of new tokens to generate.
        greedy_answer_correct: Whether the greedy answer is correct or not.

    Returns:
        Tuple containing (next_new_token, next_old_token, prefix_tokens) if found, None otherwise.

    """
    ground_truth = extract_final_answer(target)

    # Starting from last question token to accommodate the case where it is the branching token,
    # i.e. model starts deviating from the greedy reasoning output at the first reasoning token itself.
    for token_index in range(question_len - 1, len(non_greedy_generated_tokens)):
        prefix_tokens = non_greedy_generated_tokens[: token_index + 1]
        curr_prefix_text = model.tokenizer.decode(prefix_tokens, skip_special_tokens=True)

        with torch.no_grad():
            with model.generate(
                curr_prefix_text,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            ):
                generated_output = model.generator.output.save()

        generated_tokens = generated_output[0]
        generated_text = model.tokenizer.decode(
            generated_tokens[question_len:], skip_special_tokens=True
        )
        generated_text = extract_until_first_answer(generated_text)
        final_answer = extract_final_answer(generated_text)

        current_answer_correct = final_answer == ground_truth
        if current_answer_correct != greedy_answer_correct:
            branching_token_idx = token_index - 1
            next_new_token = generated_tokens[branching_token_idx + 1].item()
            return next_new_token, next_old_token, prefix_tokens[:-1].tolist()

        # This will always run when token_index == question_len -1,
        # because of the previous greedy generation step.
        next_old_token = generated_tokens[token_index + 1].item()

    return None


def generate_localization_dataset_via_branching(
    model: LanguageModel,
    samples: Dataset,
    localization_dataset_path: str,
    greedy_dataset_path: str | None = None,
    non_greedy_dataset_path: str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    force_generation: bool = False,
    temperature_correct: float | None = 1.2,
    temperature_incorrect: float | None = 0.8,
) -> Dataset:
    """Generate a localization dataset for model updates.

    This function generates a dataset that contains the common prefix tokens between greedy and non-greedy reasoning outputs.
    It identifies the first differing token and the desired and undesired tokens for model updates.

    Args:
        model: The language model to use for generation.
        samples: Dataset of samples from the GSM-Symbolic dataset.
        localization_dataset_path: Path to save/load the localization dataset.
        greedy_dataset_path: Path to save/load the greedy reasoning dataset.
        non_greedy_dataset_path: Path to save/load the non-greedy reasoning dataset.
        batch_size: Batch size for generation.
        max_new_tokens: Maximum number of new tokens to generate.
        force_generation: If True, forces generation of the localization dataset even if it exists.

    Returns:
        Dataset: A dataset containing the localization data.

    """
    # Return dataset from disk if it exists and force_generation is False
    if not force_generation and localization_dataset_path is not None:
        logger.info(
            f"Localization dataset already exists. Loading from {localization_dataset_path}..."
        )
        localization_data: list[dict] = load_json(localization_dataset_path)
        return Dataset.from_list(localization_data)

    # Generate greedy reasoning dataset
    if greedy_dataset_path is None:
        greedy_traces = generate_greedy_reasoning_traces(
            model=model,
            samples=samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        with open(localization_dataset_path.replace(".json", "_greedy.json"), "w") as f:
            json.dump(greedy_traces, f, indent=4)
    else:
        greedy_traces: list[dict] = load_json(greedy_dataset_path)

    # Generate non-greedy reasoning dataset
    if non_greedy_dataset_path is None:
        non_greedy_traces = generate_non_greedy_reasoning_traces(
            model=model,
            greedy_traces=greedy_traces,
            max_new_tokens=max_new_tokens,
            temperature_correct=temperature_correct,
            temperature_incorrect=temperature_incorrect,
        )
        with open(localization_dataset_path.replace(".json", "_non_greedy.json"), "w") as f:
            json.dump(non_greedy_traces, f, indent=4)
    else:
        non_greedy_traces: list[dict] = load_json(non_greedy_dataset_path)

    # Generate localization dataset
    logger.info("Generating localization dataset...")
    localization_dataset = []
    for greedy_trace, non_greedy_trace in zip(greedy_traces, non_greedy_traces, strict=False):
        greedy_generated_tokens = greedy_trace["generated_tokens"]
        non_greedy_generated_tokens = non_greedy_trace["generated_tokens"]
        is_greedy_correct = greedy_trace["correct_answer"]
        is_non_greedy_correct = non_greedy_trace["correct_answer"]
        prompt_len = greedy_trace["prompt_len"]
        target = greedy_trace["target"]
        template_id = greedy_trace["template_id"]
        instance_id = greedy_trace["instance_id"]

        assert is_greedy_correct != is_non_greedy_correct or (is_non_greedy_correct is None), (
            f"Greedy and non-greedy traces must have different answer correctness, got greedy: {is_greedy_correct}, non-greedy: {is_non_greedy_correct}"
        )

        if greedy_generated_tokens is not None and non_greedy_generated_tokens is not None:
            try:
                non_greedy_tokens_tensor = torch.tensor(non_greedy_generated_tokens)
                branching_info = find_branching_info(
                    model=model,
                    non_greedy_generated_tokens=non_greedy_tokens_tensor,
                    question_len=prompt_len,
                    target=target,
                    max_new_tokens=max_new_tokens,
                    greedy_answer_correct=is_greedy_correct,
                )
            except Exception as e:
                logger.error(f"Error finding branching info: {e}")
                continue

            if branching_info is None:
                continue

            next_new_token, next_old_token, prefix_tokens = branching_info
            common_prefix = model.tokenizer.decode(
                prefix_tokens,
                skip_special_tokens=True,
            )
            if is_greedy_correct:
                desired_token = next_old_token
                undesired_token = next_new_token
            else:
                desired_token = next_new_token
                undesired_token = next_old_token

            localization_dataset.append(
                {
                    "prefix": common_prefix,
                    "desired_token": desired_token,
                    "undesired_token": undesired_token,
                    "template_id": template_id,
                    "instance_id": instance_id,
                }
            )

    logger.info(f"Generated {len(localization_dataset)} samples for localization dataset.")

    os.makedirs(os.path.dirname(localization_dataset_path), exist_ok=True)
    with open(localization_dataset_path, "w") as f:
        json.dump(localization_dataset, f, indent=4)

    logger.info(f"Localization dataset saved to {localization_dataset_path}")

    return Dataset.from_list(localization_dataset)


def get_first_differing_token(tokens_1: torch.Tensor, tokens_2: torch.Tensor) -> int:
    """Find the index where two token sequences diverge.

    Compares two tensors of tokens and returns the index of the first token
    that is different in the two tensors.

    Args:
        tokens_1 (torch.Tensor): First list of tokens
        tokens_2 (torch.Tensor): Second list of tokens

    Returns:
        int: Index of the first differing token.

    """
    assert tokens_1 is not None and tokens_2 is not None, (
        "Tokens cannot be None, got tokens_1: {tokens_1}, tokens_2: {tokens_2}"
    )

    for i in range(min(len(tokens_1), len(tokens_2))):
        if tokens_1[i] != tokens_2[i]:
            return i
    return min(len(tokens_1), len(tokens_2)) - 1


def generate_localization_dataset_via_prefix(
    model: LanguageModel,
    samples: Dataset,
    localization_dataset_path: str,
    greedy_dataset_path: str | None = None,
    non_greedy_dataset_path: str | None = None,
    max_new_tokens: int = 200,
    batch_size: int = 8,
    force_generation: bool = False,
    temperature_correct: float | None = 1.2,
    temperature_incorrect: float | None = 0.8,
) -> Dataset:
    """Generate a localization dataset for model updates via the common prefix method.

    This function generates a dataset that contains the common prefix tokens between greedy and non-greedy reasoning outputs.
    It identifies the first differing token and the desired and undesired tokens for model updates.

    Args:
        model: The language model to use for generation.
        samples: Dataset of samples from the GSM-Symbolic dataset.
        greedy_dataset_path: Path to save/load the greedy reasoning dataset.
        non_greedy_dataset_path: Path to save/load the non-greedy reasoning dataset.
        localization_dataset_path: Path to save/load the localization dataset.
        max_new_tokens: Maximum number of new tokens to generate.
        batch_size: Batch size for generation.
        force_generation: If True, forces generation of the localization dataset even if it exists.
        temperature_correct: Temperature for correct greedy generation.
        temperature_incorrect: Temperature for incorrect greedy generation.

    Returns:
        Dataset containing localization data.

    """
    # Return dataset from disk if it exists and force_generation is False
    if not force_generation and localization_dataset_path is not None:
        logger.info("Localization dataset already exists. Loading from disk...")
        localization_data: list[dict] = load_json(localization_dataset_path)
        return Dataset.from_list(localization_data)

    # Generate greedy reasoning dataset
    if greedy_dataset_path is None:
        greedy_traces = generate_greedy_reasoning_traces(
            model=model,
            samples=samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        with open(localization_dataset_path.replace(".json", "_greedy.json"), "w") as f:
            json.dump(greedy_traces, f, indent=4)
    else:
        logger.info(
            f"Greedy reasoning dataset already exists. Loading from {greedy_dataset_path}..."
        )
        greedy_traces: list[dict] = load_json(greedy_dataset_path)

    # Generate non-greedy reasoning dataset
    if non_greedy_dataset_path is None:
        non_greedy_traces = generate_non_greedy_reasoning_traces(
            model=model,
            greedy_traces=greedy_traces,
            max_new_tokens=max_new_tokens,
            temperature_correct=temperature_correct,
            temperature_incorrect=temperature_incorrect,
        )
        with open(localization_dataset_path.replace(".json", "_non_greedy.json"), "w") as f:
            json.dump(non_greedy_traces, f, indent=4)
    else:
        logger.info(
            f"Non-greedy reasoning dataset already exists. Loading from {non_greedy_dataset_path}..."
        )
        non_greedy_traces: list[dict] = load_json(non_greedy_dataset_path)

    # Generate localization dataset
    logger.info("Generating localization dataset...")
    localization_dataset = []
    for greedy_trace, non_greedy_trace in zip(greedy_traces, non_greedy_traces, strict=False):
        greedy_generated_tokens = greedy_trace["generated_tokens"]
        non_greedy_generated_tokens = non_greedy_trace["generated_tokens"]
        is_greedy_correct = greedy_trace["correct_answer"]
        is_non_greedy_correct = non_greedy_trace["correct_answer"]

        assert is_greedy_correct != is_non_greedy_correct, (
            "Greedy and non-greedy traces must have different answer correctness"
        )

        if greedy_generated_tokens is not None and non_greedy_generated_tokens is not None:
            try:
                greedy_tokens_tensor = torch.tensor(greedy_generated_tokens)
                non_greedy_tokens_tensor = torch.tensor(non_greedy_generated_tokens)
                first_differing_token = get_first_differing_token(
                    greedy_tokens_tensor, non_greedy_tokens_tensor
                )
            except Exception as e:
                logger.error(f"Error finding first differing token: {e}")
                continue

            if is_greedy_correct:
                desired_token = greedy_generated_tokens[first_differing_token]
                undesired_token = non_greedy_generated_tokens[first_differing_token]
            else:
                desired_token = non_greedy_generated_tokens[first_differing_token]
                undesired_token = greedy_generated_tokens[first_differing_token]

            common_prefix = model.tokenizer.decode(
                greedy_tokens_tensor[:first_differing_token],
                skip_special_tokens=True,
            )

            localization_dataset.append(
                {
                    "prefix": common_prefix,
                    "desired_token": desired_token,
                    "undesired_token": undesired_token,
                    "template_id": greedy_trace["template_id"],
                    "instance_id": greedy_trace["instance_id"],
                }
            )

    logger.info(f"Generated {len(localization_dataset)} samples for localization dataset.")

    os.makedirs(os.path.dirname(localization_dataset_path), exist_ok=True)
    with open(localization_dataset_path, "w") as f:
        json.dump(localization_dataset, f, indent=4)

    logger.info(f"Localization dataset saved to {localization_dataset_path}")

    return Dataset.from_list(localization_dataset)
