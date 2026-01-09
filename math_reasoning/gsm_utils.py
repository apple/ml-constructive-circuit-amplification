# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import os
import random
import re  # noqa: D100
from collections import defaultdict
from typing import Any

import torch
from datasets import Dataset, load_dataset
from nnsight import LanguageModel
from tqdm import tqdm
from utils import load_json, logger


def to_list(prompts) -> list:
    """Convert prompts to list if needed."""
    if hasattr(prompts, "to_list"):
        prompts = prompts.to_list()
    if hasattr(prompts, "tolist"):
        prompts = prompts.tolist()
    if not isinstance(prompts, list):
        prompts = list(prompts)
    return prompts


def extract_until_first_answer(text: str) -> str:
    """Extract text from the beginning until the first occurrence of "answer is..." sentence (including that sentence).

    Args:
        text (str): Input text containing multiple problems and solutions

    Returns:
        str: Text from beginning up to and including the first final answer sentence

    """
    pattern = r"answer is[^.]*\."
    match = re.search(pattern, text)
    if match:
        end_position = match.end()
        return text[:end_position].strip()
    else:
        return text


@torch.no_grad()
def eval_model(
    model: LanguageModel,
    tokenizer: Any,
    samples: Dataset,
    max_new_tokens: int,
    batch_size: int,
    model_output_path: str | None = None,
) -> dict[int, dict[str, int | float]]:
    """Evaluate the model on the GSM-Symbolic dataset using batch generation.

    Args:
        model (LanguageModel): The language model to evaluate.
        tokenizer (Any): The tokenizer to use for decoding the model outputs.
        samples (Dataset): The dataset of samples from the GSM-Symbolic dataset.
        max_new_tokens (int): The maximum number of new tokens to generate.
        batch_size (int): The number of samples to process in each batch.
        model_output_path (str): The path to save the model outputs.

    Returns:
        dict[int, float]: The accuracy of the model for each template ID.

    """
    template_ids = list(set([sample["template_id"] for sample in samples]))
    template_correct = {template_id: 0 for template_id in template_ids}
    template_total = {template_id: 0 for template_id in template_ids}

    total_batches = (len(samples) + batch_size - 1) // batch_size
    tqdm_bar = tqdm(
        range(total_batches),
        total=total_batches,
        desc="Evaluating",
        smoothing=0.01,
    )

    for batch_idx in tqdm_bar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(samples))

        batch_samples = samples.select(range(start_idx, end_idx))
        batch_prompts = batch_samples["prompt"]
        batch_template_ids = batch_samples["template_id"]
        batch_targets = batch_samples["target"]

        batch_prompts = to_list(batch_prompts)

        inputs = tokenizer(
            batch_prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        ):
            outputs = model.generator.output.save()

        batch_outputs = []
        for i in range(len(batch_samples)):
            # With left padding, we can extract generated tokens from the full sequence length
            # The generated tokens start right after the original padded input tokens
            input_length = inputs["input_ids"][i].shape[0]
            generated_tokens = outputs[i][input_length:]
            output_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            output_text = extract_until_first_answer(output_text)
            batch_outputs.append(output_text)

        for i, output_text in enumerate(batch_outputs):
            template_id = batch_template_ids[i]
            instance_id = batch_samples[i]["instance_id"]
            target = batch_targets[i]

            final_answer = extract_final_answer(output_text)
            ground_truth = extract_final_answer(target)

            if model_output_path is not None:
                data = {
                    "template_id": template_id,
                    "instance_id": instance_id,
                    "output": output_text,
                    "target": target,
                    "final_answer": final_answer,
                    "ground_truth": ground_truth,
                }
                with open(model_output_path, "a") as f:
                    json.dump(data, f, indent=4)

            if final_answer == ground_truth:
                template_correct[template_id] += 1
            template_total[template_id] += 1
            tqdm_bar.set_postfix(
                {
                    "Template ID": template_id,
                    "Correct": template_correct[template_id],
                    "Total": template_total[template_id],
                    "Accuracy": template_correct[template_id] / template_total[template_id],
                }
            )

        del outputs
        torch.cuda.empty_cache()

    template_accs = {
        template_id: {
            "correct": template_correct[template_id],
            "total": template_total[template_id],
            "accuracy": template_correct[template_id] / template_total[template_id],
        }
        for template_id in template_ids
    }

    return template_accs


def filter_short_answers(example, num_sentences: int = 3):
    """Count reasoning steps and filter samples with more than specified threshold.

    Process the answer text to identify and count reasoning steps.
    Returns True for examples with fewer than num_sentences reasoning steps.

    Args:
        example: Dictionary containing an 'answer' field to analyze
        num_sentences (int): The maximum number of sentences allowed (default: 3)

    Returns:
        bool: True if the answer has exactly num_sentences reasoning steps

    """
    answer = example["answer"]
    answer = answer.split("####")[0]
    sentences = answer.split("\n")
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences)

    return sentence_count == num_sentences


def extract_final_answer(model_resp: str) -> float | None:
    """Extract the final numerical answer from a model response.

    Remove commas from the response, find the last number in the text,
    and convert it to a float value.

    Args:
        model_resp: The text response from the model containing numerical answers

    Returns:
        float: The extracted final numerical answer

    Raises:
        IndexError: If no numbers are found in the response
        ValueError: If the extracted text cannot be converted to a float

    """
    model_resp = model_resp.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", model_resp)
    if len(numbers) == 0:
        return None
    extracted_num = numbers[-1]
    return float(extracted_num)


def get_device():
    """Returns the torch device to use."""
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def filter_by_template_idx(ds: Dataset, cot_prompt: str, template_idx: list[int]) -> Dataset:
    """Filter the dataset by template index and return samples.

    Args:
        ds (Dataset): The dataset to filter.
        cot_prompt (str): The prompt template to use.
        template_idx (list[int]): The indices of the templates to filter by.

    Returns:
        Dataset: A dataset containing the question, target answer, prompt, and template_id
            for each sample.

    """
    samples = []

    for tid in template_idx:
        # Filter dataset for current template
        template_ds = ds.filter(lambda x: x["original_id"] == tid)

        # Process samples for this template
        for sample in template_ds:
            template_id = sample["original_id"]
            instance_id = sample["instance"]
            question = sample["question"]
            target = sample["answer"]
            prompt = cot_prompt.replace("<TARGET_QUESTION>", question)
            samples.append(
                {
                    "question": question,
                    "target": target,
                    "prompt": prompt,
                    "template_id": template_id,
                    "instance_id": instance_id,
                }
            )

    return Dataset.from_list(samples)


def add_few_shot_examples_and_format(ds: list[dict], cot_prompt: str) -> Dataset:
    """Add few shot examples to dataset.

    Args:
        ds (Dataset): The dataset to augment.
        cot_prompt (str): The prompt template to use.

    """
    # Process samples for this template
    results = []
    for sample in ds:
        template_id = sample["original_id"]
        instance_id = sample["instance"]
        question = sample["question"]
        target = sample["answer"]
        prompt = cot_prompt.replace("<TARGET_QUESTION>", question)
        results.append(
            {
                "question": question,
                "target": target,
                "prompt": prompt,
                "template_id": template_id,
                "instance_id": instance_id,
            }
        )

    return results


def load_gsm_symbolic(num_reasoning_clauses: int | None = None) -> tuple[Dataset, str]:
    """Load the GSM-Symbolic dataset and filters it based on number of reasoning steps and also prepares a COT template.

    This function loads the GSM-Symbolic dataset, filters out samples with
    short answers, and prepares a prompt template with examples from the dataset.

    Args:
        num_reasoning_clauses (int): The maximum number of reasoning clauses allowed in answers (default: None)

    Returns:
        tuple: A tuple containing (filtered_dataset, cot_prompt)

    """
    ds = load_dataset("apple/GSM-Symbolic", name="main", split="test")
    if num_reasoning_clauses is not None:
        filtered_ds = ds.filter(lambda x: filter_short_answers(x, num_reasoning_clauses))
        logger.info(
            f"Found {len(set(filtered_ds['original_id']))} templates with {num_reasoning_clauses} sentences."
        )
    else:
        filtered_ds = ds

    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "templates", "cot_prompt_template.txt")
    if os.path.exists(template_path):
        with open(template_path) as f:
            cot_prompt = f.read()
    else:
        logger.error(f"Template file not found: {template_path}")
        raise FileNotFoundError(f"Template file not found: {template_path}")

    gsm8k_cot_path = os.path.join(current_dir, "templates", "gsm8k-cot.json")
    if os.path.exists(gsm8k_cot_path):
        with open(gsm8k_cot_path) as f:
            gsmcot = json.load(f)
    else:
        logger.error(f"GSM8K-COT file not found: {gsm8k_cot_path}")
        raise FileNotFoundError(f"GSM8K-COT file not found: {gsm8k_cot_path}")

    # Replace placeholders in the cot_prompt with actual examples from gsmcot
    for i, shot in enumerate(gsmcot):
        question = shot["question"]
        target = shot["target"]
        curr_ques = f"<SHOT_{i + 1}_QUESTION>"
        curr_ans = f"<SHOT_{i + 1}_ANSWER>"
        cot_prompt = cot_prompt.replace(curr_ques, question)
        cot_prompt = cot_prompt.replace(curr_ans, target)

    return filtered_ds, cot_prompt


def add_instance_id(localization_dataset: list[dict], gsm_symbolic: list[dict], instance_key):
    for item in localization_dataset:
        # search for a match between prefix and question
        prefix = item["prefix"]
        found = False
        for gsm_sample in gsm_symbolic:
            # instead of instance key we could use "instance" by checking
            # that gsm_sample[original_id] matches localization[template_id]
            # but that would have to be duplicatted in multiple places.
            question = gsm_sample["question"]
            unique_id = gsm_sample[instance_key]
            if question in prefix:
                item[instance_key] = unique_id
                found = True
                break
        if not found:
            raise ValueError(f"Could not find instance id for {item}")

    return localization_dataset


def load_pipeline_dataset(
    train_template_ids: list[int] | str,
    split_percentages: dict,
    output_dir: str,
    seed: int,
    localization_data_path: str | None = None,
    sampled_templates_fraction: float = 1.0,
    num_reasoning_clauses: int | None = None,
    pipeline_dataset_path: str | None = None,
) -> dict[str, Dataset]:
    """Load the GSM symbolic dataset and split it for the training and evaluation pipeline.

    Loads the dataset from the specified path, applies the given splits, and saves
    the resulting dataset with split information to the output directory.

    Args:
        train_template_ids: List of template IDs to use for training data.
        split_percentages: Dictionary specifying split ratios, e.g.,
            {"train": 0.5, "test": 0.15, "val": 0.35}
        output_dir: Directory where the processed dataset will be saved.
        seed: Random seed for reproducible dataset splitting.
        localization_data_path: Path to the localization dataset file.
        sampled_templates_fraction: Fraction of templates to sample for training.
        num_reasoning_clauses: Optional limit on the number of reasoning clauses
            to include. If None, all clauses are included.
        pipeline_dataset_path: Path to the dataset that was used in the pipeline.

    Returns:
        The processed dataset with train/test/eval splits applied.

    """
    # check that the values in split_percentages sum to 1
    assert sum(split_percentages.values()) == 1.0

    if sampled_templates_fraction < 1:
        assert sampled_templates_fraction > 0, "Sampled templates fraction must be greater than 0"

    gsm_symbolic_ds, cot_prompt = load_gsm_symbolic(num_reasoning_clauses=num_reasoning_clauses)

    gsm_symbolic_samples: list[dict] = list(gsm_symbolic_ds)
    gsm_symbolic_samples = add_few_shot_examples_and_format(gsm_symbolic_samples, cot_prompt)

    # add a unique id to each sample
    instance_key = "unique_id"
    for sample in gsm_symbolic_samples:
        sample[instance_key] = f"{sample['template_id']}--{sample['instance_id']}"

    if pipeline_dataset_path is not None:
        if pipeline_dataset_path.startswith(("http://", "https://")):
            raise ValueError(f"Remote URLs not supported: {pipeline_dataset_path}")
        
        with open(pipeline_dataset_path) as f:
            content = f.read().strip()
            pipeline_dataset = []
            for line in content.split("\n"):
                pipeline_dataset.append(json.loads(line))

        # extract unique ids of instances in each split of the pipeline dataset
        train_instances_id = [
            instance["unique_id"] for instance in pipeline_dataset if instance["split"] == "train"
        ]
        val_instances_id = [
            instance["unique_id"] for instance in pipeline_dataset if instance["split"] == "val"
        ]
        test_instances_id = [
            instance["unique_id"] for instance in pipeline_dataset if instance["split"] == "test"
        ]

        # filter gsm-symbolic dataset based on the pipeline dataset
        train_instances = [
            instance
            for instance in gsm_symbolic_samples
            if instance["unique_id"] in train_instances_id
        ]
        val_instances = [
            instance
            for instance in gsm_symbolic_samples
            if instance["unique_id"] in val_instances_id
        ]
        test_instances = [
            instance
            for instance in gsm_symbolic_samples
            if instance["unique_id"] in test_instances_id
        ]

        return {
            "train": Dataset.from_list(train_instances),
            "val": Dataset.from_list(val_instances),
            "test": Dataset.from_list(test_instances),
        }

    # Group gsm_symbolic_samples by original_id
    gsm_symbolic_samples_by_original_id: dict[int, list[dict]] = defaultdict(list)
    for sample in gsm_symbolic_samples:
        original_id = sample["template_id"]
        gsm_symbolic_samples_by_original_id[original_id].append(sample)

    # Split samples in each group.
    rand = random.Random(seed)
    gsm_symbolic_by_split = defaultdict(list)
    for template_id, template_samples in gsm_symbolic_samples_by_original_id.items():
        # split samples randomly into train, val and test sets
        rand.shuffle(template_samples)
        num_train = int(len(template_samples) * split_percentages["train"])
        num_val = int(len(template_samples) * split_percentages["val"])
        num_test = len(template_samples) - num_train - num_val

        assert num_train + num_val + num_test == len(template_samples)

        train = template_samples[:num_train]
        val = template_samples[num_train : num_train + num_val]
        test = template_samples[num_train + num_val :]

        assert (len(train) + len(val) + len(test)) == len(template_samples)

        # mark each sample with the split it is in
        for sample in train:
            sample["split"] = "train"
            gsm_symbolic_by_split["train"].append(sample)
        for sample in val:
            sample["split"] = "val"
            gsm_symbolic_by_split["val"].append(sample)
        for sample in test:
            sample["split"] = "test"
            gsm_symbolic_by_split["test"].append(sample)

    # check that every template in the train, val and test splits have
    # the same number of samples (before filtering)
    def check_split_size_consistency(split):
        template_samples_by_id: [str, list] = defaultdict(list)
        split_samples = gsm_symbolic_by_split[split]
        for sample in split_samples:
            template_id = sample["template_id"]
            template_samples_by_id[template_id].append(sample)

        ids = list(template_samples_by_id.keys())
        assert len(ids) == 100
        samples_for_template_1 = template_samples_by_id[ids[0]]
        for template_id, template_samples_for_split in template_samples_by_id.items():
            assert len(template_samples_for_split) == len(samples_for_template_1)

    check_split_size_consistency("test")
    check_split_size_consistency("val")
    check_split_size_consistency("train")

    # Set up train_template_ids
    if train_template_ids is None or train_template_ids == "all":
        all_ids = set([d["template_id"] for d in gsm_symbolic_samples])
        train_template_ids = list(all_ids)

    assert isinstance(train_template_ids, list)
    if sampled_templates_fraction < 1.0:
        train_template_ids = rand.sample(
            train_template_ids, int(len(train_template_ids) * sampled_templates_fraction)
        )

    if localization_data_path is None:
        # filter
        train_split = gsm_symbolic_by_split["train"]
        train_split_filtered = [s for s in train_split if s["template_id"] in train_template_ids]

        train_dataset = Dataset.from_list(train_split_filtered)
        val_dataset = Dataset.from_list(gsm_symbolic_by_split["val"])
        test_dataset = Dataset.from_list(gsm_symbolic_by_split["test"])

        # # Make one dataset from all of these and save it to output_path
        full_dataset = Dataset.from_list(
            train_split_filtered + gsm_symbolic_by_split["val"] + gsm_symbolic_by_split["test"]
        )
        full_dataset.to_json(os.path.join(output_dir, "pipeline_dataset.json"))

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }



    #
    # Now filter the train split based on the localization data
    # each item in localization_data has, prefix, template_id and optionally, instance
    #
    localization_data: list[dict] = load_json(localization_data_path)
    # first filter the localization data to only include templates in train_template ids
    localization_data = [
        loc for loc in localization_data if loc["template_id"] in train_template_ids
    ]
    localization_data = add_instance_id(localization_data, gsm_symbolic_samples, instance_key)

    localization_data_by_instance = {}
    for localization in localization_data:
        localization_data_by_instance[localization[instance_key]] = localization

    # then filter train split to only include samples in localization data
    filtered_train = []
    for sample in gsm_symbolic_by_split["train"]:
        if sample[instance_key] in localization_data_by_instance:
            # add localization data to train dataset
            sample_id = sample[instance_key]
            sample.update(localization_data_by_instance[sample_id])
            filtered_train.append(sample)

    assert len(filtered_train) <= len(localization_data), (
        f"final train set is larger than localization. f:{len(filtered_train)}, l:{len(localization_data)}"
    )

    # convert train, test and val datasets to Datasets
    train_dataset = Dataset.from_list(filtered_train)
    val_dataset = Dataset.from_list(gsm_symbolic_by_split["val"])
    test_dataset = Dataset.from_list(gsm_symbolic_by_split["test"])

    # Make one dataset from all of these and save it to output_path
    full_dataset = Dataset.from_list(
        filtered_train + gsm_symbolic_by_split["val"] + gsm_symbolic_by_split["test"]
    )
    full_dataset.to_json(os.path.join(output_dir, "pipeline_dataset.json"))

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_target_modules_for_model(model):
    """Automatically determine target modules for LoRA based on model architecture.

    Args:
        model: The loaded model

    Returns:
        List of target module names for LoRA

    """
    model_name = model.config.model_type.lower()

    # Common target modules for different architectures
    if "llama" in model_name or "gemma" in model_name or "olmo" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "gpt" in model_name or "gpt2" in model_name:
        return ["c_attn", "c_proj", "c_fc"]
    elif "phi" in model_name:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "mpt" in model_name:
        return ["Wqkv", "out_proj", "up_proj", "down_proj"]
    elif "falcon" in model_name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "bloom" in model_name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "opt" in model_name:
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif "mistral" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "qwen" in model_name:
        return ["c_attn", "c_proj", "w1", "w2"]
    elif "yi" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        # Fallback: try to detect common patterns in the model
        logger.warning(
            f"Unknown model architecture: {model_name}. Attempting to auto-detect target modules."
        )

        target_modules = []
        for name, module in model.named_modules():
            if any(
                pattern in name.lower()
                for pattern in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ):
                target_modules.append(
                    name.split(".")[-1]
                )  # Get just the module name, not full path
            elif any(
                pattern in name.lower()
                for pattern in ["c_attn", "c_proj", "c_fc", "dense", "fc1", "fc2"]
            ):
                target_modules.append(name.split(".")[-1])

        if target_modules:
            # Remove duplicates while preserving order
            seen = set()
            unique_modules = []
            for module in target_modules:
                if module not in seen:
                    seen.add(module)
                    unique_modules.append(module)
            return unique_modules
        else:
            # Final fallback to common modules
            logger.warning("Could not auto-detect target modules. Using default fallback.")
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
