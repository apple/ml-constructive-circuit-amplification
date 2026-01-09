# Constructive Circuit Amplification: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates

This repository accompanies the research paper: [Constructive Circuit Amplification: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates](https://arxiv.org/abs/2512.16914)

## Abstract

Prior studies investigating the internal workings of LLMs have uncovered sparse subnetworks, often referred to as circuits, that are responsible for performing specific tasks. Additionally, it has been shown that model performance improvement through fine-tuning often results from the strengthening of existing circuits in the model. Taken together, these findings suggest the possibility of intervening directly on such circuits to make precise, task-targeted updates. Motivated by these findings, we propose a novel method called Constructive Circuit Amplification which identifies pivotal tokens from model reasoning traces as well as model components responsible for the desired task, and updates only those components. Applied to mathematical reasoning, it improves accuracy by up to +11.4% across multiple models while modifying as little as 1.59% of model components, with minimal impact on other abilities as measured by MMLU, TriviaQA, and TruthfulQA. These results demonstrate that targeted capabilities can be reliably enhanced by selectively updating a sparse set of model components.

## Setup

### Requirements

- Python >= 3.12
- CUDA-capable GPU (recommended for model training/evaluation)

### Installation

After cloning the repository 

```bash
# Install dependencies
pip install uv
uv add "nnsight==0.4.6"
uv sync
```

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required for model downloads
HF_TOKEN=your_huggingface_token_here

# Optional: For W&B logging
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=math-reasoning-cca
WANDB_BASE_URL=https://api.wandb.ai

# Optional: For OpenAI API (if using GPT models)
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### 1. Generating Localization Datasets

Generate training data that identifies critical tokens for mathematical reasoning:

```bash
python math_reasoning/generate_dataset.py \
    --config_file exps_configs/generate_data.yaml \
    --config_key by_template_id \
    --model_name google/gemma-2-9b-it \
    --generation_method prefix \
    --batch_size 16
```

This generates three files in the `artifacts/` directory:
- Greedy generated samples
- Non-greedy generated samples  
- Localization dataset with prefix tokens and target/undesired tokens

**Generation Methods:**
- `prefix`: Identifies first differing token between greedy and non-greedy outputs
- `branching`: Finds the branching point where reasoning diverges

### 2. Running the Circuit Tuning Pipeline

Update model weights using the localization dataset:

```bash
python math_reasoning/fixing_pipeline.py \
    --model_name google/gemma-2-2b-it \
    --localization_dataset_path artifacts/localization_dataset.json \
    --reg_coeff 0.001 \
    --update_coeff -0.0001 \
    --update_using_mask True \
    --n_steps 100 \
    --seed 42
```

**Key Parameters:**
- `--model_name`: HuggingFace model identifier
- `--localization_dataset_path`: Path to generated localization dataset
- `--reg_coeff`: Regularization coefficient for mask learning
- `--update_coeff`: Learning rate for model weight updates
- `--update_using_mask`: Whether to use learned masks (True) or update all weights (False)
- `--n_steps`: Number of gradient update steps

### 3. Finetuning Baselines

Compare against standard finetuning approaches:

```bash
python math_reasoning/finetune_gsm_symbolic.py \
    --model_name google/gemma-2-2b-it \
    --finetuning_method lora \
    --train_template_ids "all" \
    --pipeline_dataset_path artifacts/pipeline_dataset.json \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32
```

**Finetuning Methods:**
- `lora`: LoRA (Low-Rank Adaptation) finetuning
- `full`: Full model finetuning

### 4. Evaluation

Evaluate models on GSM-Symbolic:

```bash
python math_reasoning/eval_gsm_symbolic.py \
    --model_key best_gemma-2-2b-it_prefix_with_mask \
    --batch_size 16 \
    --max_new_tokens 1500 \
    --seed 42
```

For custom datasets:

```bash
python math_reasoning/eval_gsm_symbolic.py \
    --model_key your_model_key \
    --dataset_path path/to/your/dataset.json \
    --batch_size 16 \
    --max_new_tokens 1500 \
    --seed 42
```

### 5. LM-Evaluation-Harness Integration

Run standard benchmarks (MMLU, GSM8K, MATH, TriviaQA, TruthfulQA):

```bash
python math_reasoning/run_lm_eval.py \
    --models gemma-2-2b-it \
    --benchmarks "mmlu,math,triviaqa,gsm8k,truthfulqa" \
    --batch-size 4
```

Results are saved in `results/lm_eval/{model_key}_{benchmark}/`

## Project Structure

```
math-reasoning/
├── math_reasoning/          # Main package
│   ├── finetune_gsm_symbolic.py    # Finetuning script
│   ├── fixing_pipeline.py          # Circuit tuning pipeline
│   ├── generate_dataset.py         # Localization dataset generation
│   ├── eval_gsm_symbolic.py        # GSM-Symbolic evaluation
│   ├── run_lm_eval.py              # LM evaluation harness
│   ├── gsm_utils.py                # GSM dataset utilities
│   ├── pipeline_utils.py           # Pipeline helper functions
│   └── utils.py                    # General utilities
├── exps_configs/            # Experiment configurations
├── scripts/                 # Helper scripts
└── README.md
```

## Model Configuration

Models are configured in JSON files under `math_reasoning/models/`:
- `lm_eval_models.json`: Models for LM evaluation
- `reasoning_models.json`: Models for reasoning tasks
- `non_reasoning_models.json`: General language models

Add your own models by following the existing format.

## Experiment Configurations

Sample configurations are provided in `exps_configs/`:
- `gemma-2-2b_prefix_sample.json`: Gemma 2B with prefix method
- `finetune_gemma-2-2b-it_lora.json`: LoRA finetuning for Gemma 2B
- `generate_data.yaml`: Dataset generation configuration

Copy and modify these for your experiments.

## Hyperparameter Sweeps

For parameter sweeps, use the controller scripts as templates:

```bash
python math_reasoning/run_finetuning_pipeline_controller.py \
    --base_config_path exps_configs/finetune_sample.json \
    --learning_rates "5e-5,1e-4,2e-4" \
    --lora_rs "8,16,32" \
    --seeds "42,43,44"
```

Note: Controller scripts output configuration combinations but don't execute them automatically. Adapt them for your local job orchestration system (GNU Parallel, Ray, etc.).

## Cite

```
@misc{prakash2025constructivecircuitamplificationimproving,
      title={Constructive Circuit Amplification: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates}, 
      author={Nikhil Prakash and Donghao Ren and Dominik Moritz and Yannick Assogba},
      year={2025},
      eprint={2512.16914},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.16914}, 
}
```