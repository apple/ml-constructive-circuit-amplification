# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import os
import sys
from collections import defaultdict

import torch
from nnsight import LanguageModel
from tqdm import tqdm

sys.path.append("..")
from math_reasoning.gsm_utils import extract_final_answer, logger

OUTPUT_DIR = os.environ.get("ARTIFACT_DIR", "")
logger.info(f"OUTPUT DIR: {OUTPUT_DIR}")

model = LanguageModel(
    "google/gemma-2-9b-it", device_map="auto", torch_dtype=torch.float16, dispatch=True
)
print("Model loaded successfully.")

clean_prompt = """As an expert problem solver, solve step by step the following mathematical questions.
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8.
Q: There are 124 second-graders at Pine Valley School. 74 of them are girls. On Tuesday, 3 second-grade girls and 7 second-grade boys were absent. How many second grade boys were at Pine Valley School on Tuesday?
A: Let's think step by step."""

corrupt_prompt = """As an expert problem solver, solve step by step the following mathematical questions.
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8.
Q: There are 124 second-graders at Pine Valley School. 74 of them are girls. On Tuesday, 3 second-grade girls and 2 second-grade boys were absent. How many second grade boys were at Pine Valley School on Tuesday?
A: Let's think step by step."""

corrupt_tokens = model.tokenizer.encode(corrupt_prompt, return_tensors="pt").to(model.device)
clean_tokens = model.tokenizer.encode(clean_prompt, return_tensors="pt").to(model.device)
# patching_token = 843
token_2 = model.tokenizer.encode("2")[1]
patching_token = (corrupt_tokens == token_2).nonzero(as_tuple=True)[-1][-1].item()
prompt_length = corrupt_tokens.ne(model.tokenizer.pad_token_id).sum().item()

patching_scores_clean, patching_scores_corrupt = defaultdict(dict), defaultdict(dict)

for token_idx in range(patching_token, prompt_length, 1):
    for layer_idx in tqdm(range(0, model.config.num_hidden_layers, 2)):
        correct_clean, correct_corrupt, total = 0, 0, 0
        with torch.no_grad():
            with model.trace(corrupt_prompt):
                cache = model.model.layers[layer_idx].output[0][0, token_idx].clone().save()

            with model.generate(
                clean_prompt,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            ):
                model.model.layers[layer_idx].output[0][0, token_idx] = cache
                out = model.generator.output.save()

            output_text = model.tokenizer.decode(out[0], skip_special_tokens=True)
            pred = extract_final_answer(output_text)

            if int(pred) == 43:
                correct_clean += 1
            if int(pred) == 48:
                correct_corrupt += 1
            total += 1

        patching_scores_clean[layer_idx][token_idx] = correct_clean / total
        patching_scores_corrupt[layer_idx][token_idx] = correct_corrupt / total
        logger.info(
            f"Layer {layer_idx}, Token idx {token_idx}, Token '{model.tokenizer.decode(corrupt_tokens[0, token_idx], skip_special_tokens=True)}', Accuracy: {correct_clean / total:.2f}"
        )

        with open(f"{OUTPUT_DIR}/patching_scores_clean.json", "w") as f:
            json.dump(patching_scores_clean, f, indent=4)

        with open(f"{OUTPUT_DIR}/patching_scores_corrupt.json", "w") as f:
            json.dump(patching_scores_corrupt, f, indent=4)

logger.info("Patching scores saved successfully.")
