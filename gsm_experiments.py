import json
import logging
import random
import re
import traceback
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from fixedTreeSD import fixed_tree_speculative_generate_greedy, run_normal_baseline

_NUMBER = r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"

set_seed(42)

def build_prompt(question):
    return f"Q: {question}\n A: Let's think step by step."

def extract_gold_answer(answer):
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    matches = re.findall(_NUMBER, answer)
    return matches[-1].strip() if matches else None

def extract_prediction(text, flexible=True):
    explicit_patterns = [
        rf"####\s*({_NUMBER})",
        rf"\\boxed\{{\s*({_NUMBER})\s*\}}",
        rf"(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*({_NUMBER})",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

    if flexible:
        matches = re.findall(_NUMBER, text)
        if matches:
            return matches[-1].strip()
    return None

def canonical_number(value):
    if value is None:
        return None
    cleaned = value.strip().replace("$", "").replace(",", "").replace(" ", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    cleaned = cleaned.rstrip(".")
    try:
        result = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    return result / Decimal(100) if is_percent else result

def answers_equal(prediction, gold):
    pred_num = canonical_number(prediction)
    gold_num = canonical_number(gold)
    return pred_num is not None and gold_num is not None and pred_num == gold_num

def setup_logging(output_dir):
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    
    run_handler = logging.FileHandler(logs_dir / "run.log", mode="a")
    run_handler.setFormatter(formatter)
    root.addHandler(run_handler)
    
    error_handler = logging.FileHandler(logs_dir / "errors.log", mode="a")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root.addHandler(error_handler)
    
    return root

def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "benchmark_results"
    
    logger = setup_logging(output_dir)
    logger.info("Starting simplified benchmark script.")

    DRAFT_MODEL = "meta-llama/Llama-3.2-1B"
    TARGET_MODEL = "meta-llama/Llama-3.2-3B"
    DATASET_NAME = "openai/gsm8k"
    DATASET_CONFIG = "main"
    SPLIT = "test"
    LIMIT = 20
    MAX_NEW_TOKENS = 256
    BUDGET = 16
    MAX_K = 16
    MU = 0.03
    DEVICE = "cuda"
    DTYPE = torch.float16

    try:
        logger.info(f"Loading Tokenizer, Draft ({DRAFT_MODEL}), and Target ({TARGET_MODEL})...")
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()

        logger.info(f"Loading dataset {DATASET_NAME}...")
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
        
        indices = random.sample(range(len(dataset)), min(LIMIT, len(dataset)))
        results_file = output_dir / "results.jsonl"

        for idx, ds_idx in enumerate(indices):
            item = dataset[ds_idx]
            prompt = build_prompt(item['question'])
            gold_answer = extract_gold_answer(item['answer'])
            
            logger.info(f"Processing sample {idx+1}/{len(indices)}")
            
            try:
                # Baseline Run
                torch.cuda.reset_peak_memory_stats()
                base_latency, base_tokens, raw_base_text_list, base_output_ids = run_normal_baseline(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    target_model=target_model,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device=DEVICE,
                )

                prompt_len = tokenizer.encode(prompt, return_tensors="pt").shape[1]
                base_new_token_ids = base_output_ids[0, prompt_len:].tolist()
                base_new_text = tokenizer.decode(base_new_token_ids, skip_special_tokens=True)

                base_extracted = extract_prediction(base_new_text, flexible=True)
                base_correct = answers_equal(base_extracted, gold_answer)
                
                # TALON Run
                torch.cuda.reset_peak_memory_stats()
                talon_res, talon_metrics, _ = fixed_tree_speculative_generate_greedy(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    ssm=draft_model,
                    target_model=target_model,
                    budget=BUDGET,
                    mu=MU,
                    max_k=MAX_K,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device=DEVICE,
                    dtype=DTYPE,
                    debug=False,
                )
                
                # Extract the newly generated string directly from the dictionary
                talon_new_text = talon_res["new_text"]
                talon_extracted = extract_prediction(talon_new_text, flexible=True)
                talon_correct = answers_equal(talon_extracted, gold_answer)
                
                # Metrics
                talon_latency = float(talon_res["latency"])
                speedup = base_latency / talon_latency if talon_latency > 0 else 0
                
                # We compare only the newly generated text to ensure strict parity
                text_match = (base_new_text == talon_new_text)
                
                record = {
                    "sample_id": ds_idx,
                    "prompt": prompt,
                    "reference_solution": item['answer'],
                    "gold_answer": gold_answer,
                    "baseline_new_text": base_new_text,
                    "talon_new_text": talon_new_text,
                    "baseline_extracted_answer": base_extracted,
                    "talon_extracted_answer": talon_extracted,
                    "baseline_correct": base_correct,
                    "talon_correct": talon_correct,
                    "baseline_latency_s": float(base_latency),
                    "talon_latency_s": talon_latency,
                    "speedup": float(speedup),
                    "text_exact_match": text_match,
                    "talon_metrics": talon_metrics
                }
                
                with open(results_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    
                logger.info(f"Sample {idx+1} | Speedup: {speedup:.2f}x | Parity: {text_match} | Base Correct: {base_correct} | TALON Correct: {talon_correct}")
                
            except Exception as e:
                logger.error(f"Failed on sample {ds_idx}: {str(e)}\n{traceback.format_exc()}")
                continue

        logger.info("Benchmark finished successfully.")
        
    except Exception as e:
        logger.error(f"Fatal benchmark error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()