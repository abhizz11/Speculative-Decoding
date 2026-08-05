import json
import logging
import random
import re
import traceback
from collections import defaultdict
from statistics import mean, median
from decimal import Decimal, InvalidOperation
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Import the three respective methods
from speculative_decoding import speculative_decoding_inference
from fixed_tree_sd import fixed_tree_speculative_generate_greedy
from talon_imp import dynamic_tree_speculative_generate_greedy

_NUMBER = r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"

set_seed(40)

def build_prompt(instruction):
    return instruction.rstrip() + "\n"

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
    
    run_handler = logging.FileHandler(logs_dir / "run_combined_3.log", mode="a")
    run_handler.setFormatter(formatter)
    root.addHandler(run_handler)
    
    error_handler = logging.FileHandler(logs_dir / "errors_combined_3.log", mode="a")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root.addHandler(error_handler)
    
    return root

def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "benchmark_results"
    
    logger = setup_logging(output_dir)
    logger.info("Starting triple-method speculative decoding benchmark script.")

    DRAFT_MODEL = "meta-llama/Llama-3.2-1B"
    TARGET_MODEL = "meta-llama/Llama-3.2-3B"
    DATASET_PATH = root_dir / "combined_speculative_decoding_dataset.jsonl"
    TURN_INDEX = 0  
    CATEGORIES = None  
    MAX_NEW_TOKENS = 256
    BUDGET = 15
    MAX_K = 16
    MU = 0.03
    FIXED_K_CONFIG = [3, 2, 1] # Config for Fixed Tree
    SD_GAMMA = 4 # Lookahead for Standard Speculative Decoding
    DEVICE = "cuda"
    DTYPE = torch.float32
    
    total_question_time = 0.0
    successful_questions = 0
    failed_questions = 0

    try:
        logger.info(f"Loading Tokenizer, Draft ({DRAFT_MODEL}), and Target ({TARGET_MODEL})...")
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()

        logger.info(f"Loading JSONL dataset from {DATASET_PATH}...")
        with DATASET_PATH.open("r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        results_file = output_dir / "results_combine_3_methods.jsonl"

        seen_ids = set()
        unique_indices = []

        for ds_idx, item in enumerate(dataset[:10]):
            sample_id = item["id"]
            category = item["category"]

            if CATEGORIES is not None and category not in CATEGORIES:
                continue

            if sample_id not in seen_ids:
                seen_ids.add(sample_id)
                unique_indices.append(ds_idx)

        completed_ids = set()
        if results_file.exists():
            with results_file.open("r") as f:
                for line_number, line in enumerate(f, start=1):
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        logger.warning(f"Ignoring invalid results line {line_number}")

        indices = [ds_idx for ds_idx in unique_indices if dataset[ds_idx]["id"] not in completed_ids]
        random.Random(40).shuffle(indices)

        logger.info(f"Dataset rows: {len(dataset)}")
        logger.info(f"Unique selected samples: {len(unique_indices)}")
        logger.info(f"Previously completed: {len(completed_ids)}")
        logger.info(f"Questions remaining: {len(indices)}")

        for idx, ds_idx in enumerate(indices):
            question_start = time.perf_counter()
            question_processing_time = None

            try:
                item = dataset[ds_idx]
                sample_id = item["id"]
                category = item["category"]
                prompt = build_prompt(item["turns"][TURN_INDEX])

                logger.info(f"Processing sample {idx + 1}/{len(indices)} (id={sample_id}, category={category})")

                # 1. Standard speculative decoding
                torch.cuda.reset_peak_memory_stats()
                
                # Pre-calculate prompt length to strip it from the final text later
                prompt_len = tokenizer.encode(prompt, return_tensors="pt").shape[1]
                
                sd_start = time.perf_counter()
                sd_final_text, sd_accepted, sd_total_draft, sd_acc_rate, sd_gen_count, transfer_kb = speculative_decoding_inference(
                    large_model=target_model,
                    small_model=draft_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=DEVICE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    gamma=SD_GAMMA
                )
                sd_latency = time.perf_counter() - sd_start
                
                # sd_final_text includes the prompt, isolate new text to match tree variants
                sd_tokens = tokenizer.encode(sd_final_text)
                sd_new_text = tokenizer.decode(sd_tokens[prompt_len:], skip_special_tokens=True)

                # 2. Fixed Tree Speculative Decoding
                torch.cuda.reset_peak_memory_stats()

                fixed_res, fixed_metrics = fixed_tree_speculative_generate_greedy(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    ssm=draft_model,
                    target_model=target_model,
                    k_config=FIXED_K_CONFIG,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device=DEVICE,
                    dtype=DTYPE,
                    debug=False,
                )
                fixed_latency = float(fixed_res["latency"])
                fixed_new_text = fixed_res["new_text"]
                fixed_speedup = (sd_latency / fixed_latency) if fixed_latency > 0 else 0.0

                # 3. TALON (Dynamic Tree)
                torch.cuda.reset_peak_memory_stats()

                talon_res, talon_metrics, _ = dynamic_tree_speculative_generate_greedy(
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
                talon_latency = float(talon_res["latency"])
                talon_new_text = talon_res["new_text"]
                talon_speedup = (sd_latency / talon_latency) if talon_latency > 0 else 0.0

                question_processing_time = time.perf_counter() - question_start

                record = {
                    "sample_id": sample_id,
                    "dataset_index": ds_idx,
                    "dataset_category": category,
                    "turn_index": TURN_INDEX,
                    "prompt": prompt,
                    "reference": item.get("reference"),
                    "sd_new_text": sd_new_text,
                    "fixed_new_text": fixed_new_text,
                    "talon_new_text": talon_new_text,
                    "question_processing_time_s": float(question_processing_time),
                    "sd_latency_s": sd_latency,
                    "fixed_latency_s": fixed_latency,
                    "talon_latency_s": talon_latency,
                    "fixed_speedup_vs_sd": fixed_speedup,
                    "talon_speedup_vs_sd": talon_speedup,
                    "fixed total transfer kb": fixed_metrics["total_transfer_kb"],
                    "talon total transfer kb": talon_metrics["total_transfer_kb"],
                    "sd total transfer kb": transfer_kb,
                }

                with results_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                successful_questions += 1

                logger.info(
                    f"Sample {idx + 1}/{len(indices)}"
                    f" | ID: {sample_id}"
                    f" | Std SD: {sd_latency:.2f}s"
                    f" | Fixed Tree: {fixed_latency:.2f}s ({fixed_speedup:.2f}x)"
                    f" | TALON: {talon_latency:.2f}s ({talon_speedup:.2f}x)"
                    f" | fixed total transfer kb: {fixed_metrics["total_transfer_kb"]} KB"
                    f" | talon total transfer kb: {talon_metrics["total_transfer_kb"]} KB"
                    f" | sd total transfer kb: {transfer_kb} KB"
                )

            except Exception as e:
                failed_questions += 1
                question_processing_time = time.perf_counter() - question_start

                logger.error(
                    f"Failed on dataset index {ds_idx} "
                    f"after {question_processing_time:.2f}s: {e}\n"
                    f"{traceback.format_exc()}"
                )

            finally:
                if question_processing_time is None:
                    question_processing_time = time.perf_counter() - question_start

                total_question_time += question_processing_time

                logger.info(
                    f"Question {idx + 1}/{len(indices)} finished "
                    f"in {question_processing_time:.2f} seconds"
                )

        questions_attempted = successful_questions + failed_questions
        average_question_time = total_question_time / questions_attempted if questions_attempted > 0 else 0.0

        logger.info("=" * 80)
        logger.info("BENCHMARK TIME SUMMARY")
        logger.info(f"Total questions attempted: {questions_attempted}")
        logger.info(f"Successful questions: {successful_questions}")
        logger.info(f"Failed questions: {failed_questions}")
        logger.info(f"Total processing time: {total_question_time:.2f} seconds")
        logger.info(f"Average time per question: {average_question_time:.2f} seconds")
        logger.info("=" * 80)

        talon_speedups_by_cat = defaultdict(list)
        fixed_speedups_by_cat = defaultdict(list)
        
        if results_file.exists():
            with results_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        saved = json.loads(line)
                        cat = saved["dataset_category"]
                        talon_speedups_by_cat[cat].append(float(saved["talon_speedup_vs_sd"]))
                        fixed_speedups_by_cat[cat].append(float(saved["fixed_speedup_vs_sd"]))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue

        logger.info("SPEEDUP BY DATASET CATEGORY (Relative to Standard SD)")
        for category in sorted(talon_speedups_by_cat):
            t_vals = talon_speedups_by_cat[category]
            f_vals = fixed_speedups_by_cat[category]
            logger.info(
                f"{category:16s} | n={len(t_vals):3d} "
                f"| Fixed Mean={mean(f_vals):.3f}x "
                f"| TALON Mean={mean(t_vals):.3f}x "
            )
        logger.info("=" * 80)

        if failed_questions == 0:
            logger.info("Benchmark finished successfully.")
        else:
            logger.warning(f"Benchmark finished with {failed_questions} failed questions.")     
            
    except Exception as e:
        logger.error(f"Fatal benchmark error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()