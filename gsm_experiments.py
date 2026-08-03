import json
import logging
import random
import re
import traceback
from decimal import Decimal, InvalidOperation
from pathlib import Path
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from fixedTreeSD import fixed_tree_speculative_generate_greedy, run_normal_baseline

_NUMBER = r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"

set_seed(40)

def build_prompt(question):
        return (f"Here is a math problem. Think carefully and solve it step by step. {question} ")

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
    
    run_handler = logging.FileHandler(logs_dir / "run_train.log", mode="a")
    run_handler.setFormatter(formatter)
    root.addHandler(run_handler)
    
    error_handler = logging.FileHandler(logs_dir / "errors_train.log", mode="a")
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
    SPLIT = "train"
    MAX_NEW_TOKENS = 512
    BUDGET = 15
    MAX_K = 16
    MU = 0.03
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

        logger.info(f"Loading dataset {DATASET_NAME}...")
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
        
        results_file = output_dir / "results_train.jsonl"

        # Keep one dataset index for each unique question.
        seen_questions = set()
        unique_indices = []

        for ds_idx, item in enumerate(dataset):
            normalized_question = " ".join(item["question"].split())

            if normalized_question not in seen_questions:
                seen_questions.add(normalized_question)
                unique_indices.append(ds_idx)

        # Read previously completed samples so an interrupted experiment can resume.
        completed_ids = set()

        if results_file.exists():
            with results_file.open("r") as f:
                for line_number, line in enumerate(f, start=1):
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        logger.warning(
                            f"Ignoring invalid results line {line_number}"
                        )

        # Run only unique questions that have not already completed.
        indices = [
            ds_idx
            for ds_idx in unique_indices
            if ds_idx not in completed_ids
        ]

        logger.info(f"Dataset rows: {len(dataset)}")
        logger.info(f"Unique questions: {len(unique_indices)}")
        logger.info(f"Previously completed: {len(completed_ids)}")
        logger.info(f"Questions remaining: {len(indices)}")

        for idx, ds_idx in enumerate(indices):
            question_start = time.perf_counter()
            question_processing_time = None

            try:
                item = dataset[ds_idx]
                prompt = build_prompt(item["question"])
                gold_answer = extract_gold_answer(item["answer"])

                logger.info(
                    f"Processing question {idx + 1}/{len(indices)} "
                    f"(dataset index {ds_idx})"
                )

                # Baseline run
                torch.cuda.reset_peak_memory_stats()

                base_latency, base_tokens, _, base_output_ids = run_normal_baseline(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    target_model=target_model,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device=DEVICE,
                )

                prompt_len = tokenizer.encode(
                    prompt,
                    return_tensors="pt",
                ).shape[1]

                base_new_token_ids = base_output_ids[
                    0, prompt_len:
                ].tolist()

                base_new_text = tokenizer.decode(
                    base_new_token_ids,
                    skip_special_tokens=True,
                )

                base_extracted = extract_prediction(
                    base_new_text,
                    flexible=True,
                )
                base_correct = answers_equal(
                    base_extracted,
                    gold_answer,
                )

                # TALON run
                torch.cuda.reset_peak_memory_stats()

                talon_res, talon_metrics, _ = (
                    fixed_tree_speculative_generate_greedy(
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
                )

                talon_new_text = talon_res["new_text"]

                talon_extracted = extract_prediction(
                    talon_new_text,
                    flexible=True,
                )
                talon_correct = answers_equal(
                    talon_extracted,
                    gold_answer,
                )

                talon_latency = float(talon_res["latency"])
                speedup = (
                    float(base_latency) / talon_latency
                    if talon_latency > 0
                    else 0.0
                )

                text_match = base_new_text == talon_new_text

                # Calculate once for the JSON record.
                question_processing_time = (
                    time.perf_counter() - question_start
                )

                processing_overhead = max(
                    0.0,
                    question_processing_time
                    - float(base_latency)
                    - talon_latency,
                )

                record = {
                    "sample_id": ds_idx,
                    "dataset_split": SPLIT,
                    "prompt": prompt,
                    "reference_solution": item["answer"],
                    "gold_answer": gold_answer,
                    "baseline_new_text": base_new_text,
                    "talon_new_text": talon_new_text,
                    "baseline_extracted_answer": base_extracted,
                    "talon_extracted_answer": talon_extracted,
                    "baseline_correct": base_correct,
                    "talon_correct": talon_correct,
                    "question_processing_time_s": float(
                        question_processing_time
                    ),
                    "processing_overhead_s": float(
                        processing_overhead
                    ),
                    "baseline_latency_s": float(base_latency),
                    "talon_latency_s": talon_latency,
                    "speedup": float(speedup),
                    "text_exact_match": text_match,
                    "talon_metrics": talon_metrics,
                }

                with results_file.open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(record) + "\n")

                successful_questions += 1

                logger.info(
                    f"Sample {idx + 1}/{len(indices)}"
                    f" | Dataset ID: {ds_idx}"
                    f" | Question Time: {question_processing_time:.2f}s"
                    f" | Baseline: {base_latency:.2f}s"
                    f" | TALON: {talon_latency:.2f}s"
                    f" | Overhead: {processing_overhead:.2f}s"
                    f" | Speedup: {speedup:.2f}x"
                    f" | Parity: {text_match}"
                    f" | Base Correct: {base_correct}"
                    f" | TALON Correct: {talon_correct}"
                )

            except Exception as e:
                failed_questions += 1
                question_processing_time = (
                    time.perf_counter() - question_start
                )

                logger.error(
                    f"Failed on sample {ds_idx} "
                    f"after {question_processing_time:.2f}s: {e}\n"
                    f"{traceback.format_exc()}"
                )

            finally:
                # Safety fallback in case timing failed before assignment.
                if question_processing_time is None:
                    question_processing_time = (
                        time.perf_counter() - question_start
                    )

                # Add exactly once, regardless of success or failure.
                total_question_time += question_processing_time

                logger.info(
                    f"Question {idx + 1}/{len(indices)} finished "
                    f"in {question_processing_time:.2f} seconds"
                )

        questions_attempted = (
            successful_questions + failed_questions
        )

        average_question_time = (
            total_question_time / questions_attempted
            if questions_attempted > 0
            else 0.0
        )

        logger.info("=" * 80)
        logger.info("BENCHMARK TIME SUMMARY")
        logger.info(
            f"Total questions attempted: {questions_attempted}"
        )
        logger.info(
            f"Successful questions: {successful_questions}"
        )
        logger.info(
            f"Failed questions: {failed_questions}"
        )
        logger.info(
            f"Total processing time: "
            f"{total_question_time:.2f} seconds"
        )
        logger.info(
            f"Total processing time: "
            f"{total_question_time / 60:.2f} minutes"
        )
        logger.info(
            f"Total processing time: "
            f"{total_question_time / 3600:.2f} hours"
        )
        logger.info(
            f"Average time per question: "
            f"{average_question_time:.2f} seconds"
        )
        logger.info("=" * 80)

        if failed_questions == 0:
            logger.info("Benchmark finished successfully.")
        else:
            logger.warning(
                f"Benchmark finished with "
                f"{failed_questions} failed questions."
            )     
    except Exception as e:
        logger.error(f"Fatal benchmark error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()