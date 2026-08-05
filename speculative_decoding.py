# Actual Speculative Decoding Algorithm with no gibberish output
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Calculates the total size of passed tensors in kilobytes.
def get_transfer_size_kb(*tensors):
    return sum(t.nelement() * t.element_size() for t in tensors if t is not None) / 1024

def normal_inference(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, attention_mask=inputs["attention_mask"],  pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True, temperature=0.8, top_p=0.9)
                                # do_sample=True,        # enable sampling
                                # temperature=0.8,       # control randomness
                                # top_p=0.9)             # nucleus sampling
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs.sequences[0][input_len:]

    distributions = []
    token_probs = []

    for token_id, score in zip(new_tokens, outputs.scores):
        prob_dist = torch.softmax(score[0], dim=-1)
        prob = prob_dist[token_id].item()

        distributions.append(prob_dist)
        token_probs.append((tokenizer.decode([token_id]), prob))

    return text, new_tokens, token_probs, distributions


def speculative_decoding_inference(large_model, small_model, tokenizer, prompt, device, max_new_tokens=50, gamma=4):
    small_model.eval()
    large_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()
    prompt_len = input_ids.shape[1]
    target_len = prompt_len + max_new_tokens

    accepted_tokens = 0
    total_draft_tokens = 0
    total_transfer_kb = 0.0 
    iterations = 0 

    while generated.shape[1] < target_len:
        iterations += 1
        old_len = generated.shape[1]

        draft_outputs = small_model.generate(
            input_ids=generated,
            max_new_tokens=gamma,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        draft_sequence = draft_outputs.sequences[0]
        draft_tokens = draft_sequence[old_len:]

        if len(draft_tokens) == 0:
            break

        total_draft_tokens += len(draft_tokens)

        q_distributions = [
            torch.softmax(score[0], dim=-1)
            for score in draft_outputs.scores
        ]

        stacked_q_dist = torch.stack(q_distributions) if q_distributions else None
        transfer_kb = get_transfer_size_kb(draft_tokens, stacked_q_dist)
        total_transfer_kb += transfer_kb

        candidate_ids = torch.cat(
            [generated, draft_tokens.unsqueeze(0)],
            dim=1
        )

        with torch.no_grad():
            target_outputs = large_model(input_ids=candidate_ids)

        accepted_list = []
        rejected = False
        reject_index = None

        for i, token_id in enumerate(draft_tokens):
            q_dist = q_distributions[i]
            q_prob = q_dist[token_id].item()

            target_logits = target_outputs.logits[0, old_len + i - 1, :]
            p_dist = torch.softmax(target_logits, dim=-1)
            p_prob = p_dist[token_id].item()

            if q_prob > 0:
                accept_prob = min(1.0, p_prob / q_prob)
            else:
                accept_prob = 1.0

            random_number = torch.rand(1).item()

            if random_number <= accept_prob:
                accepted_list.append(token_id.view(1, 1))
                accepted_tokens += 1
            else:
                rejected = True
                reject_index = i
                break

        if len(accepted_list) > 0:
            accepted_tensor = torch.cat(accepted_list, dim=1)
            generated = torch.cat([generated, accepted_tensor], dim=1)

        if generated.shape[1] >= target_len:
            break

        if rejected:
            q_dist = q_distributions[reject_index]
            target_logits = target_outputs.logits[0, old_len + reject_index - 1, :]
            p_dist = torch.softmax(target_logits, dim=-1)

            adjusted_dist = torch.clamp(p_dist - q_dist, min=0)

            if adjusted_dist.sum().item() == 0:
                adjusted_dist = p_dist
            else:
                adjusted_dist = adjusted_dist / adjusted_dist.sum()

            next_token = torch.multinomial(
                adjusted_dist,
                num_samples=1
            ).view(1, 1)

            generated = torch.cat([generated, next_token], dim=1)

        else:
            next_logits = target_outputs.logits[0, -1, :]
            next_dist = torch.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                next_dist,
                num_samples=1
            ).view(1, 1)

            generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None:
            if generated[0, -1].item() == tokenizer.eos_token_id:
                break

    final_ids = generated[0, :target_len]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    final_text = " ".join(final_text.split())
    acceptance_rate = accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0

    avg_transfer_kb = total_transfer_kb / iterations if iterations > 0 else 0
    print(f"Average KB Normal Speuclative Decoding: {avg_transfer_kb}")
    return final_text, accepted_tokens, total_draft_tokens, acceptance_rate, final_ids.shape[0] - prompt_len, total_transfer_kb

# ==========================================
# Execution and Measurement
# ==========================================

def measure_latency(func, *args, **kwargs):
    """Measures execution time of a function, synchronizing CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return latency, result


if __name__ == "__main__":
    # Set Seed
    set_seed(42)

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    small_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    small_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(device)

    large_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    large_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large").to(device)

    prompt = "Once upon a time, there was a little girl named Alice "
    max_new_tokens = 100

    # Measure Normal Inference
    normal_latency, normal_outputs = measure_latency(
        normal_inference, 
        large_model, 
        large_tokenizer, 
        prompt, 
        max_new_tokens
    )

    normal_generated_tokens = len(normal_outputs[1])
    normal_throughput = normal_generated_tokens / normal_latency

    # Measure Speculative Decoding
    sd_latency, sd_outputs = measure_latency(
        speculative_decoding_inference, 
        large_model, 
        small_model, 
        small_tokenizer, 
        prompt, 
        device, 
        max_new_tokens, 
        gamma=3
    )

    sd_generated_tokens = sd_outputs[4]
    sd_throughput = sd_generated_tokens / sd_latency
    sd_total_transfer_kb = sd_outputs[5]

    # Peak GPU Memory Metrics
    if device == "cuda":
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0

    print("-" * 50)
    print(f"Normal Inference Latency: {normal_latency:.4f} seconds")
    print(f"Speculative Decoding Latency: {sd_latency:.4f} seconds")
    print(f"Speedup: {normal_latency / sd_latency:.2f}x")
    print(f"Total KB transferred Normal Speculative Decoding: {sd_total_transfer_kb}")
    print(f"Final text Normal SD: {sd_outputs[0]}")
    print(f"Final text Normal: {normal_outputs[0]}")
    print(f"Total Tokens Generated: {sd_generated_tokens}")
    print(f"Sd Tokens Throughput: {sd_throughput}")
    print(f"Total Draft Tokens Accepted:    {sd_outputs[3] * 100}")
    print("-" * 50)