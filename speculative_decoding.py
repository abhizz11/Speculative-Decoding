# Actual Speculative Decoding Algorithm with no gibberish output
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Set Seed
set_seed(42)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

small_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
small_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)

large_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
large_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B").to(device)

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

    while generated.shape[1] < target_len:
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

    return final_text, accepted_tokens, total_draft_tokens, acceptance_rate

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


prompt = "Once upon a time, there was a little girl named Alice who lived in a small village. The village was nestled in a lush valley surrounded by towering, snow-capped mountains. Every morning, Alice would wander into the nearby woods to collect wildflowers and listen to the birds sing. She was known by all her neighbors for her boundless curiosity and bright, infectious smile. Despite the quiet nature of her home, Alice secretly dreamed of embarking on a grand adventure beyond the horizon."
max_new_tokens = 50

# Measure Normal Inference
normal_latency, normal_outputs = measure_latency(
    normal_inference, 
    large_model, 
    large_tokenizer, 
    prompt, 
    max_new_tokens
)

# Measure Speculative Decoding
sd_latency, sd_outputs = measure_latency(
    speculative_decoding_inference, 
    large_model, 
    small_model, 
    small_tokenizer, 
    prompt, 
    device, 
    max_new_tokens, 
    gamma=1
)

print("-" * 50)
print(f"Normal Inference Latency: {normal_latency:.4f} seconds")
print(f"Speculative Decoding Latency: {sd_latency:.4f} seconds")
print(f"Speedup: {normal_latency / sd_latency:.2f}x")
print("-" * 50)