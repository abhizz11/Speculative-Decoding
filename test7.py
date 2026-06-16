import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, LogitsProcessor, LogitsProcessorList
import time

# Set Seed
set_seed(42)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Using heterogeneous models as requested (Gemma and Vicuna)
small_model_id = "double7/vicuna-68m" 
large_model_id = "google/gemma-2-2b-it"

small_tokenizer = AutoTokenizer.from_pretrained(small_model_id)
small_model = AutoModelForCausalLM.from_pretrained(small_model_id).to(device)

large_tokenizer = AutoTokenizer.from_pretrained(large_model_id)
large_model = AutoModelForCausalLM.from_pretrained(large_model_id).to(device)


# Creates a dictionary mapping small_model token IDs to large_model token IDs based on exact string match.
def get_vocabulary_intersection(small_tokenizer, large_tokenizer):
    small_vocab = small_tokenizer.get_vocab()
    large_vocab = large_tokenizer.get_vocab()
    
    intersection_mapping = {}
    for token_text, small_id in small_vocab.items():
        if token_text in large_vocab:
            intersection_mapping[small_id] = large_vocab[token_text]
            
    return intersection_mapping

# Forces the drafter to only output tokens present in the intersection.
class TLILogitsProcessor(LogitsProcessor):
    def __init__(self, valid_ids):
        self.valid_ids = torch.tensor(list(valid_ids), device=device)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float('inf'))
        mask[:, self.valid_ids] = 0
        return scores + mask

intersection_mapping = get_vocabulary_intersection(small_tokenizer, large_tokenizer)
logits_processor = LogitsProcessorList([TLILogitsProcessor(intersection_mapping.keys())])

# Checking to see if the small tokenizer has enough vocabs after intersection mapping
print(f"Small vocab size:    {len(small_tokenizer.get_vocab())}")
print(f"Large vocab size:    {len(large_tokenizer.get_vocab())}")
print(f"Intersection size:   {len(intersection_mapping)}")
print(f"Coverage (small):    {len(intersection_mapping)/len(small_tokenizer.get_vocab()):.1%}")


sample_ids = list(intersection_mapping.keys())[:5]
for sid in sample_ids:
    tok = small_tokenizer.convert_ids_to_tokens(sid)
    large_id = intersection_mapping[sid]
    large_tok = large_tokenizer.convert_ids_to_tokens(large_id)
    print(f"  '{tok}' (small:{sid}) → '{large_tok}' (large:{large_id})")


# ==========================================
# Inference Functions
# ==========================================

def normal_inference(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs['input_ids'], 
        max_new_tokens=max_new_tokens, 
        attention_mask=inputs["attention_mask"],  
        pad_token_id=tokenizer.eos_token_id, 
        return_dict_in_generate=True, 
        output_scores=True,
    )
    
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


def uag_tli_speculative_decoding_inference(large_model, small_model, large_tokenizer, small_tokenizer, prompt, device, max_new_tokens=50, gamma=4, intersection_mapping=intersection_mapping, logits_processor=logits_processor):
    small_model.eval()
    large_model.eval()

    current_text = prompt
    
    large_inputs = large_tokenizer(prompt, return_tensors="pt").to(device)
    generated = large_inputs["input_ids"] # Target context
    
    prompt_len = generated.shape[1]
    target_len = prompt_len + max_new_tokens

    accepted_tokens = 0
    total_draft_tokens = 0

    while generated.shape[1] < target_len:
        old_len = generated.shape[1]

        # Tokenize the current synced text for the drafter
        small_inputs = small_tokenizer(current_text, return_tensors="pt").to(device)
        small_generated = small_inputs["input_ids"]
        small_old_len = small_generated.shape[1]

        #  1: Drafting (Masked to Intersection)
        draft_outputs = small_model.generate(
            input_ids=small_generated,
            max_new_tokens=gamma,
            pad_token_id=small_tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=logits_processor
        )

        draft_sequence = draft_outputs.sequences[0]
        draft_tokens = draft_sequence[small_old_len:]

        if len(draft_tokens) == 0:
            break

        total_draft_tokens += len(draft_tokens)

        q_distributions = [
            torch.softmax(score[0], dim=-1)
            for score in draft_outputs.scores
        ]

        #  2: Translation to Target Space
        mapped_draft_tokens = torch.tensor(
            [intersection_mapping[t.item()] for t in draft_tokens], 
            device=device
        )

        candidate_ids = torch.cat(
            [generated, mapped_draft_tokens.unsqueeze(0)],
            dim=1
        )

        #  3: Verification
        with torch.no_grad():
            target_outputs = large_model(input_ids=candidate_ids)

        accepted_list = []
        rejected = False
        reject_index = None

        #  4: Rejection Sampling
        for i, small_token_id in enumerate(draft_tokens):
            small_token_id = small_token_id.item()
            large_token_id = mapped_draft_tokens[i].item()
            
            q_dist = q_distributions[i]
            q_prob = q_dist[small_token_id].item()

            target_logits = target_outputs.logits[0, old_len + i, :]
            p_dist = torch.softmax(target_logits, dim=-1)
            p_prob = p_dist[large_token_id].item()

            if q_prob > 0:
                accept_prob = min(1.0, p_prob / q_prob)
            else:
                accept_prob = 1.0

            random_number = torch.rand(1).item()

            if random_number <= accept_prob:
                accepted_list.append(torch.tensor([[large_token_id]], device=device))
                accepted_tokens += 1
            else:
                rejected = True
                reject_index = i
                break

        if len(accepted_list) > 0:
            accepted_tensor = torch.cat(accepted_list, dim=1)
            generated = torch.cat([generated, accepted_tensor], dim=1)
            
            # Sync text context for the drafter's next loop
            added_text = large_tokenizer.decode(accepted_tensor[0], skip_special_tokens=True)
            current_text += added_text

        if generated.shape[1] >= target_len:
            break

        # Resampling / Bonus Token
        if rejected:
            q_dist = q_distributions[reject_index]
            target_logits = target_outputs.logits[0, old_len + reject_index, :]
            p_dist = torch.softmax(target_logits, dim=-1)

            # Map the drafter's distribution into the target's vocabulary space for subtraction
            mapped_q_dist = torch.zeros_like(p_dist)
            for s_id, l_id in intersection_mapping.items():
                mapped_q_dist[l_id] = q_dist[s_id]

            adjusted_dist = torch.clamp(p_dist - mapped_q_dist, min=0)

            if adjusted_dist.sum().item() == 0:
                adjusted_dist = p_dist
            else:
                adjusted_dist = adjusted_dist / adjusted_dist.sum()

            next_token = torch.multinomial(
                adjusted_dist,
                num_samples=1
            ).view(1, 1)

            generated = torch.cat([generated, next_token], dim=1)
            current_text += large_tokenizer.decode(next_token[0], skip_special_tokens=True)

        else:
            # If all accepted, sample one bonus token
            next_logits = target_outputs.logits[0, -1, :]
            next_dist = torch.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                next_dist,
                num_samples=1
            ).view(1, 1)

            generated = torch.cat([generated, next_token], dim=1)
            current_text += large_tokenizer.decode(next_token[0], skip_special_tokens=True)

        if large_tokenizer.eos_token_id is not None:
            if generated[0, -1].item() == large_tokenizer.eos_token_id:
                break

    final_ids = generated[0, :target_len]
    final_text = large_tokenizer.decode(final_ids, skip_special_tokens=True)
    final_text = " ".join(final_text.split())
    acceptance_rate = accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0

    return final_text, accepted_tokens, total_draft_tokens, acceptance_rate

# Measures execution time of a function, synchronizing CUDA if available.
def measure_latency(func, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return latency, result

prompt = "Once upon a time, there was a little girl named Alice who lived in a small village."
max_new_tokens = 50

# Measure Normal Inference
normal_latency, normal_outputs = measure_latency(
    normal_inference, 
    large_model, 
    large_tokenizer, 
    prompt, 
    max_new_tokens
)

# Measure Speculative Decoding (TLI)
sd_latency, sd_outputs = measure_latency(
    uag_tli_speculative_decoding_inference, 
    large_model, 
    small_model, 
    large_tokenizer, 
    small_tokenizer,
    prompt, 
    device, 
    max_new_tokens, 
    gamma=2
)


print("-" * 50)
print(f"Normal Inference Latency: {normal_latency:.4f} seconds")
print(normal_outputs[0])
print("-" * 50)
print(f"Speculative Decoding Latency: {sd_latency:.4f} seconds")
print(sd_outputs[0])
print(f"Speedup: {normal_latency / sd_latency:.2f}x")
print("-" * 50)