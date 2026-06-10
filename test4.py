# Using HuggingFace Universal Assisted Generation to emulate something like speculative decoding, manual

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Set Seed
set_seed(42)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


small_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large") # 0.8 B parameters
small_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large").to(device)

large_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
large_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B").to(device)

max_new_tokens = 30


small_model.eval()
large_model.eval()


def normal_inference(model, tokenizer, prompt, max_new_tokens=30, device="cuda"):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, attention_mask=inputs["attention_mask"],  pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True,)
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

    return text


def speculative_decoding(prompt, target_model, target_tokenizer, assistant_model, assistant_tokenizer, device="cuda", max_new_tokens=30):
    current_text = prompt
    generated_tokens = 0
    gamma = 5

    accepted_texts = []

    while generated_tokens < max_new_tokens:
        # Draft generation
        ast_inputs = assistant_tokenizer(current_text, return_tensors="pt").to(device)

        # Generate gamma tokens
        ast_draft_outputs = assistant_model.generate(
            ast_inputs["input_ids"],
            max_new_tokens=gamma,
            attention_mask= ast_inputs["attention_mask"],  
            pad_token_id=assistant_tokenizer.eos_token_id, 
        )

        # Extract the new draft tokens
        prompt_len = ast_inputs.input_ids.shape[1]
        draft_tokens = ast_draft_outputs[0][prompt_len:]


        # Decode draft tokens to plain text
        draft_text = assistant_tokenizer.decode(draft_tokens, skip_special_tokens=True)
        proposed_text = current_text + draft_text


        # Tokenize the decoded tokens through target model
        tgt_inputs = target_tokenizer(proposed_text, return_tensors="pt").to(device)
        tgt_prompt_ids = target_tokenizer(current_text, return_tensors="pt").input_ids

        tgt_prompt_len = tgt_prompt_ids.shape[1]
        proposed_tgt_tokens = tgt_inputs.input_ids[0][tgt_prompt_len:]

        with torch.no_grad():
            outputs = target_model(tgt_inputs["input_ids"])
            logits = outputs.logits[0]
        
        accepted_tokens = []
        
        # Acceptance / Rejection
        for i in range(len(proposed_tgt_tokens)):
            step_logits = logits[tgt_prompt_len + i - 1]

            target_expected_token = torch.argmax(step_logits).item()
            proposed_token = proposed_tgt_tokens[i].item()

            if target_expected_token == proposed_token:
                accepted_tokens.append(target_expected_token)
            else:
                accepted_tokens.append(target_expected_token)
                break
            

        # Decode only verified + corrected tokens
        accepted_ids_tensor = torch.tensor([accepted_tokens])
        accepted_text = target_tokenizer.decode(accepted_ids_tensor[0], skip_special_tokens=True)

        current_text += accepted_text
        generated_tokens += len(accepted_tokens)

        accepted_texts.append(accepted_text)

        if target_tokenizer.eos_token_id in accepted_tokens:
            break
    
    return current_text, accepted_texts, len(accepted_texts)


def uag()


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



prompt = "Once upon a time, there was a little girl named Alice who lived in a small village."


# Measure Normal Inference
normal_latency, normal_outputs = measure_latency(
    normal_inference, 
    large_model, 
    large_tokenizer, 
    prompt, 
    max_new_tokens,
    device
)

# Measure Speculative Decoding
sd_latency, sd_outputs = measure_latency(
    speculative_decoding, 
    prompt, 
    large_model,
    large_tokenizer, 
    small_model, 
    small_tokenizer,
    device, 
    max_new_tokens, 
)

print(normal_outputs, normal_latency)
print(sd_outputs[0], sd_outputs[2], sd_latency)

