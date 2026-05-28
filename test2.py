# Speculative Decodin Algorithm but this one gave me gibberish output

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Setting seed for reproducible outputs
set_seed(42)

# Checking for GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model names
small_model_name = "meta-llama/Llama-3.2-1B"
big_model_name = "meta-llama/Llama-3.2-3B"

# Loading tokenizers
big_tokenizer = AutoTokenizer.from_pretrained(big_model_name)
big_tokenizer.pad_token = big_tokenizer.eos_token

# Loading models
print("Loading models")
small_model = AutoModelForCausalLM.from_pretrained(small_model_name, device_map="auto", dtype=torch.bfloat16)
big_model = AutoModelForCausalLM.from_pretrained(big_model_name, device_map="auto", dtype=torch.bfloat16)

def normal_inference(big_model, big_tokenizer, prompt, max_new_tokens=50):
    """Standard token-by-token generation using the large target model."""
    inputs = big_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = big_model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            pad_token_id=big_tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens
        )
    return big_tokenizer.decode(outputs[0], skip_special_tokens=True)

def speculative_decoding(small_model, big_model, tokenizer, prompt, max_new_tokens=50, K=4):
    """
    Speculative decoding algorithm
    Calculate and compare vocab distribution between small and big models.
    K = gamma (the number of tokens the small model drafts at a time).
    """

    inputs = tokenizer(prompt, return_tensors = "pt").to(device)
    current_tokens = inputs['input_ids']

    # Initial prompt length
    prompt_length = current_tokens.size(1)

    # Generate until the number of new tokens generated is max_new_tokens
    while(current_tokens.size(1) - prompt_length) < max_new_tokens:
        N = current_tokens.size(1) # This keeps changing

        draft_tokens = []
        small_probs_list = []

        for _ in range(K):
            with torch.no_grad():
                small_outputs = small_model(current_tokens)
                small_logits = small_outputs.logits[0, -1, :] # Extract the probability of very last token
                small_probs = F.softmax(small_logits, dim=-1) # Normalize

            next_token = torch.argmax(small_probs).unsqueeze(0).unsqueeze(0) # Greedily sample the next token 
            draft_tokens.append(next_token.item())
            small_probs_list.append(small_probs)

            # Concatenate the new token into existing token
            current_tokens = torch.cat([current_tokens, next_token], dim=-1)

        with torch.no_grad():
            big_outputs = big_model(current_tokens) # Passing original text + K draft tokens
            big_logits = big_outputs.logits[0, :, :] # Extract the raw logits
            big_probs_matrix = F.softmax(big_logits, dim=-1) # Convert to massive matrix of vocab distributions
        
        # Rejection sampling
        accepted_count = 0
        for i in range(K):
            token_id = draft_tokens[i]

            p_x = small_probs_list[i][token_id].item() # Small probability token
            
            # target model is looking at N - 1 + ith token (new one)
            big_matrix_index = N - 1 + i
            q_x = big_probs_matrix[big_matrix_index, token_id].item()

            # If big model's probability is higher accept
            if q_x >= p_x:
                accepted_count += 1
            else:
                # Gamble the fallback step
                random_chance = torch.rand(1).item()
                if random_chance < (q_x / p_x):
                    accepted_count += 1
                else:
                    # Rejection sampling didn't work
                    break
            
        
        # Rollback the tokens that were rejected
        roll_back = K - accepted_count
        if roll_back > 0:
            current_tokens = current_tokens[:, :-roll_back]
        
        # Big model provides a guaranteed token at the point of termination
        correction_index = N - 1 + accepted_count
        corrected_token_logits = big_probs_matrix[correction_index, :]
        corrected_token = torch.argmax(corrected_token_logits).unsqueeze(0).unsqueeze(0)

        # Append the verified correct token to the sequence
        current_tokens = torch.cat([current_tokens, corrected_token], dim=-1)
    
    final_output_text = tokenizer.decode(current_tokens[0], skip_special_tokens = True)
    return final_output_text

def measure_latency(small_model, big_model, tokenizer, prompt, max_new_tokens=50):
    """Helper function to run and time both processing methodologies."""
    print(f"Prompt: '{prompt}'")
    print("-" * 50)
    
    # Run Normal Inference
    start_time = time.time()
    normal_output = normal_inference(big_model, tokenizer, prompt, max_new_tokens)
    normal_latency = time.time() - start_time
    print(f"Normal Output: {normal_output}")
    print(f"Normal Latency: {normal_latency:.4f} seconds\n")
    
    # Run Speculative Decoding Inference
    start_time = time.time()
    speculative_output = speculative_decoding(small_model, big_model, tokenizer, prompt, max_new_tokens)
    speculative_latency = time.time() - start_time
    print(f"Speculative Output: {speculative_output}")
    print(f"Speculative Latency: {speculative_latency:.4f} seconds")
    print("=" * 50 + "\n")
    
    return normal_latency, speculative_latency


prompts = [
        "Generative models like GPT-3 can create ",
        "How AI is transforming the world "
    ]
    
max_new_tokens = 30  # Sized for evaluation speed
total_normal_latency = 0
total_speculative_latency = 0

# Since Llama-3 models share structural, use the same tokenizer
shared_tokenizer = big_tokenizer 

print("\nStarting evaluation loop...\n")
for prompt in prompts:
    norm_lat, spec_lat = measure_latency(small_model, big_model, shared_tokenizer, prompt, max_new_tokens)
    total_normal_latency += norm_lat
    total_speculative_latency += spec_lat
    
print(f"Total normal latency for {len(prompts)} prompts: {total_normal_latency:.4f} seconds")
print(f"Total speculative latency for {len(prompts)} prompts: {total_speculative_latency:.4f} seconds")

speedup = total_normal_latency / total_speculative_latency if total_speculative_latency > 0 else 0
print(f"Calculated Speedup Factor: {speedup:.2f}x")

