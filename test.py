import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Setting seed so that everytime we get the same set of random answers
set_seed(42)

# Checking for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model names
small = "meta-llama/Llama-3.2-1B-Instruct"
big = "meta-llama/Llama-3.2-3B-Instruct"

# Loading small model
small_tokenizer = AutoTokenizer.from_pretrained(small, device_map="auto")
small_model = AutoModelForCausalLM.from_pretrained(small, device_map="auto", dtype=torch.bfloat16)
# Loading bigger(target) model
big_tokenizer = AutoTokenizer.from_pretrained(big, device_map="auto")
big_model = AutoModelForCausalLM.from_pretrained(big, device_map="auto", dtype=torch.bfloat16)

# Normal inference to evaluate the increase in speed
def normal_inference(big_model, big_tokenizer, prompt, max_new_tokens=50):
    inputs = big_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = big_model.generate(inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            pad_token_id=big_tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens),

    return big_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Speculative decoding function
def speculative_decoding(small_model, big_model, small_tokenizer, big_tokenizer, prompt, max_new_tokens = 50):
    # First we use the small model to generate draft
    inputs = small_tokenizer(prompt, return_tensors="pt").to(device)
    small_outputs = small_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        pad_token_id=small_tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens)
    draft = small_tokenizer.decode(small_outputs[0], skip_special_tokens=True)

    # Verifying the draft with big model
    big_inputs = big_tokenizer(draft, return_tensors='pt').to(device)

    # Calculate Loglikelihood but skip gradient descent cause it's mostly used for training
    with torch.no_grad():
        outputs = big_model(big_inputs['input_ids'])
        log_probs = torch.log_softmax(outputs.logits, dim=-1) # Softmax normalizes the vector

    draft_token_ids = big_inputs['input_ids']
    log_likelihood = 0
    
    
    # Python tensors are like python lists but optimized. They use [row, col] slicing instead of chaining
    for i in range(draft_token_ids.size(1)-1):
        token_id = draft_token_ids[0, i+1] # Always start at [0,1] cause the first word is context
        log_likelihood += log_probs[0, i, token_id].item()
    
    avg_log_likelihood = log_likelihood / (draft_token_ids.size(1) - 1)

    # Return the draft and its log likelihood score
    return draft, avg_log_likelihood

# function to measure latency between normal decoding and speculative decoding
def measure_latency(small_model, big_model, small_tokenizer, big_tokenizer, prompt, max_new_tokens=50):
    start_time = time.time()
    normal_output = normal_inference(big_model, big_tokenizer, prompt, max_new_tokens)
    normal_inference_latency = time.time() - start_time

    print(f"Normal Inference output: {normal_output}")
    print(f"Normal Inference latency: {normal_inference_latency:.4f} seconds")
    print("\n\n")

    start_time = time.time()
    speculative_output, log_likelihood = speculative_decoding(
        small_model, big_model, small_tokenizer, big_tokenizer, prompt, max_new_tokens
    )

    speculative_decoding_latency = time.time() - start_time
    print(f"Speculative decoding output: {speculative_output}")
    print(f"Speculative decoding latency: {speculative_decoding_latency:.4f} seconds")
    print(f"Log likelihood (verification score): {log_likelihood:.4f}")

    return normal_inference_latency, speculative_decoding_latency


# List of prompts
prompts = [
    "I love playing chess so that means ",
    "Generative models like GPT-3 can create ",
    "My favorite food is  ",
    "How AI is transforming the world "
]

# Inference settings
max_new_tokens = 50


total_normal_latency = 0
total_speculative_latency = 0

for prompt in prompts:
    normal_latency, speculative_latency = measure_latency(
        small_model, big_model, small_tokenizer, big_tokenizer, prompt, max_new_tokens
    )
    total_normal_latency += normal_latency
    total_speculative_latency += speculative_latency

print(f"Total normal latency for {len(prompts)} prompts is {total_normal_latency:.4f} seconds")
print(f"Total speculative latency for {len(prompts)} prompts is {total_speculative_latency:.4f} seconds")




