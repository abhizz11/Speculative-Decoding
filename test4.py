# Using HuggingFace Universal Assisted Generation to emulate something like speculative decoding

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

    return text, new_tokens, token_probs, distributions


def speculative_decoding(model, assistant_model, tokenizer, assistant_tokenizer, prompt, max_new_tokens=30, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = large_model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer,pad_token_id=large_tokenizer.eos_token_id, max_new_tokens=30, return_dict_in_generate=True)

    res = large_tokenizer.batch_decode(outputs.sequences[0], skip_special_tokens=True)

    return res


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
    large_model, 
    small_model,
    large_tokenizer, 
    small_tokenizer, 
    prompt, 
    max_new_tokens, 
    device
)

print(normal_outputs[0], normal_latency)
print(sd_outputs, sd_latency)


# Results
'''
Once upon a time, there was a little girl named Alice who lived in a small village. One day, Alice was playing in the meadow when she saw a rabbit running away from her. Alice chased the rabbit and caught it, but it 1.2139147724956274
['Once upon a time, there was a little girl named Alice who lived in a small village. She was a very kind and gentle girl, and she loved to help others. One day, she was walking through the forest when she saw a little'] 1.7380648087710142

'''