# Huggingface TLI (token level intersection)
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import time

# Set Seed
set_seed(42)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Target and Tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
).to(device)

# Draft and tokenizer
assistant_name = "openai-community/gpt2-large"
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_name)
assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_name, 
    torch_dtype=torch.bfloat16, 
).to(device)

model.eval()
assistant_model.eval()

prompt = "Once upon a time, there was a little girl named Alice who lived in a small village."




def normal_inference(model, tokenizer, prompt, max_new_tokens=30, device="cuda"):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, attention_mask=inputs["attention_mask"],  pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.2)
                                # temperature=0.8,       # control randomness
                                # top_p=0.9)             # nucleus sampling
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


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


def speculative_decoding(tokenizer, assistant_tokenizer, assistant_model, model, prompt, device, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        assistant_model=assistant_model,
        tokenizer=tokenizer,
        assistant_tokenizer = assistant_tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,   # Must be True, else will do greedy instead of TLI
        temperature=0.2
    )

    # Output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

normal_latency, normal_output = measure_latency(
    normal_inference,
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    device="cuda"
)

print(normal_latency, normal_output)

print("\n" * 2)

speculative_latency, speculative_output = measure_latency(
    speculative_decoding,
    tokenizer,
    assistant_tokenizer,
    assistant_model,
    model,
    prompt,
    device,
    max_new_tokens=50
)

print(speculative_latency, speculative_output)
