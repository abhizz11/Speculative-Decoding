from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

target_model_id = "google/gemma-2-2b-it"  # Target: Gemma family (~2.6B params)
assistant_model_id = "double7/vicuna-68m" # Drafter: LLaMA family (~68M params)

tokenizer = AutoTokenizer.from_pretrained(target_model_id)
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load the Target Model
model = AutoModelForCausalLM.from_pretrained(
    target_model_id,
    torch_dtype=torch.bfloat16,
).to(device)

# Load the Assistant Model
assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id,
    torch_dtype=torch.bfloat16,
).to(device)

prompt = "There was a little girl named Alice who lived in a village"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

start_time = time.perf_counter()
print("The start_time for normal is: ", start_time)

output_normal = model.generate(
    **inputs,
    tokenizer=tokenizer,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id, # Prevents pad token warnings
)

print(tokenizer.decode(output_normal[0], skip_special_tokens = True, clean_up_tokenization_spaces=False))
end_time = time.perf_counter()
latency = end_time - start_time
print("\n" * 3)
print(f"The latency for normal is: {latency}")


print("\n" * 3)


spec_start = time.perf_counter()
print("The start_time for speculative decoding is: ", spec_start)

outputs = model.generate(
    **inputs,
    tokenizer=tokenizer,
    assistant_model=assistant_model,
    assistant_tokenizer=assistant_tokenizer,
    max_new_tokens=100,
    num_assistant_tokens= 3,    # gamma value
    do_sample=True,             # Required for UAG-TLI probabilistic decoding
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))

print("\n" * 3)
spec_end_time = time.perf_counter()
spec_latency = spec_end_time - spec_start
print("\n" * 3)
print(f"The latency for speculative decoding is: ", spec_latency)