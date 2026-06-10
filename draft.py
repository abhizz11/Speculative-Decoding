import torch
import zmq
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
model.eval()

# Setup ZeroMQ Client
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5555") # Connects through SSH tunnel

prompt = "Once upon a time there was a girl named Alice."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
generated = inputs["input_ids"]

target_len = generated.shape[1] + 50
gamma = 2

start_time = time.perf_counter()

while generated.shape[1] < target_len:
    old_len = generated.shape[1]

    # Generate Draft Tokens
    draft_outputs = model.generate(
        input_ids=generated,
        max_new_tokens=gamma,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    draft_tokens = draft_outputs.sequences[0][old_len:]
    if len(draft_tokens) == 0: break

    q_distributions = [torch.softmax(score[0], dim=-1) for score in draft_outputs.scores]

    # Package and Send to Server
    payload = {
        'generated': generated.cpu(),
        'draft_tokens': draft_tokens.cpu(),
        'q_distributions': [q.cpu() for q in q_distributions] 
    }
    socket.send_pyobj(payload)

    # Wait for Server Verification
    response = socket.recv_pyobj()
    
    # Update Generated Sequence
    accepted_tensor = response['accepted_tensor'].to(device)
    next_token = response['next_token'].to(device)

    if accepted_tensor.shape[1] > 0:
        generated = torch.cat([generated, accepted_tensor], dim=1)
        
    generated = torch.cat([generated, next_token], dim=1)

    if tokenizer.eos_token_id is not None and generated[0, -1].item() == tokenizer.eos_token_id:
        break

print(f"Distributed SD Latency: {time.perf_counter() - start_time:.4f} seconds")
print(tokenizer.decode(generated[0], skip_special_tokens=True))