import torch
import zmq
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct").to(device)
model.eval()

# Setup ZeroMQ Server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")

print("Target Model Server Listening on port 5555...")

while True:
    # Receive data from Jetson
    payload = socket.recv_pyobj()
    generated = payload['generated'].to(device)
    draft_tokens = payload['draft_tokens'].to(device)
    q_distributions = [q.to(device) for q in payload['q_distributions']]
    
    old_len = generated.shape[1]
    candidate_ids = torch.cat([generated, draft_tokens.unsqueeze(0)], dim=1)

    # Run Target Model
    with torch.no_grad():
        target_outputs = model(input_ids=candidate_ids)

    accepted_list = []
    rejected = False
    reject_index = None

    for i, token_id in enumerate(draft_tokens):
        q_dist = q_distributions[i]
        q_prob = q_dist[token_id].item()

        target_logits = target_outputs.logits[0, old_len + i - 1, :]
        p_dist = torch.softmax(target_logits, dim=-1)
        p_prob = p_dist[token_id].item()

        accept_prob = min(1.0, p_prob / q_prob) if q_prob > 0 else 1.0
        
        if torch.rand(1).item() <= accept_prob:
            accepted_list.append(token_id.view(1, 1))
        else:
            rejected = True
            reject_index = i
            break

    # Handle Rejection/Next Token
    next_token = None
    if rejected:
        q_dist = q_distributions[reject_index]
        target_logits = target_outputs.logits[0, old_len + reject_index - 1, :]
        p_dist = torch.softmax(target_logits, dim=-1)
        adjusted_dist = torch.clamp(p_dist - q_dist, min=0)
        
        if adjusted_dist.sum().item() == 0:
            adjusted_dist = p_dist
        else:
            adjusted_dist = adjusted_dist / adjusted_dist.sum()
            
        next_token = torch.multinomial(adjusted_dist, num_samples=1).view(1, 1)
    else:
        next_logits = target_outputs.logits[0, -1, :]
        next_dist = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(next_dist, num_samples=1).view(1, 1)

    # Send results back to Jetson
    accepted_tensor = torch.cat(accepted_list, dim=1) if len(accepted_list) > 0 else torch.empty((1, 0), dtype=torch.long, device=device)
    
    response = {
        'accepted_tensor': accepted_tensor.cpu(),
        'next_token': next_token.cpu(),
        'rejected': rejected
    }
    socket.send_pyobj(response)