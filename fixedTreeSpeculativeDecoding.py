import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AttentionInterface, AttentionMaskInterface
torch.set_printoptions(profile="full")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load draft model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
ssm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
dtype = torch.float16


ssm.eval()
# Configuration
prompt = "There was a little girl named Alice "
k_config = [2, 2, 2, 2] # Number of tokens in each branch

# Tokenize 
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

token_tree = input_ids[0].tolist()
global_parent_array = [-1] + list(range(prompt_length - 1))
tree_parent_array = [-1]
probs_array = [1.0] * prompt_length

# Active nodes
active_nodes = [(input_ids, prompt_length - 1)]

# Tree generation
for k in k_config:
    next_active_nodes = []

    for seq_tensor, parent_idx in active_nodes:
        # Predictions from ssm
        with torch.no_grad():
            outputs = ssm(seq_tensor)
        
        # Prediction for the next word
        next_token_logits = outputs.logits[0, -1, :]
        
        # Probabilities
        next_token_probs = F.softmax(next_token_logits, dim = -1)

        # Top-k probs and their token ids
        top_k_probs, top_k_indices = torch.topk(next_token_probs, k)

        # Branch out for each top-k token
        for prob, token_id in zip(top_k_probs, top_k_indices):
            token_val = token_id.item()
            prob_val = prob.item()

            # Update data structures
            token_tree.append(token_val)
            parent_array.append(parent_idx)
            tree_parrent_array.append(parent_idx)
            probs_array.append(prob_val)

            new_node_idx = len(token_tree) - 1
            new_token = seq_tensor.new_tensor([[token_val]])
            new_seq_tensor = torch.cat([seq_tensor, new_token], dim=1)

            # Queue the branch for next depth expansion
            next_active_nodes.append((new_seq_tensor, new_node_idx))
    
    # Overwrite the old nodes
    active_nodes = next_active_nodes


for x, y, z in zip(token_tree, probs_array, parent_array):
    print(f"{z} {tokenizer.decode(x)} {y}")


def build_topology_mask(parent_array, device, dtype=torch.float32):
    """Build 2D tree mask where each token attends to itself and ancestors."""
    seq_len = len(parent_array)
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    for i in range(seq_len):
        current_node = i
        mask[i, i] = 0.0
        while current_node != -1:
            parent_node = parent_array[current_node]
            if parent_node != -1:
                mask[i, parent_node] = 0.0
            current_node = parent_node
    return mask

def build_tree_attention_mask(prefix_len, parent_array, device, dtype=torch.float32):
    """
    Build the full 4D mask for a forward pass over (prefix + tree tokens).
    Shape: (1, 1, total_seq_len, total_seq_len)
    
    Prefix attends causally to itself.
    Tree tokens attend to all prefix tokens and their ancestors in the tree.
    """
    tree_len = len(parent_array)
    total_len = prefix_len + tree_len

    # Start fully masked
    mask_2d = torch.full((total_len, total_len), float('-inf'), device=device, dtype=dtype)

    # 1. Prefix: standard causal self-attention
    if prefix_len > 0:
        causal = torch.tril(torch.ones(prefix_len, prefix_len, device=device, dtype=dtype))
        mask_2d[:prefix_len, :prefix_len] = torch.where(causal == 1, 0.0, float('-inf'))

    # 2. Tree tokens attend to the entire accepted prefix
    if prefix_len > 0 and tree_len > 0:
        mask_2d[prefix_len:, :prefix_len] = 0.0

    # 3. Tree topology (local indices 0..tree_len-1 shifted by prefix_len)
    tree_mask = build_topology_mask(parent_array, device, dtype)
    mask_2d[prefix_len:, prefix_len:] = tree_mask

    # 4. Expand to 4D so Transformers treats it as a custom mask and skips auto-generation
    #    (batch_size, 1, query_length, kv_length)
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)
    return mask_4d


attention_mask = build_tree_attention_mask(
    prefix_len = prompt_length,
    parent_array = tree_parent_array,
    device = device,
    dtype = dtype
)

target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype = dtype,
    attn_implementation="spda"
).to(device)
target_model.eval()


with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

