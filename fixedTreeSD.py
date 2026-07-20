import torch
import time
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AttentionInterface, AttentionMaskInterface
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load draft model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
ssm = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(device)
dtype = torch.float16
target_model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2-large",
    torch_dtype = dtype,
    attn_implementation="sdpa"
).to(device)



target_model.eval()
ssm.eval()
# Configuration
prompt = "Once upon a time there was a little girl named Alice "
k_config = [1, 2, 2] # Number of tokens in each branch

# This function builds the draft tree
def build_draft_tree(prefix_input_ids, ssm, k_config, device):
    prompt_len = prefix_input_ids.shape[1]

    token_tree = prefix_input_ids[0].tolist()
    parent_array = [-1] + list(range(prompt_len - 1)) # -1 is the root and all other tokens follow a chain

    # For prompt tokens
    probs_array = [1.0] * prompt_len # Prompts have a probability of one

    active_nodes = [(prefix_input_ids, prompt_len - 1)] # Current active nodes we work on

    ssm_dists_by_parent = {} # Not used for now but stores distributions

    for k in k_config:
        next_active_nodes = [] # Storing tokens for next iteration

        for seq_tensor, parent_idx in active_nodes:
            with torch.no_grad():
                outputs = ssm(seq_tensor) # Forward pass
            
            next_token_logits = outputs.logits[0, -1, :] # Extract last logits
            next_token_probs = F.softmax(next_token_logits.float(), dim = -1) # Extract probabiltities

            ssm_dists_by_parent[parent_idx] = next_token_probs.detach() # Store distribution

            top_k_probs, top_k_indices = torch.topk(next_token_probs, k) # Section top k tokens' indicies and probabiltities

            for prob, token_id in zip(top_k_probs, top_k_indices):
                token_val = token_id.item()
                prob_val = prob.item()

                token_tree.append(token_val) # Contains all the tokens
                parent_array.append(parent_idx) # Parent for attention mask later
                probs_array.append(prob_val) # Probability array

                new_node_idx = len(token_tree) - 1 # Next node we are processing is at the end of the token tree
                new_token = seq_tensor.new_tensor([[token_val]])
                new_seq_tensor = torch.cat([seq_tensor, new_token], dim = 1) # combine new token and the tensor

                next_active_nodes.append((new_seq_tensor, new_node_idx))

        active_nodes = next_active_nodes # Replace active nodes
    
    return token_tree, parent_array, probs_array, prompt_len, ssm_dists_by_parent

# Attention mask for the tree, since we cannot use normal causal attention
def build_full_tree_attention_mask(parent_array, prompt_len, device, dtype=dtype):
    total_len = len(parent_array) 

    mask = torch.full(
        (total_len, total_len),
        float('-inf'),
        device=device,
        dtype=dtype
    ) # Create a tensor with -inf values

    causal = torch.tril(
        torch.ones(prompt_len, prompt_len, device=device, dtype=torch.bool)
    ) # Create a lower triangular matrix for the prompt, similar to causal attention

    mask[:prompt_len, :prompt_len] = torch.where(
        causal,
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(float('-inf'), device=device, dtype=dtype)
    ) # Replace all the ones with 0.0 and all the 0's with -infs in the causal mask

    for i in range(prompt_len, total_len):
        mask[i, :prompt_len] = 0.0 # Allow current index to look back at the original prompt
        cur = i
        while cur >= prompt_len:
            mask[i, cur] = 0.0
            cur = parent_array[cur] # keep going back using the parent array
    
    return mask.unsqueeze(0).unsqueeze(0)

# Depth calculation function
def build_tree_position_ids(parent_array, prompt_len, device):
    position_ids = [0] * len(parent_array)

    # Position of the prompt it's 0, 1, 2, ..... n
    for i in range(prompt_len):
        position_ids[i] = i
    
    # Next for the generated token tree, we have to figure out the parent's position and based on that add 1 to it. 
    for i in range(prompt_len, len(parent_array)):
        parent = parent_array[i]
        position_ids[i] = position_ids[parent] + 1
    
    return torch.tensor([position_ids], device = device, dtype=torch.long)

# To figure out the child nodes of a node
def build_children(parent_array, prompt_len):
    children = defaultdict(list)
    for i in range(prompt_len, len(parent_array)):
        children[parent_array[i]].append(i) # Add children onto the dictionary for O(1) lookup
    
    return children

# Greedy Decoding
def greedy_verify_tree(logits, token_tree, parent_array, prompt_len, tokenizer, max_accept_tokens, debug=True):
    children = build_children(parent_array, prompt_len)

    accepted_token_ids = [] # To store accepted tokens
    accepted_node_ids = [] # To store their ids

    cur_parent = prompt_len - 1 # start at the end

    while len(accepted_token_ids) < max_accept_tokens:
        target_next = torch.argmax(logits[0, cur_parent, :]).item() # Highest probability acc to parent

        matching_child = None
        for child_idx in children.get(cur_parent, []):
            if token_tree[child_idx] == target_next: # We only go further if there is a match
                matching_child = child_idx
                break

        if matching_child is None: # if no match add target_next token
            accepted_token_ids.append(target_next)

            if debug: # Print if debug enabled
                print("\nMISMATCH")
                print("Parent:", repr(tokenizer.decode([token_tree[cur_parent]])))
                print("Target wanted:", repr(tokenizer.decode([target_next])))

            break

        accepted_token_ids.append(target_next) # Add next nodes
        accepted_node_ids.append(matching_child)

        if debug:
            print("\nACCEPT")
            print("Token:", repr(tokenizer.decode([target_next])))
            print("Node:", matching_child)

        cur_parent = matching_child

    return accepted_token_ids, accepted_node_ids



# Greedy Iteration
def greedy_step(
    current_input_ids,
    ssm,
    target_model,
    tokenizer,
    k_config,
    device,
    dtype,
    max_accept_tokens,
    debug=False,
):
    # 1. Build draft tree from the current full sequence
    token_tree, parent_array, probs_array, prompt_length, ssm_dists_by_parent = build_draft_tree(
        prefix_input_ids=current_input_ids,
        ssm=ssm,
        k_config=k_config,
        device=device,
    )

    # 2. Pack prompt + speculative tree into one input
    packed_input_ids = torch.tensor(
        [token_tree],
        device=device,
        dtype=torch.long,
    )

    # 3. Build topology-aware attention mask
    attention_mask = build_full_tree_attention_mask(
        parent_array=parent_array,
        prompt_len=prompt_length,
        device=device,
        dtype=dtype,
    )

    # 4. Build position ids
    position_ids = build_tree_position_ids(
        parent_array=parent_array,
        prompt_len=prompt_length,
        device=device,
    )

    # 5. Run target model once over the packed tree
    with torch.inference_mode():
        outputs = target_model(
            input_ids=packed_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

    logits = outputs.logits

    # 6. Verify the tree greedily
    accepted_tokens, accepted_nodes = greedy_verify_tree(
        logits=logits,
        token_tree=token_tree,
        parent_array=parent_array,
        prompt_len=prompt_length,
        tokenizer=tokenizer,
        max_accept_tokens=max_accept_tokens,
        debug=debug,
    )

    # Returning token_tree length to calculate total rejections
    total_tree_tokens = len(token_tree) - prompt_length
    return accepted_tokens, accepted_nodes, total_tree_tokens

# Main function that generates the tree 
def fixed_tree_speculative_generate_greedy(
    prompt,
    tokenizer,
    ssm,
    target_model,
    k_config,
    max_new_tokens,
    device,
    dtype=torch.float16,
    debug=False,
):

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = input_ids.clone()
    all_new_tokens = []

    iteration = 0
    # Metric tracking variables
    metrics = {
        "total_draft_tokens_evaluated": 0,
        "total_draft_tokens_accepted": 0,
        "total_bonus_tokens": 0,
        "accepted_lengths_per_step": [],
        "total_iterations": 0
    }

    if device == "cuda": # For latency
        torch.cuda.synchronize()
    start = time.perf_counter()

    while len(all_new_tokens) < max_new_tokens:
        iteration += 1

        remaining_tokens = max_new_tokens - len(all_new_tokens)

        # len(k_config) is tree depth.
        # +1 allows the final target fallback/bonus token after the deepest accepted node.
        max_accept_this_iter = min(len(k_config) + 1, remaining_tokens)

        if debug:
            print("\n" + "=" * 80)
            print(f"ITERATION {iteration}")
            print("Current text:", repr(tokenizer.decode(generated_ids[0])))
            print("Remaining tokens:", remaining_tokens)
            print("Max accept this iteration:", max_accept_this_iter)

        accepted_tokens, accepted_nodes, total_tree_tokens = greedy_step(
            current_input_ids=generated_ids,
            ssm=ssm,
            target_model=target_model,
            tokenizer=tokenizer,
            k_config=k_config,
            device=device,
            dtype=dtype,
            max_accept_tokens=max_accept_this_iter,
            debug=debug,
        )

        # Safety check
        if len(accepted_tokens) == 0:
            print("No tokens accepted/generated. Stopping to avoid infinite loop.")
            break

        # Clip just in case
        accepted_tokens = accepted_tokens[:remaining_tokens]

        # Track metric details
        num_accepted_draft = len(accepted_nodes)
        num_bonus = len(accepted_tokens) - num_accepted_draft
        
        metrics["total_draft_tokens_evaluated"] += total_tree_tokens
        metrics["total_draft_tokens_accepted"] += num_accepted_draft
        metrics["total_bonus_tokens"] += num_bonus
        metrics["accepted_lengths_per_step"].append(num_accepted_draft)

        # Append accepted tokens to sequence
        new_token_tensor = torch.tensor(
            [accepted_tokens],
            device=device,
            dtype=torch.long,
        )

        generated_ids = torch.cat([generated_ids, new_token_tensor], dim=1)
        all_new_tokens.extend(accepted_tokens)

        if debug:
            print("\nAccepted token ids:", accepted_tokens)
            print("Accepted node ids:", accepted_nodes)
            print("Accepted text:", repr(tokenizer.decode(accepted_tokens)))
            print("Total generated so far:", len(all_new_tokens))

        # Stop if EOS appears
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in accepted_tokens:
            if debug:
                print("EOS token generated. Stopping.")
            break

    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    metrics["total_iterations"] = iteration
    metrics["latency"] = end - start
    metrics["num_new_tokens"] = len(all_new_tokens)
    metrics["text"] = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    metrics["new_text"] = tokenizer.decode(all_new_tokens, skip_special_tokens=True)


    return {
        "generated_ids": generated_ids,
        "new_token_ids": all_new_tokens,
        "text": final_text,
        "new_text": tokenizer.decode(all_new_tokens, skip_special_tokens=True),
        "latency": end - start,
        "num_new_tokens": len(all_new_tokens),
    }, metrics

# Comparison against normal generation
def run_normal_baseline(prompt, tokenizer, target_model, max_new_tokens, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        output_ids = target_model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, # Greedy matching baseline
            use_cache=True
        )
        
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    latency = end - start
    return latency, len(new_tokens), tokenizer.decode(output_ids, skip_special_tokens=True)

max_new_tokens = 500

result, spec_result = fixed_tree_speculative_generate_greedy(
    prompt=prompt,
    tokenizer=tokenizer,
    ssm=ssm,
    target_model=target_model,
    k_config=k_config,
    max_new_tokens=max_new_tokens,
    device=device,
    dtype=dtype,
    debug=True,  
)

# Normal baseline
normal_latency, normal_tokens, normal_text = run_normal_baseline(
    prompt=prompt, tokenizer=tokenizer, target_model=target_model, max_new_tokens=max_new_tokens, device=device
)

# --- Print Final Metrics Report ---
print("\n" + "=" * 80)
print("PERFORMANCE & METRICS REPORT")
print("=" * 80)
print("Generated new token ids:", result["new_token_ids"])
print("Full text:", repr(result["text"]))
print("=" * 80)
print("Normal baseline text: ", normal_text )
print(f"Total Tokens Generated:         {spec_result['num_new_tokens']}")
print(f"Total Spec Steps (Iterations):  {spec_result['total_iterations']}")
print(f"Total Draft Tokens Accepted:    {spec_result['total_draft_tokens_accepted']}")
print(f"Total Target Bonus Tokens:      {spec_result['total_bonus_tokens']}")
print(f"Total Tree Tokens Rejected:     {spec_result['total_draft_tokens_evaluated'] - spec_result['total_draft_tokens_accepted']}")

# Acceptance Rate Interpretations
path_acceptance_rate = (spec_result['total_draft_tokens_accepted'] / (spec_result['total_iterations'] * len(k_config))) * 100
tree_acceptance_rate = (spec_result['total_draft_tokens_accepted'] / spec_result['total_draft_tokens_evaluated']) * 100

print(f"Draft Acceptance Rate (Path):   {path_acceptance_rate:.2f}% (Accepted vs max potential path depth)")
print(f"Draft Acceptance Rate (Tree):   {tree_acceptance_rate:.2f}% (Accepted vs total structural tree nodes generated)")
print(f"Average Accepted Per Step:      {sum(spec_result['accepted_lengths_per_step']) / spec_result['total_iterations']:.2f} tokens")

print("-" * 80)
print("SPEED & LATENCY COMPARISON")
print("-" * 80)
spec_throughput = spec_result['num_new_tokens'] / spec_result['latency']
normal_throughput = normal_tokens / normal_latency

print(f"Speculative Tree Latency:       {spec_result['latency']:.4f} seconds")
print(f"Speculative Tree Throughput:    {spec_throughput:.2f} tokens/sec")
print(f"Normal Target Model Latency:    {normal_latency:.4f} seconds")
print(f"Normal Target Model Throughput: {normal_throughput:.2f} tokens/sec")
print(f"Speedup Factor:                 {normal_latency / spec_result['latency']:.2f}x (Values < 1.0x mean slower)")
print("=" * 80)
