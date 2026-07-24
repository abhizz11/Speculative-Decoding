import torch
import time
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers import AttentionInterface, AttentionMaskInterface
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load draft model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
ssm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
dtype = torch.float16
target_model = AutoModelForCausalLM.from_pretrained(
   "meta-llama/Llama-3.2-3B",
    torch_dtype = dtype,
    attn_implementation="sdpa"
).to(device)



target_model.eval()
ssm.eval()

# Configuration
prompt = "Once upon a time there was a little girl named Alice "
k_config = [3, 2, 1] # Number of tokens in each branch

# This function builds the draft tree
def build_draft_tree(prefix_input_ids, ssm, k_config, device, draft_cache):
    prompt_len = prefix_input_ids.shape[1]

    token_tree = prefix_input_ids[0].tolist()
    parent_array = [-1] + list(range(prompt_len - 1)) # -1 is the root and all other tokens follow a chain

    # For prompt tokens
    probs_array = [1.0] * prompt_len # Prompts have a probability of one
    frontier_cache = draft_cache # Pre-cached tokens up until the last one
    frontier_tokens = prefix_input_ids[:, -1:] # Final token awaiting to be processed


    active_nodes = [prompt_len - 1] # Current active nodes we work on

    ssm_dists_by_parent = {} # Not used for now but stores distributions

    for depth, k in enumerate(k_config):
        with torch.inference_mode():
            outputs = ssm(
                input_ids = frontier_tokens,
                past_key_values = frontier_cache,
                use_cache = True,
            )
        
        next_token_logits = outputs.logits[:, -1, :] # The next token is in the last row
        next_token_probs = F.softmax(next_token_logits.float(), dim=-1) # Probabilities for later
        top_k_probs, top_k_indices = torch.topk(next_token_probs, k, dim=-1) # Extract top k tokens acc to the config

        next_active_nodes = []
        for row, parent_idx in enumerate(active_nodes):
            ssm_dists_by_parent[parent_idx] = next_token_probs[row].detach()

            for prob, token_id in zip(top_k_probs[row], top_k_indices[row]):
                token_val = token_id.item()
                prob_val = prob.item()

                token_tree.append(token_val) # Contains all the tokens
                parent_array.append(parent_idx) # Parent for attention mask

                probs_array.append(prob_val) # Probability array
                new_node_idx = len(token_tree) - 1

                next_active_nodes.append(new_node_idx)
        
        # Each branch needs its own copy of the KV state before the next depth. 
        if depth < len(k_config) - 1:
            frontier_cache = DynamicCache([
                (
                    layer.keys.repeat_interleave(k, dim=0),
                    layer.values.repeat_interleave(k, dim=0),
                )
                for layer in outputs.past_key_values.layers
            ])
            frontier_tokens = top_k_indices.reshape(-1, 1)
            active_nodes = next_active_nodes
    
    return token_tree, parent_array, probs_array, prompt_len, ssm_dists_by_parent


# Attention mask for the tree, since we cannot use normal causal attention
def build_full_tree_attention_mask(parent_array, prompt_len, past_len, device, dtype=dtype):
    tree_len = len(parent_array) - prompt_len
    query_len = 1 + tree_len # pending token + flattened draft tree


    # dim0 = batch size which is 1
    # dim1 = 1 allows Pytorch to broadcast this mask across all heads
    # dim2 = queries, represents tokens actively passing through the model rn
    # dim3 = keys, represents total memory space, older kv cache + curr tokens

    mask = torch.full(
        (1, 1, query_len, past_len + query_len),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype
    )

    # Every query can see the already-cached comitted prefix
    if past_len > 0:
        # [All batches, all heads, all queries, keys from 0 to past_len]
        mask[:, :, :, :past_len] = 0.0 # allow to attend all previously commited tokens in KV cache
    
    # The pending token can see itself.
    mask[0, 0, 0, past_len] = 0.0

    # Every tree node can see the pending token and only its own ancestor path.
    for node_idx in range(prompt_len, len(parent_array)):
        query_idx = 1 + (node_idx - prompt_len)
        mask[0, 0, query_idx, past_len] = 0.0

        cur = node_idx
        while cur >= prompt_len:
            tree_query_idx = 1 + (cur - prompt_len)
            mask[0, 0, query_idx, past_len + tree_query_idx] = 0.0
            cur = parent_array[cur]
        
    return mask

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
def greedy_verify_tree(logits, token_tree, parent_array, prompt_len, tokenizer, max_accept_tokens, logit_offset = 0, debug=True):
    children = build_children(parent_array, prompt_len)

    accepted_token_ids = [] # To store accepted tokens
    accepted_node_ids = [] # To store their ids

    cur_parent = prompt_len - 1 # start at the end

    while len(accepted_token_ids) < max_accept_tokens:
        target_next = torch.argmax(logits[0, cur_parent - logit_offset, :]).item() # Highest probability acc to parent

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
    draft_cache,
    target_cache,
    debug=False,
):
    # 1. Build draft tree from the current full sequence
    token_tree, parent_array, probs_array, prompt_length, ssm_dists_by_parent = build_draft_tree(
        prefix_input_ids=current_input_ids,
        ssm=ssm,
        k_config=k_config,
        device=device,
        draft_cache=draft_cache,
    )
    # 2. The target cache contains all commited tokens except the current final
    # pending token. Verify pending token + the complete speculative tree
    past_len = prompt_length - 1
    packed_input_ids = torch.tensor(
        [token_tree[prompt_length - 1:]],
        device=device,
        dtype=torch.long,
    )

    # 3. Build topology-aware attention mask
    attention_mask = build_full_tree_attention_mask(
        parent_array=parent_array,
        prompt_len=prompt_length,
        past_len=past_len,
        device=device,
        dtype=dtype,
    )

    # 4. Reuse existing tree position calculation, keep only pending token and tree positions
    # that are part of this cached forward
    position_ids = build_tree_position_ids(
        parent_array=parent_array,
        prompt_len=prompt_length,
        device=device,
    )[:, prompt_length - 1:]

    # 5. Run target model once over the pending token + packed tree
    with torch.inference_mode():
        outputs = target_model(
            input_ids=packed_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=target_cache,
            use_cache=True,
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
        logit_offset=prompt_length - 1,
        debug=debug,
    )

    # 7. Keep the old cached prefix, the pending token, and only the accepted
    # tree path. Every rejected speculative branch is discarded.
    cache_indices = torch.tensor(
        list(range(past_len))
        + [past_len]
        + [
            past_len + 1 + (node_idx - prompt_length)
            for node_idx in accepted_nodes
        ],
        dtype=torch.long,
        device=device,
    )

    updated_target_cache = DynamicCache([
        (
            layer.keys.index_select(2, cache_indices),
            layer.values.index_select(2, cache_indices),
        )
        for layer in outputs.past_key_values.layers
    ])

    # Returning token_tree length to calculate total rejections
    total_tree_tokens = len(token_tree) - prompt_length
    return accepted_tokens, accepted_nodes, total_tree_tokens, updated_target_cache


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

    # Maintain independent caches for the draft and target
    target_cache = DynamicCache(config=target_model.config)
    draft_cache = DynamicCache(config=ssm.config)

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

    # Prefill both caches once with every prompt token except the final pending
    # token. This forward pass is part of speculative-generation latency.
    with torch.inference_mode():
        if input_ids.shape[1] > 1:
            target_model(
                input_ids=input_ids[:, :-1],
                past_key_values=target_cache,
                use_cache=True,
            )
            ssm(
                input_ids=input_ids[:, :-1],
                past_key_values=draft_cache,
                use_cache=True,
            )

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

        accepted_tokens, accepted_nodes, total_tree_tokens, target_cache = greedy_step(
            current_input_ids=generated_ids,
            ssm=ssm,
            target_model=target_model,
            tokenizer=tokenizer,
            k_config=k_config,
            device=device,
            dtype=dtype,
            max_accept_tokens=max_accept_this_iter,
            draft_cache=draft_cache,
            target_cache=target_cache,
            debug=debug,
        )

        # Safety check
        if len(accepted_tokens) == 0:
            print("No tokens accepted/generated. Stopping to avoid infinite loop.")
            break

        # Clip just in case
        accepted_tokens = accepted_tokens[:remaining_tokens]
        accepted_nodes = accepted_nodes[:min(len(accepted_nodes), len(accepted_tokens))]

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
        
        if len(all_new_tokens) >= max_new_tokens:
            break

        # The first draft forward already cached the old pending token. Add only
        # target-accepted draft tokens; the fallback/bonus token remains pending
        # for the next iteration.
        if accepted_nodes:
            accepted_draft_ids = torch.tensor(
                [accepted_tokens[:len(accepted_nodes)]],
                device=device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                ssm(
                    input_ids=accepted_draft_ids,
                    past_key_values=draft_cache,
                    use_cache=True,
                )

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
