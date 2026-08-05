import torch
import time
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers import AttentionInterface, AttentionMaskInterface
from collections import defaultdict

# Calculates the total size of passed tensors in kilobytes.
def get_transfer_size_kb(*tensors):
    return sum(t.nelement() * t.element_size() for t in tensors if t is not None) / 1024

# This function builds the draft tree
# TALON uses a fixed global node budget rather than a fixed width/depth config.
def build_draft_tree(
    prefix_input_ids,
    ssm,
    budget,
    device,
    draft_cache,
    mu=0.03,
    max_k=16,
):
    if budget < 5:
        raise ValueError("budget must be at least 5 for TALON robust tree initialization")
    if not 0.0 < mu <= 1.0:
        raise ValueError("mu must be in the interval (0, 1]")
    if max_k < 1:
        raise ValueError("max_k must be at least 1")

    prompt_len = prefix_input_ids.shape[1]

    token_tree = prefix_input_ids[0].tolist()
    parent_array = [-1] + list(range(prompt_len - 1)) # -1 is the root and all other tokens follow a chain

    # Preserve the existing public probability array as local token probabilities.
    probs_array = [1.0] * prompt_len # Prompts have a probability of one

    # TALON gates on cumulative path probability, so keep that score separately.
    path_probs_array = [1.0] * prompt_len

    frontier_cache = draft_cache # Pre-cached tokens up until the last one
    frontier_tokens = prefix_input_ids[:, -1:] # Final token awaiting to be processed

    active_nodes = [prompt_len - 1] # Current active nodes we work on

    ssm_dists_by_parent = {} # Not used for now but stores distributions

    depth = 0
    while (len(token_tree) - prompt_len) < budget:
        with torch.inference_mode():
            outputs = ssm(
                input_ids=frontier_tokens,
                past_key_values=frontier_cache,
                use_cache=True,
            )

        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits.float(), dim=-1)

        next_active_nodes = []
        cache_mapping = []
        finalized_token_ids = []

        # for row, parent_idx in enumerate(active_nodes):
        #     ssm_dists_by_parent[parent_idx] = next_token_probs[row].detach()

        if depth == 0:
            # Robust Tree Initialization: always seed the root with five branches.
            if next_token_probs.shape[-1] < 5:
                raise ValueError("draft model vocabulary must contain at least 5 tokens")

            top_k_probs, top_k_indices = torch.topk(
                next_token_probs,
                k=5,
                dim=-1,
            )

            root_parent_idx = active_nodes[0]
            for prob, token_id in zip(top_k_probs[0], top_k_indices[0]):
                token_val = token_id.item()
                prob_val = prob.item()

                token_tree.append(token_val)
                parent_array.append(root_parent_idx)
                probs_array.append(prob_val)
                path_probs_array.append(prob_val)

                new_node_idx = len(token_tree) - 1
                next_active_nodes.append(new_node_idx)
                finalized_token_ids.append(token_val)

                # Every initialized branch comes from frontier row 0.
                cache_mapping.append(0)
        else:
            candidate_k = min(max_k, next_token_probs.shape[-1])
            top_k_probs, top_k_indices = torch.topk(
                next_token_probs,
                k=candidate_k,
                dim=-1,
            )

            # Each tuple stores:
            # (cumulative path probability, token id, parent node, frontier row,
            #  local token probability).
            candidate_pool = []
            for frontier_row_idx, parent_idx in enumerate(active_nodes):
                parent_path_prob = path_probs_array[parent_idx]
                for prob, token_id in zip(
                    top_k_probs[frontier_row_idx],
                    top_k_indices[frontier_row_idx],
                ):
                    token_prob = prob.item()
                    path_prob = parent_path_prob * token_prob
                    candidate_pool.append(
                        (
                            path_prob,
                            token_id.item(),
                            parent_idx,
                            frontier_row_idx,
                            token_prob,
                        )
                    )

            anchor = max(candidate[0] for candidate in candidate_pool)
            threshold = mu * anchor
            surviving_candidates = [
                candidate
                for candidate in candidate_pool
                if candidate[0] >= threshold
            ]

            # Enforce the exact global node budget before modifying the tree.
            remaining_slots = budget - (len(token_tree) - prompt_len)
            if len(surviving_candidates) > remaining_slots:
                surviving_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                surviving_candidates = surviving_candidates[:remaining_slots]

            for (
                path_prob,
                token_val,
                parent_idx,
                frontier_row_idx,
                token_prob,
            ) in surviving_candidates:
                token_tree.append(token_val)
                parent_array.append(parent_idx)
                probs_array.append(token_prob)
                path_probs_array.append(path_prob)

                new_node_idx = len(token_tree) - 1
                next_active_nodes.append(new_node_idx)
                finalized_token_ids.append(token_val)
                cache_mapping.append(frontier_row_idx)

        # Route each selected child to the exact parent history that produced it.
        cache_mapping_tensor = torch.tensor(
            cache_mapping,
            dtype=torch.long,
            device=device,
        )
        frontier_cache = DynamicCache([
            (
                layer.keys.index_select(0, cache_mapping_tensor),
                layer.values.index_select(0, cache_mapping_tensor),
            )
            for layer in outputs.past_key_values.layers
        ])

        frontier_tokens = torch.tensor(
            finalized_token_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        active_nodes = next_active_nodes
        depth += 1

    return token_tree, parent_array, probs_array, prompt_len, ssm_dists_by_parent


# Attention mask for the tree, since we cannot use normal causal attention
def build_full_tree_attention_mask(parent_array, prompt_len, past_len, device, dtype=torch.float16):
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
        
        if target_next == tokenizer.eos_token_id:
            break

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
    budget,
    device,
    dtype,
    max_accept_tokens,
    draft_cache,
    target_cache,
    mu=0.03,
    max_k=16,
    debug=False,
):
    # 1. Build draft tree from the current full sequence
    token_tree, parent_array, probs_array, prompt_length, ssm_dists_by_parent = build_draft_tree(
        prefix_input_ids=current_input_ids,
        ssm=ssm,
        budget=budget,
        mu=mu,
        max_k=max_k,
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

    transfer_kb = get_transfer_size_kb(packed_input_ids, attention_mask, position_ids)

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
    return accepted_tokens, accepted_nodes, total_tree_tokens, updated_target_cache, transfer_kb

# Main function that generates the tree 
def dynamic_tree_speculative_generate_greedy(
    prompt,
    tokenizer,
    ssm,
    target_model,
    budget,
    max_new_tokens,
    device,
    mu=0.03,
    max_k=16,
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
        # "accepted_lengths_per_step": [],
        "total_iterations": 0,
        "total_transfer_kb": 0.0
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

        # A path can contain at most one node per budget slot.
        # +1 allows the final target fallback/bonus token after the deepest accepted node.
        max_accept_this_iter = min(budget + 1, remaining_tokens)

        if debug:
            print("\n" + "=" * 80)
            print(f"ITERATION {iteration}")
            print("Current text:", repr(tokenizer.decode(generated_ids[0])))
            print("Remaining tokens:", remaining_tokens)
            print("Max accept this iteration:", max_accept_this_iter)

        accepted_tokens, accepted_nodes, total_tree_tokens, target_cache, transfer_kb = greedy_step(
            current_input_ids=generated_ids,
            ssm=ssm,
            target_model=target_model,
            tokenizer=tokenizer,
            budget=budget,
            mu=mu,
            max_k=max_k,
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
        # metrics["accepted_lengths_per_step"].append(num_accepted_draft)
        metrics["total_transfer_kb"] += transfer_kb

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
    metrics["avg_transfer_kb_per_step"] = metrics["total_transfer_kb"] / iteration


    return {
        "generated_ids": generated_ids,
        "new_token_ids": all_new_tokens,
        "text": final_text,
        "new_text": tokenizer.decode(all_new_tokens, skip_special_tokens=True),
        "latency": end - start,
        "num_new_tokens": len(all_new_tokens),
    }, metrics, generated_ids

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
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=target_model.generation_config.eos_token_id,
        )
        
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    latency = end - start
    return latency, len(new_tokens), tokenizer.decode(output_ids, skip_special_tokens=True), output_ids

if __name__ == "__main__":
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
    budget = 15 # Exact number of speculative tree nodes per iteration
    mu = 0.03 # TALON confidence-gating threshold multiplier
    max_k = 16 # Maximum candidates considered per active parent after depth 0
    max_new_tokens = 100

    result, spec_result, spec_tensor = dynamic_tree_speculative_generate_greedy(
        prompt=prompt,
        tokenizer=tokenizer,
        ssm=ssm,
        target_model=target_model,
        budget=budget,
        mu=mu,
        max_k=max_k,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        debug=False,  
    )

    # Normal baseline
    normal_latency, normal_tokens, normal_text, normal_tensor = run_normal_baseline(
        prompt=prompt, tokenizer=tokenizer, target_model=target_model, max_new_tokens=max_new_tokens, device=device
    )

    # --- Print Final Metrics Report ---
    print("\n" + "=" * 80)
    print("PERFORMANCE & METRICS REPORT")
    print("=" * 80)
    # print("Generated new token ids:", result["new_token_ids"])
    print("Full text:", repr(result["text"]))
    print("=" * 80)
    print("Normal baseline text: ", normal_text )
    print(f"Total Tokens Generated:         {spec_result['num_new_tokens']}")
    print(f"Total Spec Steps (Iterations):  {spec_result['total_iterations']}")
    print(f"Total Draft Tokens Accepted:    {spec_result['total_draft_tokens_accepted']}")
    print(f"Total Target Bonus Tokens:      {spec_result['total_bonus_tokens']}")
    print(f"Total Tree Tokens Rejected:     {spec_result['total_draft_tokens_evaluated'] - spec_result['total_draft_tokens_accepted']}")
    print(f"Total KB transferred:            {spec_result["total_transfer_kb"]}")
    print(f"Average KB transferred:          {spec_result["avg_transfer_kb_per_step"]}")

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
    print(f"Same tensors:  {torch.equal(spec_tensor, normal_tensor)}")
    print("=" * 80)
