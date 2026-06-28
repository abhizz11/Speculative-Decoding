# Speculative Sampling based on Tree Sampling
# Took References: https://github.com/bassrehab/speculative-decoding/blob/main/tree_speculation.py 

import torch
import torch.nn.functional as F
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass, field 

@dataclass
class TreeNode:
    ''' Represents a node in the speculation tree'''
    token_id: int
    prob: float
    depth: int
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    position_in_sequence: int = 0

    def is_leaf(self):
        ''' Returns if current node is a leaf node'''
        return len(self.children) == 0
    
    def get_path_to_root(self):
        '''Get token IDs from root to current node'''
        path = []
        node = self
        while node:
            path.append(node.token_id)
            node = node.parent
        
        return list(reversed(path))

@dataclass
class TreeSpecConfig:
    '''Tree Speculation Configuration'''
    draft_model_name: str = "gpt2" # Draft model
    target_model_name: str = "gpt2-large" # Target model
    max_depth: int = 4 # gamma
    branch_factor: int = 2 # Guess top 2 most likely words
    max_candidates: int = 16 # Absolute maximum number of guesses allowed
    temperature: float = 0.9 # How random the AI is
    top_p: float = 0.9  # Adjust the choice based on probability 
    max_new_tokens: int = 100 # Max number of tokens to guess


@dataclass
class TreeMetrics:
    '''Tree speculation matrics'''
    total_tokens_generated: int = 0 
    total_candidates_evaluated: int = 0
    total_iterations: int = 0 # How many times target had to review a batch
    total_time_ms: float = 0.0
    tree_depths: List[int] = field(default_factory=list) # How many words target accepted on each turn


    @property
    def avg_accepted_per_iteration(self):
        return self.total_tokens_generated / max(self.total_iterations, 1) # To prevent division by 0 errors
    
    @property
    def tokens_per_second(self):
        return (self.total_tokens_generated / max(self.total_time_ms, 1)) * 1000 
    
    @property
    def avg_tree_depth(self):
        return sum(self.tree_depths) / max(len(self.tree_depths), 1)

class SpeculationTree:
    def __init__(self, config):
        self.config = config
        self.root = None
        self.nodes = []
        self.leaves = []
    
    def build_tree(self, draft_model, input_ids, past_key_values, device):
        '''
        Build speculation tree using draft model.
        Returns tuple of candidate tokens, candidate probs, and tree attention mask
        '''
        self.nodes = []
        self.leaves = []

        # Root node, last token in sequence
        self.root = TreeNode(
            token_id = input_ids[0,-1].item(),
            prob = 1.0,
            depth = 0,
            position_in_sequence = 0
        )

        self.nodes.append(self.root)

        # BFS
        current_level = [self.root]
        position = 1

        prefix_ids = input_ids[0, :-1].tolist()

        for depth in range(1, self.config.max_depth + 1):
            if len(self.nodes) >= self.config.max_candidates:
                break
            
            next_level = []
            
            for parent in current_level:
                if len(self.nodes) >= self.config.max_candidates:
                    break
                
                path_tokens = parent.get_path_to_root()
                full_sequence = prefix_ids + path_tokens
                path_tensor = torch.tensor([full_sequence], device=device)

                with torch.no_grad():
                    outputs = draft_model(path_tensor, use_cache=False)
                    logits = outputs.logits[0, -1, :]
                
                probs = F.softmax(logits / max(self.config.temperature, 1e-8), dim = -1)
                top_probs, top_indices = torch.topk(probs, self.config.branch_factor)

                # Creating child nodes
                for i in range(self.config.branch_factor):
                    if len(self.nodes) >= self.config.max_candidates:
                        break
                    
                    child = TreeNode(
                        token_id = top_indices[i].item(),
                        prob = top_probs[i].item(),
                        depth = depth,
                        parent = parent,
                        position_in_sequence = position
                    )

                    parent.children.append(child)
                    self.nodes.append(child)
                    next_level.append(child)
                    position += 1
            
            current_level = next_level
            if not current_level:
                break
            
            # Leaves
            self.leaves = [n for n in self.nodes if n.is_leaf()]

            # Flatten tree 
        return self.flatten_tree(device)           
    
    def flatten_tree(self, device):
        '''Flatten tree into tensors for batched verification, returns tokens, probs, attention_mask'''
        num_nodes = len(self.nodes)

        # Token ID of each node
        tokens = torch.tensor(
            [n.token_id for n in self.nodes],
            dtype = torch.long,
            device=device
        )

        # Probability of each node
        probs = torch.tensor(
            [n.prob for n in self.nodes],
            dtype = torch.float,
            device=device
        )

        # Attention mask, so that each node can only attend to tokens before itself
        attention_mask = torch.zeros(num_nodes, num_nodes, device=device)

        for i, node in enumerate(self.nodes):
            # Can attend itself
            attention_mask[i, i] = 1

            ancestor = node.parent
            while ancestor is not None:
                ancestor_idx = self.nodes.index(ancestor)
                attention_mask[i, ancestor_idx] = 1
                ancestor = ancestor.parent
            
        return tokens, probs, attention_mask

    def get_paths(self):
        """Returns a list of paths where each path is a lost of nodes from root to leaf"""
        paths = []
        for leaf in self.leaves:
            path = []
            node = leaf
            while node:
                path.append(node)
                node = node.parent
            
            paths.append(list(reversed(path)))
        
        return paths

@torch.no_grad()
def verify_tree(
    target_model, 
    input_ids, 
    tree, 
    candidate_tokens, 
    candidate_probs, 
    tree_attention_mask,
    past_key_values,
    temperature=1.0,
    device="cuda"):
    '''
    Verifies all tree paths with target model and select best accepted sequence
    Returns best path and best length
    '''
    paths = tree.get_paths()
    if not paths:
        return [], 0
        
    prefix_ids = input_ids[0, :-1].tolist()
    
    # Batch all valid root-to-leaf paths
    batch_input_ids = []
    for path in paths:
        path_tokens = [node.token_id for node in path]
        batch_input_ids.append(prefix_ids + path_tokens)
        
    # Pad sequences to max length in this batch for the standard forward pass
    max_len = max(len(seq) for seq in batch_input_ids)
    padded_batch = []
    attention_mask = []
    
    for seq in batch_input_ids:
        pad_len = max_len - len(seq)
        padded_batch.append(seq + [0] * pad_len) 
        attention_mask.append([1] * len(seq) + [0] * pad_len)
        
    batch_tensor = torch.tensor(padded_batch, device=device)
    mask_tensor = torch.tensor(attention_mask, device=device)
    
    outputs = target_model(batch_tensor, attention_mask=mask_tensor, use_cache=False)
    batch_logits = outputs.logits 
    
    best_path = []
    best_length = 0
    seq_len_prefix = len(prefix_ids)
    
    for path_idx, path in enumerate(paths):
        accepted = []
        path_logits = batch_logits[path_idx]
        
        for i, node in enumerate(path[1:], start=1):
            logit_idx = seq_len_prefix + i - 1
            
            target_probs_step = F.softmax(path_logits[logit_idx] / max(temperature, 1e-8), dim=-1)
            p_target = target_probs_step[node.token_id].item()
            p_draft = node.prob
            
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            # Coin Toss and Adjusted Probability Distribution
            if torch.rand(1).item() < accept_prob:
                accepted.append(node.token_id)
            else:
                adjusted = target_probs_step.clone()
                adjusted[node.token_id] = max(0, adjusted[node.token_id] - p_draft)
                adjusted = adjusted / (adjusted.sum() + 1e-10)
                
                # Normalization
                if adjusted.sum() > 1e-8:
                    new_token = torch.multinomial(adjusted, 1).item()
                else:
                    new_token = path_logits[logit_idx].argmax().item()
                
                accepted.append(new_token)
                break 
        
        # Update best path
        if len(accepted) > best_length:
            best_length = len(accepted)
            best_path = accepted
            
            # Bonus token if the entire branch was accepted
            if len(accepted) == len(path) - 1:
                last_logit_idx = seq_len_prefix + len(path) - 1
                bonus_probs = F.softmax(path_logits[last_logit_idx] / max(temperature, 1e-8), dim=-1)
                bonus_token = torch.multinomial(bonus_probs, 1).item()
                best_path.append(bonus_token)
                best_length += 1
                
    return best_path, best_length

@torch.no_grad()
def tree_speculative_decode(draft_model, target_model, tokenizer, prompt, config, device):
    '''
    Generate text using tree-based speculative decoding, returns generated text and metrics
    '''
    # Metrics object, gets updated in each iteration
    metrics = TreeMetrics()
    start_time = time.perf_counter()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    all_token_ids = input_ids.clone()

    generated_tokens = 0
    tree = SpeculationTree(config)

    while generated_tokens < config.max_new_tokens:
        metrics.total_iterations += 1

        # Speculation Tree
        candidate_tokens, candidate_probs, tree_mask = tree.build_tree(
            draft_model, all_token_ids, None, device
        )

        metrics.total_candidates_evaluated += len(candidate_tokens)

        # Verify with target
        accepted_tokens, num_accepted = verify_tree(
            target_model, all_token_ids, tree,
            candidate_tokens, candidate_probs, tree_mask,
            None, config.temperature, device
        )

        if not accepted_tokens:
            with torch.no_grad():
                outputs = target_model(all_token_ids, use_cache=False)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / config.temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
                accepted_tokens = [token]
            
        
        # Update sequence
        new_tokens = torch.tensor([accepted_tokens], device=device)
        all_token_ids = torch.cat([all_token_ids, new_tokens], dim=-1)
        generated_tokens += len(accepted_tokens)

        metrics.total_tokens_generated += len(accepted_tokens)
        metrics.tree_depths.append(len(accepted_tokens))

        if tokenizer.eos_token_id in accepted_tokens:
            break
        
        metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
        generated_text = tokenizer.decode(all_token_ids[0], skip_special_tokens=True)

    return generated_text, metrics


def normal_inference(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, attention_mask=inputs["attention_mask"],  pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True, do_sample=True, temperature=0.9, top_p=0.9)
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

config = TreeSpecConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
draft_model = AutoModelForCausalLM.from_pretrained(config.draft_model_name).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(config.target_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


target_model = AutoModelForCausalLM.from_pretrained(config.target_model_name).to(device).eval()
prompt = "Once upon a time there was a little girl named Alice "

print(f"Prompt: {prompt}")
print(f"Using device: {device}")
print(f"Draft model: {config.draft_model_name}")
print(f"Target model: {config.target_model_name}")
print(f"Tree depth: {config.max_depth}")
print(f"Branch factor: {config.branch_factor}")
print(f"Max candidates: {config.max_candidates}")
print("-" * 60)


start_normal = time.perf_counter()
normal = normal_inference(target_model, tokenizer, prompt, 100)
end_normal = time.perf_counter()

print("Normal latency: ", (end_normal - start_normal) * 1000)
print("Normal output: ", normal[0])
print("-" * 60)
print("\n")


text, metrics = tree_speculative_decode(
    draft_model, target_model, tokenizer, prompt, config, device
)

print("\n" + "=" * 60)
print("TREE SPECULATION RESULTS")
print("=" * 60)
print(f"Tokens generated: {metrics.total_tokens_generated}")
print(f"Total iterations: {metrics.total_iterations}")
print(f"Avg tokens/iteration: {metrics.avg_accepted_per_iteration:.2f}")
print(f"Avg tree depth accepted: {metrics.avg_tree_depth:.2f}")
print(f"Candidates evaluated: {metrics.total_candidates_evaluated}")
print(f"Time: {metrics.total_time_ms:.1f} ms")
print(f"Throughput: {metrics.tokens_per_second:.1f} tok/s")
print(text)