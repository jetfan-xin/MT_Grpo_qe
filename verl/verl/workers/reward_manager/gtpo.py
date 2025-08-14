# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("gtpo")
class GTPORewardManager:
    """
    GTPO (Group-Token Policy Optimization) Reward Manager.
    
    Implements token-level reward design using policy entropy weighting
    as described in "GTPO and GRPO-S: Token and Sequence-Level Reward 
    Shaping with Policy Entropy".
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        # GTPO-specific parameters
        entropy_beta=1.0,          # Entropy weighting factor (α in paper formula)
        **reward_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.reward_kwargs = reward_kwargs
        
        # GTPO parameters
        self.entropy_beta = entropy_beta

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def _compute_entropy_bonus(self, data, response_mask, base_rewards):
        """
        Compute entropy-based bonus for each token according to GTPO paper formula:
        r̃i,t = ri + α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
        
        CORRECTED: dt should be the number of successful sequences within EACH PROMPT GROUP,
        not across the entire batch. Each prompt generates n=16 responses, so dt ∈ [1,16].
        
        Args:
            data: DataProto containing token information
            response_mask: Attention mask for response tokens (bs, response_len)
            base_rewards: Base reward scores for each sequence (bs,) - used to filter successful sequences
            
        Returns:
            entropy_bonus: Bonus for each token (bs, response_len)
        """
        with torch.no_grad():
            # Get prompt UIDs to group sequences (if available)
            if "uid" not in data.non_tensor_batch:
                # Fallback: treat each sequence as its own group (dt=1 for successful sequences)
                uids = [f"val_seq_{i}" for i in range(len(base_rewards))]
                is_validation = True
            else:
                uids = data.non_tensor_batch["uid"]
                is_validation = False
            
            num_groups = len(set(uids))
            
            # Create successful sequence mask (reward > 0, as requested by user)
            successful_mask = base_rewards > 0  # (bs,)
            num_successful = successful_mask.sum()
            
            mode = "VAL" if is_validation else "TRAIN"
            print(f"🔍 GTPO [{mode}]: {len(base_rewards)} seqs, {num_groups} groups, {num_successful} successful")
            
            if num_successful == 0:
                return torch.zeros_like(response_mask, dtype=torch.float32)
            
            # Extract token entropy
            token_entropy = self._extract_token_entropy(data, response_mask)
            
            # Apply attention mask to entropy
            token_entropy = token_entropy * response_mask
            
            # Group sequences by UID
            uid_to_indices = defaultdict(list)
            for idx, uid in enumerate(uids):
                uid_to_indices[uid].append(idx)
            
            # Initialize entropy bonus
            bs, response_len = token_entropy.shape
            entropy_bonus = torch.zeros_like(token_entropy)
            
            # Process each prompt group independently
            group_stats = []
            for group_id, (uid, seq_indices) in enumerate(uid_to_indices.items()):
                # Extract group data
                group_base_rewards = base_rewards[seq_indices]
                group_successful_mask = successful_mask[seq_indices]
                group_token_entropy = token_entropy[seq_indices]  # (group_size, response_len)
                group_response_mask = response_mask[seq_indices]
                
                num_successful_in_group = group_successful_mask.sum().item()
                
                # Skip groups with no successful sequences
                if num_successful_in_group == 0:
                    continue
                
                # Apply GTPO formula for each time step within this group
                for t in range(response_len):
                    # Get entropy at time step t for this group
                    group_entropy_t = group_token_entropy[:, t]  # (group_size,)
                    group_active_mask_t = group_response_mask[:, t]  # (group_size,)
                    
                    # Only consider successful sequences in this group
                    group_successful_active_mask_t = group_successful_mask & group_active_mask_t.bool()
                    dt_group = group_successful_active_mask_t.sum()  # This is the correct dt (1-16)
                    
                    if dt_group > 0:
                        # Sum of entropies at time t across successful sequences in THIS GROUP
                        entropy_sum_group_t = (group_entropy_t * group_successful_active_mask_t.float()).sum()
                        
                        if entropy_sum_group_t > 1e-8:
                            # Apply GTPO formula within this group: α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
                            group_entropy_ratios = group_entropy_t / entropy_sum_group_t
                            group_entropy_bonus_t = (self.entropy_beta * 
                                                   group_entropy_ratios * 
                                                   dt_group * group_active_mask_t)
                            
                            # PAPER REQUIREMENT: Only assign to SUCCESSFUL sequences
                            # Unsuccessful sequences should have r̃i,t := 0 for all tokens
                            for local_idx, global_idx in enumerate(seq_indices):
                                if group_successful_mask[local_idx]:  # Only successful sequences get bonus
                                    entropy_bonus[global_idx, t] = group_entropy_bonus_t[local_idx]
                                # Unsuccessful sequences keep entropy_bonus[global_idx, t] = 0
                            
                        else:
                            # Uniform bonus within this group (only for successful sequences)
                            uniform_bonus = self.entropy_beta
                            for local_idx, global_idx in enumerate(seq_indices):
                                if group_successful_mask[local_idx]:  # Only successful sequences get bonus
                                    entropy_bonus[global_idx, t] = uniform_bonus * group_active_mask_t[local_idx]
                
                # Collect group statistics
                group_successful_indices = [seq_indices[i] for i in range(len(seq_indices)) if group_successful_mask[i]]
                if len(group_successful_indices) > 0:
                    group_bonus_sum = entropy_bonus[group_successful_indices].sum().item()
                    group_stats.append({
                        'group_id': group_id,
                        'uid': uid,
                        'total_seqs': len(seq_indices),
                        'successful_seqs': num_successful_in_group,
                        'max_dt': num_successful_in_group,
                        'bonus_sum': group_bonus_sum
                    })
            
            # Summary statistics
            dt_values = [stats['max_dt'] for stats in group_stats]
            if dt_values:
                max_dt = max(dt_values)
                avg_dt = sum(dt_values) / len(dt_values)
                print(f"📊 Entropy bonus applied, dt_range=[1,{max_dt}], avg_dt={avg_dt:.1f}")
            
            return entropy_bonus

    def _extract_token_entropy(self, data, response_mask):
        """Extract token-level entropy from various sources"""
        # Method 1: Use pre-computed entropys if available
        if "entropys" in data.batch:
            entropys = data.batch["entropys"]  # (bs, ?) - could be seq_len or response_len
            prompt_len = data.batch["prompts"].shape[-1]
            response_len = response_mask.shape[-1]
            
            # Check entropy shape to determine if it includes prompt or not
            if entropys.shape[-1] == prompt_len + response_len:
                token_entropy = entropys[:, prompt_len:]  # Extract response part
            elif entropys.shape[-1] == response_len:
                token_entropy = entropys
            else:
                # Unexpected shape, try to handle gracefully
                if entropys.shape[-1] >= response_len:
                    token_entropy = entropys[:, -response_len:]
                else:
                    raise ValueError(f"GTPO: Cannot extract response entropy from shape {entropys.shape}")
            
        # Method 2: Compute from logits if available
        elif "logits" in data.batch:
            logits = data.batch["logits"]  # (bs, seq_len, vocab_size)
            prompt_len = data.batch["prompts"].shape[-1]
            response_logits = logits[:, prompt_len:, :]  # Only response part
            
            # Compute token-level entropy: H(p) = -sum(p * log(p))
            probs = F.softmax(response_logits, dim=-1)
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (bs, response_len)
            
        # Method 3: Fallback to uniform entropy
        else:
            bs, response_len = response_mask.shape
            token_entropy = torch.ones(bs, response_len, device=response_mask.device)
        
        return token_entropy

    def _prepare_batch_data(self, data: DataProto):
        """Extract and preprocess all data for batch processing"""
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Batch decode responses
        responses_str = []
        prompts_str = []
        for i in range(len(data)):
            # Get valid prompt
            valid_prompt_length = attention_mask[i, :prompt_len].sum()
            valid_prompt_ids = prompt_ids[i, -valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            prompts_str.append(prompt_str)
            
            # Get valid response
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i, :valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]
            responses_str.append(response_str)

        # Extract other batch data
        ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
        data_sources = [item.non_tensor_batch[self.reward_fn_key] for item in data]
        extra_infos = [item.non_tensor_batch.get("extra_info", None) for item in data]

        return {
            'responses_str': responses_str,
            'prompts_str': prompts_str, 
            'ground_truths': ground_truths,
            'data_sources': data_sources,
            'extra_infos': extra_infos,
            'valid_response_lengths': valid_response_lengths
        }

    def __call__(self, data: DataProto, return_dict: bool = False):
        """GTPO reward computation with entropy-weighted token rewards"""

        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # GTPO: Extract response mask for entropy computation
        attention_mask = data.batch["attention_mask"]  # (bs, seq_len)
        prompt_len = data.batch["prompts"].shape[-1]
        response_mask = attention_mask[:, prompt_len:]  # Only response part
        
        # Use batch processing approach
        batch_data = self._prepare_batch_data(data)
        
        # Try batch processing first, fall back to individual if needed
        try:
            batch_result = self.compute_score(
                data_sources=batch_data['data_sources'],
                solution_strs=batch_data['responses_str'],
                ground_truths=batch_data['ground_truths'],
                extra_infos=batch_data['extra_infos'],
                **self.reward_kwargs
            )
            print(f"GTPO: Using BATCH processing for {len(batch_data['responses_str'])} items")
            
            scores = batch_result if isinstance(batch_result, list) else [batch_result] * len(data)
            
        except Exception as batch_error:
            print(f"GTPO: Batch processing failed ({batch_error}), falling back to individual processing")
            scores = []
            for i in range(len(data)):
                try:
                    result = self.compute_score(
                        data_source=batch_data['data_sources'][i],
                        solution_str=batch_data['responses_str'][i],
                        ground_truth=batch_data['ground_truths'][i],
                        extra_info=batch_data['extra_infos'][i],
                    )
                    if isinstance(result, dict):
                        score = result["score"]
                        for key, value in result.items():
                            reward_extra_info[key].append(value)
                    else:
                        score = result
                    scores.append(score)
                except Exception as individual_error:
                    print(f"GTPO: Individual processing failed for item {i}: {individual_error}")
                    scores.append(0.0)

        # GTPO: First collect base rewards, then compute entropy bonus
        base_rewards = torch.tensor(scores, device=reward_tensor.device, dtype=torch.float32)
        
        # Compute entropy bonus based on successful sequences (reward > 0)
        entropy_bonus = self._compute_entropy_bonus(data, response_mask, base_rewards)
        
        # GTPO: Apply entropy-weighted token-level rewards
        already_print_data_sources = {}
        successful_count = 0
        total_token_rewards = 0.0
        
        for i, score in enumerate(scores):
            base_reward = score
            valid_response_length = batch_data['valid_response_lengths'][i]
            
            # Apply overlong buffer logic if enabled
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                base_reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # GTPO: Apply the paper formula r̃i,t = ri + α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
            
            # Check if this sequence is successful (reward > 0) according to paper
            is_successful = base_reward > 0
            
            if is_successful:
                # For successful sequences: apply full GTPO formula r̃i,t = ri + entropy_bonus
                successful_count += 1
                
                # GTPO Formula: Each token gets the FULL base reward + its entropy bonus
                entropy_bonus_seq = entropy_bonus[i, :valid_response_length]
                token_rewards = torch.full((valid_response_length,), base_reward, device=reward_tensor.device)
                token_rewards += entropy_bonus_seq
                
            else:
                # For unsuccessful sequences: r̃i,t := 0 for all tokens (paper requirement)
                token_rewards = torch.zeros(valid_response_length, device=reward_tensor.device)
                entropy_bonus_seq = torch.zeros(valid_response_length, device=reward_tensor.device)
            
            total_token_rewards += token_rewards.sum().item()
            
            reward_tensor[i, :valid_response_length] = token_rewards
            
            # Store scalar information for debugging (avoid arrays with different shapes)
            reward_extra_info["base_rewards"].append(base_reward)
            reward_extra_info["entropy_bonus_sum"].append(entropy_bonus_seq.sum().cpu().item())
            reward_extra_info["entropy_bonus_mean"].append(entropy_bonus_seq.mean().cpu().item())
            reward_extra_info["token_rewards_sum"].append(token_rewards.sum().cpu().item())

            # Maintain printing logic for first few examples only
            data_source = batch_data['data_sources'][i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < min(self.num_examine, 2):  # Limit to 2 examples
                already_print_data_sources[data_source] += 1
                status = "SUCCESSFUL" if is_successful else "UNSUCCESSFUL"
                print(f"\n📝 [{status}] {data_source}: base={base_reward:.3f}, tokens={valid_response_length}, sum={token_rewards.sum():.1f}")
        
        print(f"🎯 GTPO Applied: {successful_count}/{len(scores)} successful, total_rewards={total_token_rewards:.1f}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor