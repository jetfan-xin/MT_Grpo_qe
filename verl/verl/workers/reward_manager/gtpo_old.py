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

    def _compute_entropy_bonus(self, data, response_mask):
        """
        Compute entropy-based bonus for each token according to GTPO paper formula:
        r̃i,t = ri + α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
        
        Args:
            data: DataProto containing token information
            response_mask: Attention mask for response tokens (bs, response_len)
            
        Returns:
            entropy_bonus: Bonus for each token (bs, response_len)
        """
        with torch.no_grad():
            print(f"\n🔍 GTPO ENTROPY DEBUG - Starting computation")
            print(f"📊 Input shapes: response_mask={response_mask.shape}")
            print(f"⚙️  Entropy parameters: beta={self.entropy_beta} (α in paper formula)")
            
            # Try to get entropy from multiple sources
            token_entropy = None
            
            # Method 1: Use pre-computed entropys if available
            if "entropys" in data.batch:
                print("✅ GTPO: Using pre-computed entropys")
                entropys = data.batch["entropys"]  # (bs, ?) - could be seq_len or response_len
                prompt_len = data.batch["prompts"].shape[-1]
                response_len = response_mask.shape[-1]
                
                print(f"📈 Entropy data: shape={entropys.shape}, dtype={entropys.dtype}")
                print(f"📏 Lengths: prompt_len={prompt_len}, response_len={response_len}")
                print(f"📊 Entropy stats: min={entropys.min():.6f}, max={entropys.max():.6f}, mean={entropys.mean():.6f}")
                
                # Check entropy shape to determine if it includes prompt or not
                if entropys.shape[-1] == prompt_len + response_len:
                    # Case 1: entropy includes prompt + response (use_remove_padding=True)
                    print("🔄 GTPO: Entropy includes prompt+response, extracting response part")
                    token_entropy = entropys[:, prompt_len:]  # Extract response part
                elif entropys.shape[-1] == response_len:
                    # Case 2: entropy only has response part (use_remove_padding=False)
                    print("✅ GTPO: Entropy is response-only")
                    token_entropy = entropys
                else:
                    # Unexpected shape, try to handle gracefully
                    print(f"⚠️  GTPO: Unexpected entropy shape {entropys.shape}, expected full_len={prompt_len + response_len} or response_len={response_len}")
                    # Try to take the last response_len tokens
                    if entropys.shape[-1] >= response_len:
                        token_entropy = entropys[:, -response_len:]
                        print(f"🔧 GTPO: Using last {response_len} tokens from entropy")
                    else:
                        raise ValueError(f"GTPO: Cannot extract response entropy from shape {entropys.shape}")
                
            # Method 2: Compute from logits if available
            elif "logits" in data.batch:
                print("🧮 GTPO: Computing entropy from logits")
                logits = data.batch["logits"]  # (bs, seq_len, vocab_size)
                prompt_len = data.batch["prompts"].shape[-1]
                response_logits = logits[:, prompt_len:, :]  # Only response part
                
                print(f"📈 Logits shape: {logits.shape}, response_logits: {response_logits.shape}")
                
                # Compute token-level entropy: H(p) = -sum(p * log(p))
                probs = F.softmax(response_logits, dim=-1)
                log_probs = F.log_softmax(response_logits, dim=-1)
                token_entropy = -(probs * log_probs).sum(dim=-1)  # (bs, response_len)
                
                print(f"📊 Computed entropy stats: min={token_entropy.min():.6f}, max={token_entropy.max():.6f}, mean={token_entropy.mean():.6f}")
                
            # Method 3: Fallback to uniform entropy
            else:
                print("⚠️  GTPO: No entropy source found, using uniform entropy")
                bs, response_len = response_mask.shape
                token_entropy = torch.ones(bs, response_len, device=response_mask.device)
                print(f"🔄 Created uniform entropy: shape={token_entropy.shape}")
            
            # Apply attention mask to entropy
            print(f"👀 Entropy before mask: shape={token_entropy.shape}, sum={token_entropy.sum():.4f}")
            token_entropy = token_entropy * response_mask
            print(f"👀 Entropy after mask: shape={token_entropy.shape}, sum={token_entropy.sum():.4f}")
            
            # Log per-sequence entropy stats for first few sequences
            bs = token_entropy.shape[0]
            for i in range(min(3, bs)):
                seq_entropy = token_entropy[i]
                active_tokens = response_mask[i].sum().int()
                seq_entropy_active = seq_entropy[:active_tokens]
                print(f"📋 Seq {i}: length={active_tokens}, entropy_sum={seq_entropy_active.sum():.4f}, entropy_mean={seq_entropy_active.mean():.4f}")
            
            # GTPO Formula: α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
            # For each time step t, compute sum of entropies across all sequences still generating
            bs, response_len = token_entropy.shape
            entropy_bonus = torch.zeros_like(token_entropy)
            
            print(f"\n🎯 GTPO FORMULA APPLICATION")
            print(f"📐 Processing {response_len} time steps for {bs} sequences")
            
            # Track statistics for debugging
            bonus_stats = []
            
            for t in range(response_len):
                # Get entropy at time step t across all sequences
                entropy_t = token_entropy[:, t]  # (bs,)
                
                # Count how many sequences are still generating at time step t (dt)
                active_mask_t = response_mask[:, t]  # (bs,)
                dt = active_mask_t.sum()  # Number of active sequences
                
                if dt > 0:
                    # Sum of entropies at time step t across all active sequences
                    entropy_sum_t = (entropy_t * active_mask_t).sum()
                    
                    if entropy_sum_t > 1e-8:  # Avoid division by zero
                        # Apply GTPO formula: α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
                        entropy_ratios = entropy_t / entropy_sum_t  # Hi,t / Σ(k=1 to n)Hk,t
                        entropy_bonus[:, t] = (self.entropy_beta * 
                                             entropy_ratios * 
                                             dt * active_mask_t)
                        
                        # Debug logging for first few time steps
                        if t < 5 or t % 10 == 0:
                            active_entropies = entropy_t[active_mask_t.bool()]
                            active_bonuses = entropy_bonus[:, t][active_mask_t.bool()]
                            print(f"⏰ t={t}: dt={dt.int()}, entropy_sum={entropy_sum_t:.6f}")
                            print(f"   Active entropies: {active_entropies.tolist()}")
                            print(f"   Active bonuses: {active_bonuses.tolist()}")
                        
                        # Track stats
                        active_bonuses_t = entropy_bonus[:, t][active_mask_t.bool()]
                        if len(active_bonuses_t) > 0:
                            bonus_stats.append({
                                't': t,
                                'dt': dt.item(),
                                'entropy_sum': entropy_sum_t.item(),
                                'bonus_mean': active_bonuses_t.mean().item(),
                                'bonus_std': active_bonuses_t.std().item() if len(active_bonuses_t) > 1 else 0.0
                            })
                        
                    else:
                        # If all entropies are zero, give uniform bonus
                        uniform_bonus = self.entropy_beta * (1.0 / dt) * dt
                        entropy_bonus[:, t] = uniform_bonus * active_mask_t
                        print(f"⚠️  t={t}: Zero entropy sum, using uniform bonus={uniform_bonus:.6f}")
                        
                        bonus_stats.append({
                            't': t,
                            'dt': dt.item(),
                            'entropy_sum': 0.0,
                            'bonus_mean': uniform_bonus,
                            'bonus_std': 0.0
                        })
            
            # Summary statistics
            total_bonus = entropy_bonus.sum()
            active_bonuses = entropy_bonus[response_mask.bool()]
            
            print(f"\n📊 ENTROPY BONUS SUMMARY")
            print(f"✅ Total bonus sum: {total_bonus:.4f}")
            print(f"📈 Active bonus stats: mean={active_bonuses.mean():.6f}, std={active_bonuses.std():.6f}")
            print(f"📊 Bonus range: min={active_bonuses.min():.6f}, max={active_bonuses.max():.6f}")
            
            # Show per-sequence bonus distribution
            for i in range(min(3, bs)):
                seq_bonus = entropy_bonus[i]
                active_tokens = response_mask[i].sum().int()
                seq_bonus_active = seq_bonus[:active_tokens]
                print(f"🎯 Seq {i} bonus: sum={seq_bonus_active.sum():.4f}, mean={seq_bonus_active.mean():.6f}")
            
            return entropy_bonus

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
        
        # Compute entropy bonus for response tokens according to GTPO formula
        entropy_bonus = self._compute_entropy_bonus(data, response_mask)
        
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

        # GTPO: Apply entropy-weighted token-level rewards
        print(f"\n🎯 GTPO REWARD APPLICATION")
        print(f"📊 Processing rewards for {len(scores)} sequences")
        
        already_print_data_sources = {}
        for i, score in enumerate(scores):
            base_reward = score
            valid_response_length = batch_data['valid_response_lengths'][i]
            
            print(f"\n🔄 Processing sequence {i}: base_reward={base_reward:.6f}, length={valid_response_length}")
            
            # Apply overlong buffer logic if enabled
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                base_reward += overlong_reward
                print(f"📏 Overlong logic: expected={expected_len}, exceed={exceed_len}, penalty={overlong_reward:.6f}")
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # GTPO: Apply the paper formula r̃i,t = ri + α * (Hi,t / Σ(k=1 to n)Hk,t) * dt
            
            # Step 1: Give base reward to last token (like DAPO)
            token_rewards = torch.zeros(valid_response_length, device=reward_tensor.device)
            token_rewards[-1] = base_reward  # Only last token gets base reward
            print(f"🎯 Step 1: Assigned base_reward={base_reward:.6f} to last token (position {valid_response_length-1})")
            
            # Step 2: Add entropy bonus to all tokens
            entropy_bonus_seq = entropy_bonus[i, :valid_response_length]
            token_rewards += entropy_bonus_seq
            
            print(f"🔧 Step 2: Added entropy bonus to all tokens")
            print(f"   Entropy bonus stats: min={entropy_bonus_seq.min():.6f}, max={entropy_bonus_seq.max():.6f}, mean={entropy_bonus_seq.mean():.6f}")
            print(f"   Token reward stats: min={token_rewards.min():.6f}, max={token_rewards.max():.6f}, sum={token_rewards.sum():.6f}")
            
            # Verify GTPO formula application
            final_last_token_reward = token_rewards[-1].item()
            entropy_bonus_last = entropy_bonus_seq[-1].item()
            expected_last_token = base_reward + entropy_bonus_last
            
            print(f"🧮 Formula verification for last token:")
            print(f"   base_reward + entropy_bonus = {base_reward:.6f} + {entropy_bonus_last:.6f} = {expected_last_token:.6f}")
            print(f"   actual_final_reward = {final_last_token_reward:.6f}")
            print(f"   difference = {abs(expected_last_token - final_last_token_reward):.8f}")
            
            # Show token-by-token breakdown for first few tokens
            print(f"🎯 Token breakdown (first 5 and last 3):")
            for tok_idx in list(range(min(5, valid_response_length))) + list(range(max(0, valid_response_length-3), valid_response_length)):
                if tok_idx < valid_response_length:
                    base_part = base_reward if tok_idx == valid_response_length - 1 else 0.0
                    bonus_part = entropy_bonus_seq[tok_idx].item()
                    total_reward = token_rewards[tok_idx].item()
                    print(f"   Token {tok_idx:2d}: base={base_part:.6f} + bonus={bonus_part:.6f} = {total_reward:.6f}")
            
            reward_tensor[i, :valid_response_length] = token_rewards
            
            # Store scalar information for debugging (avoid arrays with different shapes)
            reward_extra_info["base_rewards"].append(base_reward)
            reward_extra_info["entropy_bonus_sum"].append(entropy_bonus_seq.sum().cpu().item())
            reward_extra_info["entropy_bonus_mean"].append(entropy_bonus_seq.mean().cpu().item())
            reward_extra_info["token_rewards_sum"].append(token_rewards.sum().cpu().item())

            # Maintain printing logic
            data_source = batch_data['data_sources'][i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n📝 DETAILED EXAMPLE {data_source}:")
                print("[prompt]", batch_data['prompts_str'][i])
                print("[response]", batch_data['responses_str'][i])
                print("[ground_truth]", batch_data['ground_truths'][i])
                print("[base_score]", score)
                print(f"[entropy_bonus] avg: {entropy_bonus_seq.mean():.4f}, sum: {entropy_bonus_seq.sum():.4f}")
                print(f"[total_reward] base: {base_reward:.4f}, with_bonus: {token_rewards.sum():.4f}")
                print(f"[reward_distribution] last_token: {token_rewards[-1]:.4f}, other_avg: {token_rewards[:-1].mean():.4f}")
                print(f"[gtpo_verification] Total tokens: {valid_response_length}, Non-zero rewards: {(token_rewards != 0).sum()}")
        
        print(f"\n📋 GTPO REWARD SUMMARY")
        total_reward_sum = reward_tensor.sum()
        non_zero_rewards = (reward_tensor != 0).sum()
        print(f"✅ Total reward sum across all sequences: {total_reward_sum:.6f}")
        print(f"📊 Non-zero reward positions: {non_zero_rewards}")
        print(f"📈 Reward tensor shape: {reward_tensor.shape}")
        print(f"🎯 GTPO reward computation completed!")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor