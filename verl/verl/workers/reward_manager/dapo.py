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

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        # BATCH_MODIFICATION: Add batch processing parameters like BatchRewardManager
        **reward_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        # BATCH_MODIFICATION: Store additional reward kwargs for batch processing
        self.reward_kwargs = reward_kwargs

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    # BATCH_MODIFICATION: Add batch preprocessing method like BatchRewardManager
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
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # BATCH_MODIFICATION: Use batch processing approach
        batch_data = self._prepare_batch_data(data)
        
        # BATCH_MODIFICATION: Try batch processing first, fall back to individual if needed
        try:
            # Check if compute_score supports batch processing (like your compute_score_batch)
            batch_result = self.compute_score(
                data_sources=batch_data['data_sources'],
                solution_strs=batch_data['responses_str'],
                ground_truths=batch_data['ground_truths'],
                extra_infos=batch_data['extra_infos'],
                **self.reward_kwargs
            )
            print(f"BATCH_MODIFICATION: Using BATCH processing for {len(batch_data['responses_str'])} items")
            
            # Handle batch results
            scores = batch_result if isinstance(batch_result, list) else [batch_result] * len(data)
            
        except Exception as batch_error:
            print(f"BATCH_MODIFICATION: Batch processing failed ({batch_error}), falling back to individual processing")
            # Fall back to individual processing
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
                        # Store the information including original reward
                        for key, value in result.items():
                            reward_extra_info[key].append(value)
                    else:
                        score = result
                    scores.append(score)
                except Exception as individual_error:
                    print(f"BATCH_MODIFICATION: Individual processing failed for item {i}: {individual_error}")
                    scores.append(0.0)  # Default score on error

        # BATCH_MODIFICATION: Process scores and apply overlong buffer in batch
        already_print_data_sources = {}
        for i, score in enumerate(scores):
            reward = score
            valid_response_length = batch_data['valid_response_lengths'][i]
            
            # BATCH_MODIFICATION: Apply overlong buffer logic (keeping DAPO's special feature)
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            # BATCH_MODIFICATION: Maintain printing logic
            data_source = batch_data['data_sources'][i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", batch_data['prompts_str'][i])
                print("[response]", batch_data['responses_str'][i])
                print("[ground_truth]", batch_data['ground_truths'][i])
                print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
