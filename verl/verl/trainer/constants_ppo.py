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

import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # Ensure CUDA devices are visible to Ray workers
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    Special handling for CUDA_VISIBLE_DEVICES to ensure GPU visibility in Ray workers.
    """
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    for key in list(runtime_env["env_vars"].keys()):
        # Always preserve CUDA_VISIBLE_DEVICES to ensure GPU access in Ray workers
        if key != "CUDA_VISIBLE_DEVICES" and os.environ.get(key) is not None: # can define other environment config by yourself
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
