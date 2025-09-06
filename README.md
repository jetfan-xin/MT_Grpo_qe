# MT_Grpo
## 2025.9.6
1. 修改 comet_reward_batch.py -> 分成两版:
  * comet_reward_batch_wo_ray.py: 不使用 Ray，修复设备选择 (如device = 'cuda' if torch.cuda.is_available() else 'cpu')
  * comet_reward_batch_with_ray.py: 使用 Ray actor，给 COMET 单独分配一张 GPU，正确调用 GPU（但是当所有卡都给GRPO用于训练时，启动comet computation后卡住）
2. 更新 custom_grpo_fast.sh 脚本"
    Word level comet QE指标在相同环境中成功跑通例子，接下来需要将其传入GRPO作为reward。
