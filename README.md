# Ltgpu3 GRPO-QE Experiment

## 目标

要在同一个 verl-qe 环境里“同时用”：

- 序列级（新版，unified_metric；wmt23/24 ckpt）
- 词级（旧版 legacy，unite_metric_multi_task；WMT24 word-level ckpt）



## 修改步骤：

### 2025.9.6

0. QEmodel不相关的：

   - 修改 comet_reward_batch.py -> 分成两版:

     - comet_reward_batch_wo_ray.py: 不使用 Ray，修复设备选择 (如device = 'cuda' if torch.cuda.is_available() else 'cpu')

     - comet_reward_batch_with_ray.py: 使用 Ray actor，给 COMET 单独分配一张 GPU，正确调用 GPU（但是当所有卡都给GRPO用于训练时，启动comet computation后卡住）

   - 更新了 custom_grpo_fast.sh 脚本

1. pip clone 对应 legacy后我做了 pip install -e .，但陆陆续续出现库版本依赖问题，主要修改包括：卸载torchvision，更改tranformer版本，更改torch=2.6.0，vllm=0.8.4，解决版本冲突。unbabel-comet==2.2.6

2. 在使用wmt22-comet-legacy 的comet库（版本较旧）时发现，版本较新的cometkiwi sequence level的 wmt23的ckpt无法使用，因为它指定参数class_identifier: unified_metric，而这个参数在legacy的旧版本comet库中不存在，会报错。

   最终计划采用的解决方式是：**并存两套包，但用不同的模块名导入**。也就是：让新版继续叫 comet，保持你现有的 sequence-level（wmt23/24 unified_metric）路径不变。把 legacy 作为“源码目录”用别名比如 comet_legacy 动态导入（不去覆盖 site-packages 里的新版 comet），另起一个 word-level legacy 模型实例。这样就可以在 predict 时并行/顺序调用两路，然后把词级结果加到返回的 extra_infos，或组合到奖励里（例如线性加权、阈值裁剪等）。

3. 现在能够跑通word level cometliwi指标计算例子（legacy路径下的test.py）。
   - 在例子输出中，word level cometliwi predict的score不是comet，是句级 QE 预测值（拟合 DA/MQM/HTER 的回归输出）

4. 在现有 comet_reward_batch.py 里怎么同时用 sequence-level + word-level

   1. 改用**别名加载**（仅在需要时把 legacy 当作一个“命名模块”注入）。

   2. 把 legacy 包里的所有comet绝对导入改成相对导入。如：

      ```python
      # from comet.models import available_metrics
      from .models import available_metrics
      ```

      使用批量查找代码：

      ```shell
      grep -R "from comet\." -n ~/MT_Grpo_qe/wmt22-comet-legacy/comet || true
      # 如果有其它命中，再按需把 `from comet.xxx` 改成 `from .xxx`
      ```

      把层级修对！需要把深层模块的相对层级改正确。常见的修法是：

      - 顶层 comet/*.py 里：

        from comet.xxx import ... → from .xxx import ...

      - 子目录 comet/models/*.py 里：

        from comet.models.base import ... → from .base import ...

        from comet.models.utils import ... → from .utils import ...

      - 子子目录 comet/models/regression/*.py、comet/models/ranking/*.py 里：

        from comet.models.base import ... → from ..base import ...

        from comet.models.utils import ... → from ..utils import ...

   3.  **comet_reward_batch.py中载入QE checkpoint**。修改custom_reward_batch.py，在 comet_reward_batch.py 里，用“临时别名加载器”把 legacy 的comet包加载为 comet_legacy。加载完成后立刻恢复，所以不会污染后续对新版 comet 的使用。

   4. 接下来需要把word-level comet（以下简称QE）reward计算整合到comet_reward_batch.py
      TBD



### Other Possible TO-DOs:

解决comet不调用GPU的问题。ray actor只需要**声明需要多少块 GPU**，Ray 就会**自动挑选空闲的卡**并把 CUDA_VISIBLE_DEVICES 设置好给该任务/actor。

- num_gpus=1 ⇒ 这 1 张卡在 **Ray 内部**就**专属于 COMET Actor**，训练 Actor 就不会再拿到它。
- 如果训练进程也在 Ray 里调度，就不会“抢同一张卡”。若有 Ray 之外的进程，它们依然可能看到那张物理卡——尽量把所有 GPU 任务纳入 Ray 管理，或用 placement group 明确打包资源。