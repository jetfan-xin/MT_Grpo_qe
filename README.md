### Ltgpu3 GRPO-QE Experiment

### 目标

将word-level QE作为奖励指标之一，加入到当前的机器翻译质量优化任务（包括中英双语互译）。



当前思路是把MQM sentence-level QE作为训练中的奖励指标，替换掉原有的comet+bleu。这个评分理论上是用error spans的翻译错误严重程度加权计算得到，间接融入word-level的特点。

可以用各种符合条件的QE。

实现了使用2022年cometkiwi论文中的sentence-level MQM的代码，训练交由Jingheng。

cometkiwi的sentence-level MQM中文分词效果不好，整句评分能用，但无法回过头找到对应的错误片段。

探索其他可用的指标，包括Cometkiwi，xCOMET，Instructscore，GEMBA-MQM。

发现**xCOMET**既能给出合理的error spans，也能适配中英双语互译，并且sentence-level MQM是完全基于error spans计算得到的。适合当前任务。



### 修改步骤：

要在同一个 verl-qe 环境里“同时用”：

- 序列级（新版，unified_metric；wmt23 ckpt）
- 词级（旧版 legacy，unite_metric_multi_task；WMT22 word-level ckpt）

#### 2025.9.6

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

      4.  接下来需要把word-level comet（以下简称QE）reward计算整合到comet_reward_batch.py
          TBD

### 2025.9.16

1. QE已整合到comet_reward_batch_qe.py。因为训练和验证共用一套指标计算函数，即comet_reward_batch_qe.py中的compute_score function，对该函数相关的函数如compute_score_batch、代码文件如ray_trainer等添加compute_val_reward判断参数，根据训练/验证情景选择相应的指标。训练使用format+QE；验证使用format+sequence level cometkiwi+bleu。

2. 中文日文分词问题：
   - tag计算方式：首先将一个句子用transformer分词为tokens（代码中变量为subwords），并识别哪几个tokens拼成一个词（word），对每个词首个token前方加"▁"。在词级计算tag，一个词一个tag。
   - 为什么中日文句子只有一个tag：transformer用空格做分界识别词范围，空格后的第一个token会被识别为词头，加"▁"。而中日文没有空格，也就不会加"▁"，这导致后续在词级计算时，会把所有tokens识别为一个词，只得到一个tags。
   - 修复方式：很简单，在文本句子转subwords list的函数运行后，通过jieba/fugashi分词，参照分词结果给subwords list中tokens加"▁"。（wmt22-comet-legacy/comet/encoders/bert.py中）
   - 但是不能这样做：因为该指标训练时使用原本的分token分词方式，如果我在使用checkpoint predict指标值时把原本的方式更改，则和checkpoint算法的逻辑不同，会造成偏差。最终没有实现分词。



### 2025.10.7

从几个论文里，cometkiwi, xcomet, instructscore, gemba mqm或者别的什么llm base的mqm方法，看看他们怎么算的error span的准确率的还是什么指标，然后测一下中到英和英到中。

#### CometKiwi: 

MQM和error spans使用不同的模型分别预测，二者没有关联。

##### Corpora:

Critical: Fine-tuning: zh-en (75k+ samples) & en-zh (500 samples)

1. MQM: a sentence-level QE

   - Pretraining: WMT 2017–2019 Metrics Shared Task 的 DA 数据

   - Fine-tuning: (not references) MQM annotations from WMT 2020 and 2021

2. Error span = a word-level QE task = predict OK/BAD labels for each word (combined by subwords).

   模型在训练时分为两个主要来源类型（即两类语料）

   1. Post-edit originated LPs: 人类在机器翻译基础上做修改后的语料

      Pretraining:

      - WMT 2017–2019 Metrics Shared Task 的 DA 数据; 
      - QT21 and 
      - APEQuest that include both word-level labels and sentence (HTER) scores

   2. MQM originated LPs: MQM标注数据

      Fine-tuning: (not references)

      - MQM annotations from WMT 2020 and 2021

      - Improvement: concatenated DA and MQM datasets together for a single fine-tuning.

##### Result Table:

Word-level QE on Post-edit originated LPs:  No zh-en & en-zh

MQM & word-level QE on MQM originated LPs: Only zh-en, no en-zh



#### 🟢 xCOMET: 

使用一个统一模型预测sentence-level MQM和error spans (combined by subwords)。MQM直接由error spans计算出来（“learn-to-detect” errors → infer MQM）。xCOMET 的 error span 模型只预测错误的严重程度（minor、major、critical、OK），不预测错误的类型（ fluency、accuracy 等）。

##### Corpora: (WMT22中有en-zh & zh-en MQM)

1. MQM annotations sourced from WMT from 2020 to 2022.
2. IndicMT
3. DEMETR

including source, MT, and references

##### Results Table:

Only zh-en, no en-zh.



#### Instructscore:

##### Corpora：(no En-Zh)

https://github.com/xu1998hz/InstructScore_SEScore3/tree/main/data

主要依赖 GPT-4 自动生成的 MQM 风格伪语料 进行大规模训练，
再利用 WMT22 MQM (Zh→En, En→De) 数据做小规模验证与对齐

1. Sentence-level MQM prediction
   - GPT-4 合成评分语料
   - DEMETR
   - WMT22 MQM (Zh→En, En→De)

2. Error Span prediction（错误类型、位置、严重度、解释字段的准确率与召回率）
   - GPT-4 合成错误标注语料
   - DEMETR 
   - MLQE-PE 
   - WMT22 MQM

| **Corpora**                   | **描述**                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| **GPT-4 合成语料**            | 10k 句子 × 100 域（news、medical、legal、social 等），覆盖多种 NLG 任务。 |
| **Synthetic Error Injection** | GPT-4 在原文上注入 1–5 个错误（error type、severity、位置、解释），模拟 MQM 式标注。 |
| **Error Type 集**             | 来自 MQM 框架（Freitag et al., 2021）：Addition, Omission, Mistranslation, Grammar, Style, Terminology 等。 |
| **Fine-tuning 数据规模**      | 每种任务生成约 10k pseudo pairs（reference + candidate + diagnostic report）。 |
| **Refinement 阶段数据**       | 从 WMT20 系统输出（2,000 Zh→En 样本）采样，GPT-4 自动评价模型输出的 failure modes 并进一步微调。 |



#### GEMBA-MQM

 GEMBA-MQM 并没有使用任何“训练数据”进行参数学习。基于 GPT-4 通过 prompt（fixed three-shot prompting technique）直接执行 MQM 风格错误检测。相当于直接用OpenAI GPT4进行检测，不是一个可下载、可训练的模型，

和我们需要快速测评用于训练环节的任务不适配。



#### WMT22-24

过去三年官方测评数据中没有En-Zh。但是可以参考MQM evaluation排名，看效果更好的指标是否在en-zh数据集上训练，有en-zh翻译测评能力。

22: English→German,  English→Russian, Chinese→English

23: English→German, Chinese→English, Hebrew→English

24: English→German, Japanese→Chinese, English→Spanish





### 2025.10.8

1. 将数据、模型、checkpoints结果、环境（copy）迁移到公共空间/mnt/data1/users/4xin
2. 在～./bashrc中设置了 HF_HOME / HF_HUB_CACHE：
   - HF_HOME：Hugging Face 的“总家目录”。设置后，Hub 模型缓存、datasets 缓存等都会默认放到这个根目录下（如 HF_HOME/hub、HF_HOME/datasets、HF_HOME/transformers）。
   - HF_HUB_CACHE：专门指定“Hub 模型缓存目录”。优先级高于 HF_HOME。Transformers/huggingface_hub 下载的模型权重、配置文件都会放在这里。
3. 成功在新环境verl运行原有项目
4. TO-DO：
   1. 总结bashrc中修改了什么
   2. 迁移了哪些内容？通过mnt中新增的内容轻松得知

遗留问题：当只有<=2张卡时运行项目，想要测试时，会卡在ray actor建立需要GPU但是卡不够的情况。

```
(autoscaler +9m40s) Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.
(autoscaler +9m40s) Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}. This is likely due to all cluster resources being claimed by actors. Consider creating fewer actors or adding more nodes to this Ray cluster.
(autoscaler +10m15s) Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}. This is likely due to all cluster resources being claimed by actors. Consider creating fewer actors or adding more nodes to this Ray cluster.
...

```



### 2025.10.9

1. Qwen2.5-3B可以在两块卡上训练起来。

2. 项目中Ray调度原理：main_ppo.py是项目的主程序，其中已经用Ray的ResourcePoolManager管理训练卡。

   - 会把要跑的角色（roles，本项目包括ActorRollout / Critic / RefPolicy / RewardModel）映射到一组Ray placement group资源池，资源池的容量为： trainer.nnodes × trainer.n_gpus_per_node，按照给每个role预留 GPU（以及必要的 CPU/内存），在所需的资源上创建各类worker。每个 worker 只“看到”自己那几张卡，从而实现卡的独占与隔离。

     - 把“角色”变成可远程创建的 Ray 类：

       ```python
       role_worker_mapping = {
                   Role.ActorRollout: ray.remote(actor_rollout_cls),
                   Role.Critic: ray.remote(CriticWorker),
               }
       ...
       role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
       ...
       role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
       ```

     - 定义每类角色用哪个资源池。本项目中所有角色使用同样的资源池（因为只有一个）：

       ```python
       global_pool_id = "global_pool"
       resource_pool_spec = {
         # 每个元素是一台节点的一个 bundle，值是该节点为训练预留的 GPU 数
         # 例：nnodes=2, n_gpus_per_node=4 → {"global_pool":[4,4]}
         global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
       }
       mapping = {
         Role.ActorRollout: global_pool_id,
         Role.Critic:       global_pool_id,
         # 可选
         Role.RewardModel:  global_pool_id,
         Role.RefPolicy:    global_pool_id,
       }
       ```

     - 创建管理器

       ```python
       resource_pool_manager = ResourcePoolManager(
         resource_pool_spec=resource_pool_spec,
         mapping=mapping
       )
       ```

     - 最后开始占卡训练：

       ```python
       trainer = RayPPOTrainer(..., role_worker_mapping=..., resource_pool_manager=..., ...)
       # Initialize the workers of the trainer.
       trainer.init_workers()
       # Start the training process.
       trainer.fit()
       ```

   - test阶段导致actor pending无法分配到GPU的原因：**卡不够所有actors分的就 Pending**：

     如果**必须同时使用卡**的actors所需要的卡的数量的需求 > 所有可用的卡数，Ray 就报：

     ```terminal
     cannot be scheduled ... {'CPU':1,'GPU':1}<--以上是想要建立新的actor需要的cpu和gpu个数，但是其中有不能满足的
     ```

     就会有 actor PENDING_CREATION。要么减少并行，要么把 trainer.n_gpus_per_node 或 CUDA_VISIBLE_DEVICES 扩到更多卡。

   - 这涉及到了具体的Ray actor占用GPU的原理：

     - trainer.n_gpus_per_node=2 时，**placement group** 会在该节点**预留 2 张 GPU** 给训练（ActorRollout/Critic/RefPolicy/RewardModel 这些“训练角色”在整个进程生命周期中一直持有 PG 的资源，不会中途释放）。
     - 这意味着**集群层面**已经“账面锁住”了这两张卡，哪怕某个训练 worker 暂时“闲着”，它依然占着这张卡。
     - 当测试阶段你的 **COMET** 想要另起一个 **独立 Ray actor（@ray.remote(num_gpus=1)）** 时，Ray 看账面没有“剩余 GPU”，于是 **这个 COMET actor 进入 Pending**，就会不停出现：

     ```
     cannot be scheduled right now: {'CPU': 1.0, 'GPU': 1.0}
     ```

     - 而且会**一直等**，除非训练释放 PG（不会）或你减少训练 PG 的 GPU 预留、或者机器有更多可见卡。
     - 反过来，trainer.n_gpus_per_node=1 时，PG 只锁 1 张卡；如果你机器可见 2 张卡，就还剩 1 张自由卡，COMET 的 num_gpus=1 actor 就能排上 → **不会 pending**。这也正是你观察到的现象。

     - 核心点：**训练的 PG 预留 == 长期占位**；你另外再起一个“独立占 GPU 的 COMET actor”，就必须有“PG 之外的空闲卡”。没有的话，就会一直 Pending。

   - Ray actor最终修改：**不要把 COMET 作为独立 Ray GPU actor 起**；而是**训练申请的同一个actor内**直接做 COMET 推理（共享同一 PG 的 GPU，不新增 num_gpus 需求）。

3. 重新给results按照GPU数量建立checkpoint目录。一般情况下使用不同数量的GPU得到的checkpoints不能通用。

   从

   ```
   trainer.default_local_dir="${results_path}/qwen2.5_3b_r1-zero" \
   ```

   变成

   ```
   trainer.default_local_dir="${results_path}/qwen2.5_3b_r1-zero/trainner_npus_${trainner_npus}" \
   ```

   



### 15/10 - 14/10/2025

1. Review the coding logic in reward and validation stages with different metrics.

2. The reason why COMET can only utilize cpu and cannot load on GPU:

   - In `main_ppo.py`, we have:

     ```python
     @ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
     class TaskRunner:
     ....
     ```

     which decides that COMET computation is loaded in this TaskRunner. While this TaskRunner obtains the  Ray actor by setup a process on cpu `@ray.remote(num_cpus=1)`: TaskRunner itself NO GPU!!! So TaskRunner process cannot see and call GPU. Because Comet computation is in TaskRunner process, it cannot see GPU.

   - 但**训练用的那些 worker**（Actor/Critic/Ref/RewardModel 等）都是 **单独的 Ray actor**，由 ResourcePoolManager + placement group 申请了 GPU，Ray 会给**每个 worker 进程**设置自己的 CUDA_VISIBLE_DEVICES，因此它们能在 GPU 上跑——这和 TaskRunner 是否有 GPU 完全无关。TaskRunner 只是“总控/调度”，不参与算力。

   - 如果不想再单独为 COMET 起一个 GPU actor（担心额外抢卡或 pending）。那就用==**方案 B**：把 COMET 的加载与推理嵌入**已有的 GPU worker**（例如 ActorRollout/RefPolicy/Critic 任意一个角色），由该 worker 在它自己的进程里（已经有 GPU 暴露的进程）完成 COMET 推理==。这样**没有新 actor**，也不会多申请 GPU；COMET 与训练共享这张（或几张）卡。

3. Realize the Ray actor assginment only on the trainning worker level, without exclusive gpu assginment for comet computation. --> test on both training (QE_MODE==off) and validation procedure

4. Add word level qe also in test stage. Just for a check

5. 删掉了在 Ray 多进程环境中重新初始化 CUDA 上下文（原因见gpt）

6. Replace current "word-level QE" with xcomet and run through the code



### 16/10/2025

1. 把指标计算加入到已有的GPU worker

   1. 解读main_ppo.py

      - Assign all the specified available CUDA for training (TaskRunner).

        - Setting num_gpus on a actor reserves GPU(s) for that actor, making it unavailable to others while the actor is alive. It does not change the cluster’s GPU capacity (specified in ray.init config); it just consumes from it.

      - When running TaskRunner, the config is defined through `/MT_Grpo_qe/verl/verl/trainer/config/ppo_trainer.yaml`, which has a basic config template maerged and overridden by CLI specified settings. We use Hydra (a multitask coordinater package) to coordinate them.

      - `ray.init(num_cpus=...)` vs `@ray.remote(num_cpus=1)` / `.options(num_gpus=1)`

        They’re related but not overlapping:

        `ray.init(num_cpus=...)` sets the total CPU capacity Ray believes this node has (a pool/limit for scheduling). We don’t pass `num_cpus` to `ray.init`, Ray auto-detects the machine’s CPU count and uses that as cluster capacity.

        `@ray.remote(num_cpus=1)` / `.options(num_cpus=..., num_gpus=...)` specifies the resources an actor/task reserves when scheduled. Our actor has `@ray.remote(num_cpus=1)`, so the scheduler will reserve 1 logical CPU for it as long as at least 1 is available.

      - Now, trainer.profile_steps = null (which enables profiling instrumentation 固定搭配，性能分析). We don't need it currently. We need it only if chase performance issues. It adds overhead 开销.

      - Add setting `ray.init: timeline_json_file: "ray_timeline.json"` . It is useful for diagnosing scheduling/perf issues across Ray workers.

      - **GPU assignment tips:**

        - If you do `TaskRunner.options(num_gpus=x)` (or `@ray.remote(num_gpus=x)`), that actor will hold x GPU for its lifetime. **If your training workers also need all GPUs, this will reduce what’s available for them.**
        - If your goal is to run COMET/XCOMET inside a training worker’s existing GPU, **don’t give the controller (`TaskRunner`) a GPU; instead, ensure the worker that computes rewards has `num_gpus=1`** and you run the metric on that same device (or coordinate device index explicitly). Otherwise you risk starving the vLLM/FSDP workers.

      - I didn’t set `TaskRunner.options(num_gpus=1)`. Did TaskRunner “get all GPUs” because of `ray.init(...)` with `CUDA_VISIBLE_DEVICES`?

        - **No.** GPUs are **not** reserved by `ray.init`. Ray only reserves GPUs when you specify `num_gpus` on a **task/actor** (`@ray.remote(num_gpus=...)` or `.options(num_gpus=...)`).
        - Without `num_gpus` on `TaskRunner`, it reserves **0 GPUs**. If your code inside TaskRunner calls CUDA directly, it *may* still **see** GPUs (via `CUDA_VISIBLE_DEVICES`) and try to use them—but Ray won’t account for that usage → **contention / OOM risk**.
        - Setting `CUDA_VISIBLE_DEVICES` globally in `runtime_env` only limits visibility; it does **not** reserve the devices. Other actors could still be scheduled to use the same GPUs unless they also reserve them.
        - **Bottom line:** You did **not** assign all GPUs to TaskRunner unless you explicitly set `num_gpus` on it.
        - **Ray can only protect you from Ray-scheduled contention.** If other students launch non-Ray jobs on the same machine, Ray can’t stop them from taking “your” GPU and causing OOM. To truly reserve GPUs from *everyone*, you need **system/cluster-level isolation** (Slurm, Kubernetes quotas, or nvidia-smi exclusive mode by an admin). That said, here’s how to best reserve GPUs *within Ray* and keep your own job safe/consistent.
        - However, since we only instantialize one Ray task with only one actor, the single TaskRunner can *see* all GPUs because of CUDA_VISIBLE_DEVICES.
        - **TaskRunner 要不要 GPU？**
          - **TaskRunner 就是一个 Ray actor**（你代码顶部有 @ray.remote(num_cpus=1)），但它只是**调度/控制器**：创建数据集、拼 worker 图、启动训练、拉日志等。
          - 给 **TaskRunner 配 num_gpus 并不会让 vLLM/FSDP 真正用到这些 GPU**；相反，这会**把 GPU 资源锁在控制器进程**上，导致**真正干活的 GPU worker（ActorRollout/Critic/RefPolicy）拿不到卡**。
          - 我之前提到 TaskRunner.options(num_gpus=available_gpus) 的语境是“**粗暴占卡**防别人抢”，但这和你当前使用 **RayPPOTrainer + FSDP/vLLM 的多 GPU worker** 模型是**冲突的**：你占住的卡，worker 就用不到了。因此**在你的架构里不要给 TaskRunner 配 GPU**。
          - 结论：**保持现在的 @ray.remote(num_cpus=1) 就好**；GPU 交给真正的 worker 去申请/使用。

      - If I set `ray.init: timeline_json_file: "ray_timeline.json"`, will it add overhead?

        - **Overhead:** minimal to moderate; Ray records events anyway and just dumps them when you call `ray.timeline`. Keeping it on occasionally is fine; for round-the-clock runs, leave it off unless diagnosing.



### 17/10/2025

实现方案B。但是采用的是自行创建RewardActor。



### 18/10/2025 - 19/10/2025

实现MetricRewardWorker，结构与RewardActorWorker相似，融入verl框架，将奖励作为平行worker。



### 20/10/2025

运行发现在第一轮训练中就OOM。原因：

- XCOMET-XXL 本体非常大（磁盘 40 GB，显存占用也很猛），而你同时还在 GPU 上跑：

- Qwen2.5-3B 的 Actor/Ref/VLLM KV-Cache；
- 以及 COMET（kiwi-xl）/XCOMET（xxl）的推理。

在同一台卡上“撞车”时，很容易把 80 GB A100 直接吃爆（你的回溯里 OOM 正是在 xcomet 前向时发生的）。

##### 解决办法：

首先尝试没有Xcomet是否能运行：

- 发现WORD_QE_MODE=off还是会加载xcomet：说明代码有错误，没有按照WORD_QE_MODE的要求调整加载的模型，修改代码：
  - 已删除mix_weights参数和“scores” key
    - 当前代码实现了把comet和xcomet放到GPU计算奖励值，之后传回comet_reward_batch_with_ray.py函数与format和bleu结合计算最终的激励值
    - 保留验证阶段判断。发现`batch.py`被我更改过，那个时候我是为了区别训练和验证输出的奖励指标值不同设置的。给BatchRewardManager.verify() added a config "compute_val_reward", which has a default value "None". 
      - 相应的，为了增加“compute_val_reward”调用参数，同时期我还修改了 `ray_trainer.py`,  这两个文件都没有保存original版本

- 代码修改成功，没有xcomet运行



### 21/10/2025

1. 但又发现comet模型不在CPU上加载：

   - 原因是reward_fn.bind_rm_wg(...) 没被执行（见上面的“函数对象 vs 模块函数”问题）。按照GPT所说的更改

   - 发现_use_remote_worker() is not None，但是

     ```
     out = _RM_WG.score(
                     src_mt_pairs=comet_data,
                     triplets=xcomet_data,
                     metrics=need_metrics
                 )
     ```

     **报错AssertionError:**

     Solution: one line code above MetricRewardWorker.score():@register() should specify the dispatch mode as: @register(dispatch_mode=Dispatch.ONE_TO_ALL)

     **报错TypeError：**

     Solution: **你这份 metric_worker.py 里 score() 本身“在单个 worker 上”的返回值确实是 Dict[str, List[float]]。** 但因为你给它加了 @register(dispatch_mode=Dispatch.ONE_TO_ALL)，当你从 **RayWorkerGroup 上调用** rm_wg.score(...) 时，**Ray 会把每个 DP rank 的返回值收集成一个 list**，所以**调用方拿到的是 List[Dict[str, List[float]]]**（每个元素对应一个 rank）。这正是你在 wandb 里看到 TypeError: list indices must be integers or slices, not str 的原因——你把这份 “list” 当成 “dict” 去 out["comet"] 了。

     - 改调用侧（保持你当前 worker 不变）

       在 comet_reward_batch_with_ray_debug.py 里加一个小工具_merge_remote_metric_out(out)，并在远端分支用它.

2. 成功在GPU上加载模型并计算comet

3. 首先尝试运行off mode 40轮，看看训练效果：

   - GPU利用率呈现**周期性的波峰波谷**：![image-20251021172254015](/Users/jingfanxin/Library/Application Support/typora-user-images/image-20251021172254015.png)

     #### 原因：

     非常好的观察 👍。

     你这两张 GPU 利用率曲线非常典型，表现出一种**“周期性波峰波谷”**的 GPU 负载模式。

     下面是你看到这种周期性“GPU 停工”的真实原因分析 👇

     

     **🧠 一句话结论**

     你的 GPU 在周期性地“暂停工作”，不是显卡出问题，而是因为：

     > **计算（forward/backward）阶段在 GPU 上，非计算阶段（reward 计算、数据整理、Ray 通信、CPU 同步、日志、advantage 估计等）在 CPU 上执行。**

     GPU 在等待这些 CPU 操作完成时就会闲置，于是形成“高负载 → 短暂停 → 再次高负载”的节奏。

     

     **🔍 从曲线看问题**

   - **左图（GPU Utilization）**：

     规律性的锯齿形波动，每隔几十秒掉到 0%，随后又瞬间回到 100%。

     ➤ 表明 GPU 训练是 **间歇性的计算—等待循环**。

   - **右图（Memory Allocated）**：

     内存保持中等偏高（约 40–80%），没有释放 → 说明模型常驻 GPU，只是计算暂时停止。

     ➤ 不是显存碎片问题或频繁加载模型的问题。

     

     ##### **🚦 周期性暂停的常见原因（按出现概率排序）**

     **① 奖励计算阶段（COMET / BLEU / XCOMET）**

     每个 rollout batch 结束后，会：

   - 把生成的翻译从 GPU 拷回 CPU；

   - 通过 Ray 调用远端 MetricRewardWorker.score()；

   - 加载并运行 COMET 模型（通常单独 GPU 或 CPU 上执行）；

   - 汇总后再回传 reward tensor。

     ➡️ 在这期间，**主训练 GPU 在等 reward 结果**，所以负载掉到 0%。

     > 你可以在日志中看到：

     ```
     Predicting DataLoader 0: 100%|██████████| 1/1 [01:00<00:00]
     invalid items: 85 / 90
     ```

     > 这整一分钟 GPU 几乎不干活，就是 CPU/IO/Ray 阶段。

     

     **② 数据加载（Dataloader + Tokenization）**

     在每个 batch 结束后，需要：

   - 解码/编码文本；

   - 转 tensor；

   - 拼 batch；

   - 分发到各 rank；

     这些都在 CPU 完成，GPU idle。

     > 表现：GPU 利用率掉 0–10 秒，CPU 占用短时飙高。

     

     **③ Ray actor 间通信 / 同步**

     你使用 RayWorkerGroup，每次 _RM_WG.score(...) 会：

   - 把 batch 切片发送给远端 actors；

   - 等待远端返回；

   - 聚合结果。

     如果网络延迟或序列化耗时高，GPU 也在“干等”。

     

     **④ 优势估计 / 归一化阶段**

     在日志中你看到：

     ```
     🎯 GRPO ADVANTAGE ESTIMATION - Starting computation
     📈 Advantages stats: min=..., mean=...
     ```

     这段是在 CPU 上做 torch 张量归一化统计的阶段。

     计算量小、I/O 多 → GPU idle。

     

     ##### **🧩 为什么不是“显存不够”或“并行问题”**

   - 如果是显存问题：曲线会出现**突降 + 长时间 0%**（甚至 crash）。

   - 如果是分布式死锁：利用率曲线会不同步（而你这几张几乎完全同步，说明 FSDP 同步良好）。

   - 所以你的 GPU 分配/并行没问题，只是 pipeline 有非 GPU 部分成为瓶颈。

     

     ##### **🚀 如何让 GPU 利用率更平滑**

     **✅ 1. 异步奖励计算（Async Reward)**

     不要在主训练 loop 等 reward。

     可以：

   - 把 MetricRewardWorker 改为异步 ray.remote(num_gpus=1)；

   - 主进程提前 prepare 下一个 batch；

   - 或使用 queue 异步获取 reward 结果。

     💡 Ray 官方推荐的 pattern:

     ```
     futures = [remote_score.remote(batch_i) for batch_i in batches]
     results = ray.get(futures)
     ```

     主 GPU 就可以继续跑下一批 rollout。

     **✅ 2. 增加 DataLoader worker 数**

     在 rollout 阶段：

     ```
     DataLoader(..., num_workers=8, pin_memory=True)
     ```

     让 CPU 能并行准备数据。

     **✅ 3. 缩短 reward 批处理延迟**

     你的 Predicting DataLoader 0: 1/1 [01:00<00:00] 说明每个 reward batch 太大。

     可减小 reward batch size（例如从 90 降到 32），减少每次 predict 的等待时间。

     **✅ 4. 提前 move tensors to GPU**

     有些 reward 函数（尤其 BLEU、COMET）会先在 CPU 预测后再 .to(device)。

     可改为直接在 GPU 上加载模型并在 batch 内运算，避免 device 来回切换。

     

     **✅ 5. 调低 logging / checkpoint 频率**

     每次 log 到 wandb、保存 checkpoint 都会阻塞训练几秒。

     

     **📊 如果你想确认每个阶段的时间占比**

     在训练主循环插入简单的计时器：

     ```
     import time
     
     t0 = time.time()
     rewards = compute_reward(...)
     print(f"⚙️ Reward compute time: {time.time()-t0:.2f}s")
     
     t1 = time.time()
     data = compute_advantage(...)
     print(f"🎯 Advantage compute time: {time.time()-t1:.2f}s")
     
     t2 = time.time()
     loss, stats = self.trainer.train_step(data)
     print(f"🚀 Train step (GPU) time: {time.time()-t2:.2f}s")
     ```

     你就能清楚看到：

   - GPU 部分占用（train step）

   - CPU 阶段（reward、advantage）

   - 等待时间（data I/O）

     

     **✅ 总结图解**

     

     | **周期阶段**     | **主要设备**  | **GPU状态** | **原因**            |
     | ---------------- | ------------- | ----------- | ------------------- |
     | forward/backward | GPU           | 🔥 高负载    | 模型训练            |
     | reward计算       | CPU / 远端GPU | 💤 空闲      | COMET/XCOMET评分    |
     | advantage估计    | CPU           | 😴 空闲      | GRPO统计/标准化     |
     | 数据准备         | CPU           | 💤 空闲      | Dataloader/tokenize |
     | 下一个batch开始  | GPU           | 🔥 再次100%  | forward再起         |

     👉 **你现在看到的 GPU “波动”其实是整个 RLHF/GRPO pipeline 的自然呼吸节奏。**

     只要 GPU 没长时间闲置（>50% 时间在 idle），整体 pipeline 就是健康的。

     如果你想进一步压榨利用率，可以尝试 “异步奖励 + 多线程数据加载 + 小批量评估” 三件事。

4. **一条 step 生成与训练所用的“样本条数（序列数）”**：

   ```
   samples_per_step_total = data.train_batch_size  ×  actor_rollout_ref.rollout.n
   ```

5. 更新筛选有用的wadb projects，并根据使用的奖励指标、模式和一轮step生成与寻来呢所有的样本条数命名这些projects

### 23/10/2025

1. 了解wandb上面各种参数的意思

2. checkpoints保存路径重新命名格式：

   ```
   trainer.default_local_dir="${results_path}/qwen2.5_3b_r1-zero/xcomet_${WORD_QE_MODE}/trainer_npus_${gpu_count}" \
   ```

   TODO：把之前off 128的路径改到：

   ```
   trainer.default_local_dir="/mnt/data1/users/4xin/MT_Grpo_qe/results/qwen2.5_3b_r1-zero/xcomet_off/trainer_npus_4/"
   ```


### config

更改GPU数量：

```shell
# 当GPU数量=3时
actor_rollout_ref.rollout.n=6 \ # 被三整除
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \ # 新增
actor_rollout_ref.rollout.log_prob_micro_batch_size=15 \
actor_rollout_ref.ref.log_prob_micro_batch_size=15\
data.train_batch_size=15 \
actor_rollout_ref.actor.ppo_mini_batch_size=15 \
trainer.n_gpus_per_node=3 \
rollout.n=6 \
# 删掉：与per gpu结尾的参数实际上是一个东西
actor_rollout_ref.actor.ppo_micro_batch_size=16  \
```



-------

### 开会纪要

1. 首先把xcomet的各种参数修改一下：

   1. optmizer占显存这件事（拍照了），可以修改参数，防止别人趁显存占用少，也把卡给占了。

   2. 验证存翻译的句子，看翻译效果（拍照了），修改参数，首先跑下试试

   3. 如果需要把项目发给xintong运行，需要上传到github，其中所有路径要修改成xintong八块卡的路径结构（详情见https://github.com/p1k0pan/zh_tox_lora README.md最下方2025年5月15日里ckpts的路径）

   4. 现在sensecore外部服务器还能用，占着卡呢。尽快把xcomet指标放到外部服务器上运行，别用XXL，用XL模型

2. 看些paper找找思路：

   1. 需要看用到的nonverifiable翻译任务各个不同的数据集，先看，找灵感。（数据集说的是https://arxiv.org/pdf/2510.06471）

   2. 看有没有什么别的PO可以借鉴（verl官网algorithm有列，如SPPO OPO）
   3. 读文章：INSTRUCTSCORE: Explainable Text Generation Evaluation with Fine-grained Feedback；Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation；MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators etc，还有数据集那篇，还可以看paper的references，看能不能找到什么灵感

3. 我们目前的方向是：
   1. 反方向证明GRPO可用性而DAPO/高熵效果没那么好
   2. 从什么角度，能够把QE和thinking结合成为一个方法
   3. 适应新的数据集，比如术语密度高的、文学数据集等