# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import os
import torch

from verl import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.decorator import register, Dispatch

# ========= 后端实现 =========

class _CometBackend:
    def __init__(self, ckpt: str, batch: int = 32, io: str = "pair"):
        from comet import load_from_checkpoint
        self.ckpt = ckpt
        self.batch = batch
        assert io in ("pair", "triplet")
        self.io = io
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None  # 延迟加载

    def _ensure_loaded(self):
        if self.model is None:
            from comet import load_from_checkpoint
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model = load_from_checkpoint(self.ckpt).to(dev)
                self.device = dev
            except RuntimeError as e:
                # 显存不够则退回 CPU
                self.model = load_from_checkpoint(self.ckpt).to("cpu")
                self.device = "cpu"
                print("[MetricRewardWorker][_CometBackend] Warning: loaded model on CPU due to OOM:", e)

    def score(self, items: List[Dict[str, str]]) -> List[float]:
        self._ensure_loaded()
        use_gpu = 1 if (self.device.startswith("cuda") and torch.cuda.is_available()) else 0
        if use_gpu == 0:
            print("[MetricRewardWorker][_CometBackend] Warning: running on CPU, this may be slow.")
        out = self.model.predict(items, batch_size=self.batch, gpus=use_gpu)
        return out["scores"] if isinstance(out, dict) else list(out.scores)


class _BLEUBackend:
    def __init__(self, tokenize: str = "13a", io: str = "triplet"):
        import sacrebleu
        self.sb = sacrebleu
        self.tokenize = tokenize
        self.io = io  # 需要 ref，因此默认 triplet

    def score(self, items: List[Dict[str, str]]) -> List[float]:
        scores = []
        for d in items:
            pred, ref = d.get("mt", ""), d.get("ref", "")
            s = self.sb.sentence_bleu(pred, [ref], lowercase=True, tokenize=self.tokenize).score / 100.0
            scores.append(float(s))
        return scores


# ========= Worker =========

class MetricRewardWorker(Worker):
    """
    轻量 Reward Worker：
      - score(src_mt_pairs, triplets): 返回各指标分与可选融合分
      - compute_rm_score(data): 兼容 VERL 训练循环的接口，产出 token-level rm_scores
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config  # 会是 config.reward_model
        self.backends: Dict[str, object] = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """metrics（来自 YAML 的 reward_model.reward_kwargs.metrics）示例：
        {
          "comet":  {"type":"comet",  "ckpt":"/path/kiwi.ckpt",   "batch":32, "io":"pair"},
          "xcomet": {"type":"comet",  "ckpt":"/path/xcomet.ckpt", "batch":32, "io":"triplet"},
          "bleu":   {"type":"bleu",   "tokenize":"zh", "io":"triplet"}
        }
        """
        print("[MetricRewardWorker] CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        print("[MetricRewardWorker] torch.cuda.device_count():", torch.cuda.device_count())
        
        metrics=self.config.reward_kwargs.get("metrics", {})
       
        # 构造后端
        for name, spec in metrics.items():
            spec = dict(spec)  # 复制以免修改上层
            io = spec.get("io", "pair")
            typ = spec.pop("type")
            spec.pop("io", None)
            if typ == "comet":
                self.backends[name] = _CometBackend(io=io, **spec) # 此时还没有加载checkpoint模型（None），但预留出了model的位置
            elif typ == "bleu":
                self.backends[name] = _BLEUBackend(io=io, **spec)
            else:
                raise NotImplementedError(f"[MetricRewardWorker] Unknown metric type: {typ}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def score(self,
              src_mt_pairs: Optional[List[Dict[str, str]]] = None,  # [{"src","mt"}]
              triplets: Optional[List[Dict[str, str]]] = None,      # [{"src","mt","ref"}]
              metrics: Optional[List[str]] = None):                 # 只跑这些指标；None 表示跑全部
        """只按需计算指定的 metrics，避免无谓的 GPU 开销。返回：{metric_name: [scores...]}"""
        run_set = set(metrics) if metrics else set(self.backends.keys())
        # 推断样本数
        
        out: Dict[str, List[float]] = {}

        for name, backend in self.backends.items():
            if name not in run_set:
                continue
            if backend.io == "pair":
                if src_mt_pairs:
                    out[name] = backend.score(src_mt_pairs)
            else:  # "triplet"
                if triplets:
                    out[name] = backend.score(triplets)
        return out

    # ====== 关键：提供和内置 RewardModelWorker 一致的接口 ======
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        """
        路线 A：不走内置 RM 打分短路，因此这里不返回 'rm_scores'。
        但 fit() 里会调用 `batch = batch.union(reward_tensor)`，
        所以仍需返回一个带 batch_size 的“占位” DataProto。
        我们返回一个零列张量占位，几乎无开销。
        """
        # 复用已有张量取得 batch 维度与 device，然后切出 0 列作为占位
        # 优先用 attention_mask；若不存在，可换 prompts
        if "attention_mask" in data.batch:
            base = data.batch["attention_mask"]
        elif "prompts" in data.batch:
            base = data.batch["prompts"]
        else:
            # 兜底：创建一个 [bsz, 0] 的 CPU 占位（极少出现）
            bsz = data.batch.batch_size[0]
            empty = torch.empty((bsz, 0), dtype=torch.float32)
            return DataProto.from_dict(tensors={"_rm_noop": empty})

        placeholder = base[:, :0].to(torch.float32)  # 形状 [bsz, 0]，零列
        # 关键点：不要使用键名 'rm_scores'，否则 BatchRewardManager 会短路
        return DataProto.from_dict(tensors={"_rm_noop": placeholder})