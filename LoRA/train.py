import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import re
import math
from collections import defaultdict

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    full_finetune : Optional[bool] = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_weights: bool | str = field(default=True,metadata={"help": ("True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    use_dora : Optional[bool] = field(default=False)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset : Optional[bool] = field(default=False)
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)
    # Per-target-module grad normalization
    per_module_norm: bool = field(default=False, metadata={"help": "Enable per-target-module grad norm clipping (module-wise), disables global max_grad_norm to avoid conflicts."})
    per_module_max_norm: float = field(default=1.0, metadata={"help": "Max L2 norm per target module when per_module_norm=True."})
    per_module_norm_type: float = field(default=2.0, metadata={"help": "Norm type for per-module clipping, typically 2.0."})
    per_module_apply_to: str = field(default="lora_only", metadata={"help": "Apply module-wise norm to 'lora_only' or 'all_trainable'."})
    # Auto bounds (DeepSpeed-like): use EMA across modules and percentiles to determine lower/upper bounds
    per_module_auto_bounds: bool = field(default=False, metadata={"help": "Auto-compute per-module lower/upper bounds from EMA percentiles across modules."})
    per_module_min_norm: float = field(default=0.0, metadata={"help": "Lower floor for per-module norm when auto bounds disabled or as floor when enabled."})
    per_module_p_low: float = field(default=0.1, metadata={"help": "Lower percentile (0-1) for auto lower bound."})
    per_module_p_high: float = field(default=0.9, metadata={"help": "Upper percentile (0-1) for auto upper bound."})
    per_module_low_mult: float = field(default=1.0, metadata={"help": "Multiplier for auto lower bound."})
    per_module_high_mult: float = field(default=1.0, metadata={"help": "Multiplier for auto upper bound."})
    per_module_raise_small: bool = field(default=False, metadata={"help": "Also raise (scale up) module grads below lower bound (use with caution)."})
    per_module_ema_beta: float = field(default=0.98, metadata={"help": "EMA decay for per-module grad norm statistics used in auto bounds."})
    per_module_sync_dist: bool = field(default=False, metadata={"help": "All-reduce per-module grad L2^2 across DP ranks to compute unified global norms and apply consistent scaling on all ranks."})
    # Boundary schedule: adjust multipliers according to training progress
    per_module_bounds_schedule: str = field(default="none", metadata={"help": "Boundary schedule: 'none' or 'linear'."})
    per_module_high_mult_late: float = field(default=0.0, metadata={"help": "When schedule enabled, later-phase high_mult (<= initial). 0 means disabled -> use initial."})
    per_module_low_mult_late: float = field(default=0.0, metadata={"help": "When schedule enabled, later-phase low_mult (>= initial). 0 means disabled -> use initial."})
    per_module_bounds_switch_ratio: float = field(default=0.5, metadata={"help": "When schedule enabled, progress ratio to start transitioning [0,1]."})
    per_module_bounds_transition_ratio: float = field(default=0.2, metadata={"help": "When schedule enabled, linear transition width as ratio of total steps [0,1]."})
    # Scaling limits: cap max up/down scale per step
    per_module_max_up_scale: float = field(default=0.0, metadata={"help": ">1.0 to enable: cap scale upper bound (e.g., 1.1). 0 disables."})
    per_module_max_down_scale: float = field(default=0.0, metadata={"help": "in (0,1] to enable: cap scale lower bound (e.g., 0.9). 0 disables."})

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


class LoraGradPlotCallback(transformers.TrainerCallback):
    """
    A Trainer callback for clipping per-target-module gradient norms during training.
    """
    def __init__(self, output_dir: str, local_rank: int = 0,
                 enable_per_module_norm: bool = False,
                 per_module_max_norm: float = 1.0,
                 per_module_norm_type: float = 2.0,
                 per_module_apply_to: str = "lora_only",
                 per_module_auto_bounds: bool = False,
                 per_module_min_norm: float = 0.0,
                 per_module_p_low: float = 0.1,
                 per_module_p_high: float = 0.9,
                 per_module_low_mult: float = 1.0,
                 per_module_high_mult: float = 1.0,
                 per_module_raise_small: bool = False,
                 per_module_ema_beta: float = 0.98,
                 per_module_sync_dist: bool = False,
                 per_module_bounds_schedule: str = "none",
                 per_module_high_mult_late: float = 0.0,
                 per_module_low_mult_late: float = 0.0,
                 per_module_bounds_switch_ratio: float = 0.5,
                 per_module_bounds_transition_ratio: float = 0.2,
                 per_module_max_up_scale: float = 0.0,
                 per_module_max_down_scale: float = 0.0):
        # output_dir: Directory for saving logs and plots
        self.output_dir = output_dir
        # local_rank: Local rank of the current process for distributed training
        self.local_rank = local_rank
        # is_master: Flag indicating if this is the master process (rank 0), only master writes files and plots
        self.is_master = (local_rank == 0)

        # enabled: Flag indicating if the callback is enabled
        self.enabled = True
        self.hooks = []

        self.layer_sums = defaultdict(float)   # lid -> sum(grad^2)
        self.module_sums = defaultdict(float)  # mod -> sum(grad^2)

        self.per_layer_series = defaultdict(list)   # lid -> list[(step, l2)]
        self.per_module_series = defaultdict(list)  # mod -> list[(step, l2)]
        self.step_index = []

        self.layer_ids = set()
        self.modules = set()
        self.mod2params = defaultdict(list)  # target_module -> List[Parameter]
        
        # apply_to: Scope of application, "lora_only" or "all_trainable"
        self.apply_to = per_module_apply_to
        self.enable_per_module_norm = enable_per_module_norm
        self.per_module_max_norm = float(per_module_max_norm)
        self.per_module_norm_type = float(per_module_norm_type)
        self.per_module_auto_bounds = bool(per_module_auto_bounds)
        self.per_module_min_norm = float(per_module_min_norm)
        self.per_module_low_mult = float(per_module_low_mult)
        self.per_module_high_mult = float(per_module_high_mult)
        self.per_module_raise_small = bool(per_module_raise_small)
        self.ema_beta = float(per_module_ema_beta)
        self.per_module_sync_dist = bool(per_module_sync_dist)
        
        self.bounds_schedule = str(per_module_bounds_schedule or "none").lower()
        self.high_mult_late = float(per_module_high_mult_late)
        self.low_mult_late = float(per_module_low_mult_late)
        self.bounds_switch_ratio = float(per_module_bounds_switch_ratio)
        self.bounds_transition_ratio = float(per_module_bounds_transition_ratio)
        self.max_up_scale = float(per_module_max_up_scale)
        self.max_down_scale = float(per_module_max_down_scale)
        
        self.module_ema = defaultdict(float)  # EMA per module
        self._micro_step = 0  
        self._clipped_step_marker = -1  
        self._total_steps = 0
        self._last_effective_module_norms: Dict[str, float] = {}

        self.viz_dir = os.path.join(self.output_dir, "grad_viz")
        if self.is_master:
            os.makedirs(self.viz_dir, exist_ok=True)
        self.csv_layer = os.path.join(self.viz_dir, "grad_per_layer.csv")
        self.csv_module = os.path.join(self.viz_dir, "grad_per_module.csv")
        self.csv_module_clip = os.path.join(self.viz_dir, "per_module_clip.csv")

        self.re_layer_llama = re.compile(r"model\.layers\.(\d+)\.")
        self.re_layer_gpt2  = re.compile(r"transformer\.h\.(\d+)\.")

        try:
            self._dist_inited = torch.distributed.is_available() and torch.distributed.is_initialized()
            self._world_size = torch.distributed.get_world_size() if self._dist_inited else 1
            self._rank = torch.distributed.get_rank() if self._dist_inited else 0
        except Exception:
            self._dist_inited = False
            self._world_size = 1
            self._rank = 0

    def _get_current_multipliers(self, step: int):
        base_low = self.per_module_low_mult
        base_high = self.per_module_high_mult

        if self.bounds_schedule != "linear" or self._total_steps <= 0:
            return base_low, base_high
            
        late_low = self.low_mult_late if self.low_mult_late > 0 else base_low
        late_high = self.high_mult_late if self.high_mult_late > 0 else base_high
        
        prog = max(0.0, min(1.0, float(step) / max(1, self._total_steps)))
        
        s = self.bounds_switch_ratio
        w = max(1e-6, self.bounds_transition_ratio)
        
        if prog <= s:
            return base_low, base_high
        if prog >= s + w:
            return late_low, late_high
            
        alpha = (prog - s) / w
        cur_low = base_low * (1 - alpha) + late_low * alpha
        cur_high = base_high * (1 - alpha) + late_high * alpha
        
        return cur_low, cur_high

    def _infer_layer_id(self, name: str):
        m = self.re_layer_llama.search(name) or self.re_layer_gpt2.search(name)
        return int(m.group(1)) if m else None

    def _infer_target_module(self, name: str):
        tokens = name.split(".")
        for i, t in enumerate(tokens):
            if t in ("lora_A", "lora_B") and i - 1 >= 0:
                return tokens[i - 1]
        return None

    def _register_hooks(self, model: torch.nn.Module):
        found = 0
        for n, p in model.named_parameters():
            if p.requires_grad and ("lora_" in n):
                lid = self._infer_layer_id(n)
                mod = self._infer_target_module(n)

                if lid is not None:
                    self.layer_ids.add(lid)

                if mod is not None:
                    self.modules.add(mod)

                def make_hook(layer_id, module_name):
                    def _hook(grad):
                        if grad is None:
                            return
                        g2 = grad.detach().float().pow(2).sum().item()
                        if layer_id is not None:
                            self.layer_sums[layer_id] += g2
                        if module_name is not None:
                            self.module_sums[module_name] += g2
                    return _hook

                h = p.register_hook(make_hook(lid, mod))
                self.hooks.append(h)
                found += 1

        if found == 0:
            self.enabled = False

    def _index_params_by_module(self, model: torch.nn.Module):
        self.mod2params.clear()

        known_tokens = set(self.modules) if len(self.modules) > 0 else set([
            'q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','c_attn','c_proj'] )

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self.apply_to == 'lora_only' and ('lora_' not in n):
                continue

            mod = self._infer_target_module(n)

            if mod is None:
                for t in known_tokens:
                    if f'.{t}.' in n or n.endswith(f'.{t}.weight') or n.endswith(f'.{t}.bias'):
                        mod = t
                        break

            if mod is None:
                continue

            self.mod2params[mod].append(p)

    @torch.no_grad()
    def _clip_per_module(self, precomputed_norms: Dict[str, float] | None = None, cur_step: int | None = None):
        """Perform per-target-module grad norm clipping/normalization on current accumulated grads.
        """
        if not self.enable_per_module_norm or len(self.mod2params) == 0:
            return []

        eps = 1e-12
        results = []  # collect (module, norm, lower, upper, applied_scale)
        
        if cur_step is None:
            cur_low_mult, cur_high_mult = self.per_module_low_mult, self.per_module_high_mult
        else:
            cur_low_mult, cur_high_mult = self._get_current_multipliers(cur_step)
            
        for mod, params in self.mod2params.items():
            # Determine per-module bounds
            if self.per_module_auto_bounds:
                ema = float(self.module_ema.get(mod, 0.0))
                lower_bound = max(self.per_module_min_norm, ema * cur_low_mult)
                upper_bound = max(0.0, ema * cur_high_mult)
            else:
                lower_bound = self.per_module_min_norm
                upper_bound = self.per_module_max_norm

            if precomputed_norms is not None and mod in precomputed_norms:
                norm = float(precomputed_norms[mod])
            elif mod in self.module_sums and self.module_sums.get(mod, 0.0) > 0.0:
                norm = float(math.sqrt(self.module_sums.get(mod, 0.0) + 1e-16))
            else:
                if self.per_module_norm_type != 2.0:
                    total = 0.0
                    for p in params:
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        total += g.abs().pow(self.per_module_norm_type).sum().item()
                    norm = total ** (1.0 / self.per_module_norm_type)
                else:
                    total = 0.0
                    for p in params:
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        total += g.float().pow(2).sum().item()
                    norm = math.sqrt(total)
                    
            applied_scale = 1.0
            
            if upper_bound > 0.0 and norm > (upper_bound + eps):
                scale = upper_bound / (norm + eps)
                if self.max_down_scale > 0.0:
                    scale = max(scale, self.max_down_scale)

                for p in params:
                    if p.grad is not None:
                        p.grad.detach().mul_(scale)

                applied_scale = scale
                results.append({"module": mod, "norm": float(norm), "lower": float(lower_bound), "upper": float(upper_bound), "scale": float(applied_scale)})
                continue

            if self.per_module_raise_small and lower_bound > 0.0 and norm < (lower_bound - eps):
                scale = lower_bound / (norm + eps) if norm > 0 else 1.0
                if self.max_up_scale > 0.0:
                    scale = min(scale, self.max_up_scale)
                for p in params:
                    if p.grad is not None and scale != 1.0:
                        p.grad.detach().mul_(scale)
                applied_scale = scale
            results.append({"module": mod, "norm": float(norm), "lower": float(lower_bound), "upper": float(upper_bound), "scale": float(applied_scale)})

        return results

    @torch.no_grad()
    def _compute_global_module_norms(self) -> Dict[str, float]:
        mods = sorted(list(self.modules))
        if len(mods) == 0:
            return {}
        local_vals = [float(self.module_sums.get(m, 0.0)) for m in mods]
        if not (self.per_module_sync_dist and self._dist_inited and self._world_size > 1):
            return {m: float(math.sqrt(v + 1e-16)) for m, v in zip(mods, local_vals)}
        device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
        vec = torch.tensor(local_vals, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.SUM)
        vec = vec / float(self._world_size)
        reduced = vec.tolist()
        return {m: float(math.sqrt(v + 1e-16)) for m, v in zip(mods, reduced)}

    @torch.no_grad()
    def _compute_global_layer_norms(self) -> Dict[int, float]:
        lids = sorted(list(self.layer_ids))
        if len(lids) == 0:
            return {}
        local_vals = [float(self.layer_sums.get(l, 0.0)) for l in lids]
        if not (self.per_module_sync_dist and self._dist_inited and self._world_size > 1):
            return {l: float(math.sqrt(v + 1e-16)) for l, v in zip(lids, local_vals)}
        device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
        vec = torch.tensor(local_vals, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.SUM)
        vec = vec / float(self._world_size)
        reduced = vec.tolist()
        return {l: float(math.sqrt(v + 1e-16)) for l, v in zip(lids, reduced)}

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control
        self._register_hooks(model)
        # index params for per-module clipping (after LoRA is prepared)
        self._index_params_by_module(model)
        if self.is_master and self.enabled:
            with open(self.csv_layer, "w", encoding="utf-8") as f:
                f.write("step,layer_id,grad_l2\n")
            with open(self.csv_module, "w", encoding="utf-8") as f:
                f.write("step,module,grad_l2\n")
            with open(self.csv_module_clip, "w", encoding="utf-8") as f:
                f.write("step,module,norm,lower,upper,scale\n")
            if self._rank == 0:
                try:
                    print(f"[per-module-clip] modules detected: {sorted(list(self.modules))}", flush=True)
                    if self.enable_per_module_norm:
                        print(
                            f"[per-module-clip] apply_to={self.apply_to}, auto_bounds={self.per_module_auto_bounds}, "
                            f"ema_beta={self.ema_beta}, low_mult={self.per_module_low_mult}, high_mult={self.per_module_high_mult}, "
                            f"min_norm={self.per_module_min_norm}, norm_type={self.per_module_norm_type}, sync_dist={self.per_module_sync_dist}",
                            flush=True,
                        )
                        if self.bounds_schedule != "none":
                            print(
                                f"[per-module-clip] schedule={self.bounds_schedule}, switch_ratio={self.bounds_switch_ratio}, transition_ratio={self.bounds_transition_ratio}, "
                                f"late_low={self.low_mult_late or self.per_module_low_mult}, late_high={self.high_mult_late or self.per_module_high_mult}",
                                flush=True,
                            )
                        if self.max_up_scale > 0.0 or self.max_down_scale > 0.0:
                            print(
                                f"[per-module-clip] scale caps: max_up={self.max_up_scale or 'off'}, max_down={self.max_down_scale or 'off'}",
                                flush=True,
                            )
                except Exception:
                    pass
        # cache total steps if available
        try:
            self._total_steps = int(getattr(state, 'max_steps', 0) or 0)
        except Exception:
            self._total_steps = 0
        # cache logging interval
        try:
            self._print_every = int(getattr(args, 'logging_steps', 50) or 50)
        except Exception:
            self._print_every = 50
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        """Called after each backward (micro) step, before potential optimizer step.
        We use it to apply per-target-module clipping at accumulation boundary.
        """
        self._micro_step += 1
            # Compatibility fallback: if not using fp16 (no GradScaler), clip at accumulation boundary here.
        if self.enable_per_module_norm:
            try:
                if (self._micro_step % max(1, args.gradient_accumulation_steps)) == 0 and not getattr(args, 'fp16', False):
                    step = int(state.global_step)
                    # prevent double-run within same step
                    if self._clipped_step_marker != step:
                        cur_norms = self._compute_global_module_norms()
                        results = self._clip_per_module(precomputed_norms=cur_norms, cur_step=step)
                        if len(cur_norms) > 0 and results:
                            for r in results:
                                mod = r['module']
                                base = float(cur_norms.get(mod, r['norm']))
                                eff = base * float(r['scale'])
                                self._last_effective_module_norms[mod] = eff
                                self.module_ema[mod] = self.ema_beta * self.module_ema[mod] + (1.0 - self.ema_beta) * eff
                        if self.is_master and results:
                            with open(self.csv_module_clip, "a", encoding="utf-8") as f:
                                for r in results:
                                    f.write(f"{step},{r['module']},{r['norm']},{r['lower']},{r['upper']},{r['scale']}\n")
                            # brief console
                            log_every = getattr(args, 'logging_steps', 0) or 0
                            if log_every > 0 and (step % log_every == 0) and self._rank == 0:
                                changed = [f"{r['module']}x{r['scale']:.3f}" for r in results if abs(r['scale'] - 1.0) > 1e-3]
                                if changed:
                                    print(f"[per-module-clip] step {step}: " + ", ".join(changed[:8]))
                        self._clipped_step_marker = step
                        # Always print a brief summary every logging interval
                        if self._rank == 0:
                            try:
                                log_every = getattr(args, 'logging_steps', 0) or self._print_every
                                if log_every > 0 and (step % log_every == 0):
                                    downs = [r for r in results if r['scale'] < 1 - 1e-3]
                                    ups = [r for r in results if r['scale'] > 1 + 1e-3]
                                    msg = f"[per-module-clip] step {step}: down={len(downs)}, up={len(ups)}"
                                    if downs:
                                        mins = sorted(downs, key=lambda x: x['scale'])[:3]
                                        msg += " | min scales: " + ", ".join([f"{m['module']}x{m['scale']:.3f}" for m in mins])
                                    if ups:
                                        maxs = sorted(ups, key=lambda x: x['scale'], reverse=True)[:3]
                                        msg += " | max scales: " + ", ".join([f"{m['module']}x{m['scale']:.3f}" for m in maxs])
                                    print(msg, flush=True)
                            except Exception:
                                pass
            except Exception as e:
                if self.is_master:
                    print(f"[LoraGradPlotCallback] per-module clip (substep) failed: {e}")
        return control

    def on_before_optimizer_step(self, args, state, control, optimizer, **kwargs):
        """Called right before optimizer.step(); at this point, Trainer has already unscaled grads when using AMP.
        Safe point to apply per-module clipping regardless of AMP.
        """
        if self.enable_per_module_norm:
            try:
                cur_norms = self._compute_global_module_norms()
                if len(self.module_sums) == 0 and len(cur_norms) == 0:
                    for mod, params in self.mod2params.items():
                        total = 0.0
                        for p in params:
                            if p.grad is None:
                                continue
                            total += p.grad.detach().float().pow(2).sum().item()
                        if total > 0.0:
                            cur_norms[mod] = float(math.sqrt(total + 1e-16))
                step = int(state.global_step)
                results = [] if self._clipped_step_marker == step else self._clip_per_module(precomputed_norms=cur_norms, cur_step=step)
                if len(cur_norms) > 0 and results:
                    for r in results:
                        mod = r['module']
                        base = float(cur_norms.get(mod, r['norm']))
                        eff = base * float(r['scale'])
                        self._last_effective_module_norms[mod] = eff
                        self.module_ema[mod] = self.ema_beta * self.module_ema[mod] + (1.0 - self.ema_beta) * eff
                if self.is_master and results:
                    # write CSV rows
                    with open(self.csv_module_clip, "a", encoding="utf-8") as f:
                        for r in results:
                            f.write(f"{step},{r['module']},{r['norm']},{r['lower']},{r['upper']},{r['scale']}\n")
                    # Always print a concise summary every logging interval
                    if self._rank == 0:
                        try:
                            log_every = getattr(args, 'logging_steps', 0) or self._print_every
                            if log_every > 0 and (step % log_every == 0):
                                downs = [r for r in results if r['scale'] < 1 - 1e-3]
                                ups = [r for r in results if r['scale'] > 1 + 1e-3]
                                msg = f"[per-module-clip] step {step}: down={len(downs)}, up={len(ups)}"
                                if downs:
                                    mins = sorted(downs, key=lambda x: x['scale'])[:3]
                                    msg += " | min scales: " + ", ".join([f"{m['module']}x{m['scale']:.3f}" for m in mins])
                                if ups:
                                    maxs = sorted(ups, key=lambda x: x['scale'], reverse=True)[:3]
                                    msg += " | max scales: " + ", ".join([f"{m['module']}x{m['scale']:.3f}" for m in maxs])
                                print(msg, flush=True)
                        except Exception:
                            pass
            except Exception as e:
                if self.is_master:
                    print(f"[LoraGradPlotCallback] per-module clip (before opt step) failed: {e}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            self.layer_sums.clear()
            self.module_sums.clear()
            return control

        step = int(state.global_step)
        try:
            global_module = self._compute_global_module_norms()
            global_layer = self._compute_global_layer_norms()
        except Exception:
            global_module, global_layer = {}, {}
        layer_vals = global_layer if len(global_layer) > 0 else {int(k): math.sqrt(v + 1e-16) for k, v in self.layer_sums.items()}
        module_vals = global_module if len(global_module) > 0 else {str(k): math.sqrt(v + 1e-16) for k, v in self.module_sums.items()}

        self.step_index.append(step)
        for lid in sorted(self.layer_ids):
            val = float(layer_vals.get(lid, 0.0))
            self.per_layer_series[lid].append((step, val))
        for mod in sorted(self.modules):
            val = float(module_vals.get(mod, 0.0))
            self.per_module_series[mod].append((step, val))
        if self.is_master:
            for lid in sorted(self.layer_ids):
                val = float(layer_vals.get(lid, 0.0))
                with open(self.csv_layer, "a", encoding="utf-8") as f:
                    f.write(f"{step},{lid},{val}\n")
            for mod in sorted(self.modules):
                val = float(module_vals.get(mod, 0.0))
                with open(self.csv_module, "a", encoding="utf-8") as f:
                    f.write(f"{step},{mod},{val}\n")

        self.layer_sums.clear()
        self.module_sums.clear()
        self._last_effective_module_norms.clear()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.is_master or not self.enabled:
            return control

        try:
            if len(self.per_layer_series) > 0:
                plt.figure(figsize=(12, 7), dpi=150)
                cmap = plt.get_cmap("turbo", max(8, len(self.per_layer_series)))
                for i, lid in enumerate(sorted(self.per_layer_series.keys())):
                    series = sorted(self.per_layer_series[lid], key=lambda x: x[0])
                    xs = [s for s, _ in series]
                    ys = [v for _, v in series]
                    plt.plot(xs, ys, color=cmap(i), label=f"layer {lid}", linewidth=1.2)
                plt.xlabel("global step")
                plt.ylabel("||grad||_2 (LoRA, per layer)")
                plt.title("LoRA Gradient L2 per Layer")
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, "grad_per_layer.png"))
                plt.close()

            if len(self.per_module_series) > 0:
                plt.figure(figsize=(12, 7), dpi=150)
                cmap2 = plt.get_cmap("tab10", max(3, len(self.per_module_series)))
                for i, mod in enumerate(sorted(self.per_module_series.keys())):
                    series = sorted(self.per_module_series[mod], key=lambda x: x[0])
                    xs = [s for s, _ in series]
                    ys = [v for _, v in series]
                    plt.plot(xs, ys, color=cmap2(i % 10), label=mod, linewidth=1.8)
                plt.xlabel("global step")
                plt.ylabel("||grad||_2 (LoRA, per target_module)")
                plt.title("LoRA Gradient L2 per Target Module")
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=2, fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, "grad_per_module.png"))
                plt.close()
        except Exception as e:
            print(f"[LoraGradPlotCallback] Drawing failed: {e}")

        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        return control

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def build_model(script_args, checkpoint_dir):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    # Tokenizer
    
    if not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize LoRA/PiSSA/CLOVER adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True)
        else:
            logger.info(f'Init LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                use_dora=script_args.use_dora,
                runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=script_args.use_dora),
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                inference_mode=False,
                r=script_args.lora_rank, 
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    return model

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)

    all_training_dataset = []
    for task in script_args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{script_args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, script_args.dataset_split

        ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if script_args.local_rank == 0:
            print(f"{script_args.data_path}/{cur_task}/{cur_split}/{ds.num_rows}")
            for k,v in ds[0].items():
                print("-"*100)
                print(k,end=':\t')
                print(v)
            print("+"*100)
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset:
        if script_args.local_rank == 0:
            print(f"Shuffle dataset with seed={script_args.seed}")
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)

    if script_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

        
    if script_args.local_rank == 0:
        torch.distributed.barrier()
        print(model)
        logger.info("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    # if use per-module grad norm, disable global max_grad_norm to avoid conflicts
    if getattr(script_args, 'per_module_norm', False):
        script_args.max_grad_norm = 0.0
        if script_args.local_rank == 0:
            logger.info("Per-module grad norm enabled; disabling global max_grad_norm in Trainer to avoid conflicts.")
            print("[per-module-clip] enabled: disable global max_grad_norm; DeepSpeed gradient_clipping should be 0.0/auto.", flush=True)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(tokenizer))
    try:
        trainer.add_callback(LoraGradPlotCallback(
            output_dir=script_args.output_dir,
            local_rank=script_args.local_rank,
            enable_per_module_norm=getattr(script_args, 'per_module_norm', False),
            per_module_max_norm=getattr(script_args, 'per_module_max_norm', 1.0),
            per_module_norm_type=getattr(script_args, 'per_module_norm_type', 2.0),
            per_module_apply_to=getattr(script_args, 'per_module_apply_to', 'lora_only'),
            per_module_auto_bounds=getattr(script_args, 'per_module_auto_bounds', False),
            per_module_min_norm=getattr(script_args, 'per_module_min_norm', 0.0),
            per_module_p_low=getattr(script_args, 'per_module_p_low', 0.1),
            per_module_p_high=getattr(script_args, 'per_module_p_high', 0.9),
            per_module_low_mult=getattr(script_args, 'per_module_low_mult', 1.0),
            per_module_high_mult=getattr(script_args, 'per_module_high_mult', 1.0),
            per_module_raise_small=getattr(script_args, 'per_module_raise_small', False),
            per_module_ema_beta=getattr(script_args, 'per_module_ema_beta', 0.98),
            per_module_sync_dist=getattr(script_args, 'per_module_sync_dist', False),
            per_module_bounds_schedule=getattr(script_args, 'per_module_bounds_schedule', 'none'),
            per_module_high_mult_late=getattr(script_args, 'per_module_high_mult_late', 0.0),
            per_module_low_mult_late=getattr(script_args, 'per_module_low_mult_late', 0.0),
            per_module_bounds_switch_ratio=getattr(script_args, 'per_module_bounds_switch_ratio', 0.5),
            per_module_bounds_transition_ratio=getattr(script_args, 'per_module_bounds_transition_ratio', 0.2),
            per_module_max_up_scale=getattr(script_args, 'per_module_max_up_scale', 0.0),
            per_module_max_down_scale=getattr(script_args, 'per_module_max_down_scale', 0.0)
        ))
    except Exception as e:
        if script_args.local_rank == 0:
            print(f"Failed to add LoraGradPlotCallback: {e}")

    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    trainer.save_state()
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        

if __name__ == "__main__":
    train()
