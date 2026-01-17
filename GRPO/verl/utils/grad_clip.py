import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class PerModuleGradNormOptions:
    enable: bool = False
    auto_bounds: bool = False
    min_norm: float = 0.0
    max_norm: float = 1.0
    norm_type: float = 2.0
    low_mult: float = 1.0
    high_mult: float = 1.0
    raise_small: bool = False
    ema_beta: float = 0.98
    sync_dist: bool = False
    bounds_schedule: str = "none"
    high_mult_late: float = 0.0
    low_mult_late: float = 0.0
    bounds_switch_ratio: float = 0.5
    bounds_transition_ratio: float = 0.2
    max_up_scale: float = 0.0
    max_down_scale: float = 0.0
    output_dir: Optional[str] = None
    keep_global_clip: bool = False


class PerModuleGradClipper:
    """Module-based gradient norm clipper.
    """

    def __init__(self, *, options: PerModuleGradNormOptions, local_rank: int = 0):
        self.opt = options
        self.local_rank = int(local_rank)
        self.is_master = (self.local_rank == 0)

        # Buffers (accumulated per step)
        self.layer_sums: Dict[int, float] = defaultdict(float)
        self.module_sums: Dict[str, float] = defaultdict(float)

        # Parsing helpers
        self.layer_ids = set()
        self.modules = set()
        self.mod2params: Dict[str, List[torch.nn.Parameter]] = defaultdict(list)

        # EMA
        self.module_ema: Dict[str, float] = defaultdict(float)
        self._total_steps: int = 0

        # hooks
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Distributed info
        try:
            self._dist_inited = torch.distributed.is_available() and torch.distributed.is_initialized()
            self._world_size = torch.distributed.get_world_size() if self._dist_inited else 1
            self._rank = torch.distributed.get_rank() if self._dist_inited else 0
        except Exception:
            self._dist_inited = False
            self._world_size = 1
            self._rank = 0

        # Log Output directory
        self.viz_dir = self.opt.output_dir or os.path.join(os.getcwd(), "outputs", "grad_viz")
        if self.is_master:
            os.makedirs(self.viz_dir, exist_ok=True)
        self.csv_layer = os.path.join(self.viz_dir, "grad_per_layer.csv")
        self.csv_module = os.path.join(self.viz_dir, "grad_per_module.csv")
        self.csv_clip = os.path.join(self.viz_dir, "per_module_clip.csv")

        # Layer ID parsing (common paths for llama/gpt2)
        self.re_layer_llama = re.compile(r"model\.layers\.(\d+)\.")
        self.re_layer_gpt2 = re.compile(r"transformer\.h\.(\d+)\.")

        self.enabled = bool(self.opt.enable)

    # ---------- Name parsing and hooks ----------
    def _infer_layer_id(self, name: str):
        m = self.re_layer_llama.search(name) or self.re_layer_gpt2.search(name)
        return int(m.group(1)) if m else None

    def _infer_target_module(self, name: str):
        # 1. Match common module names
        common_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "down_proj", "up_proj", 
            "c_attn", "c_proj", "w1", "w2", "w3",
            "lm_head", "embed_tokens"
        ]
        for t in common_modules:
            if f".{t}." in name or name.endswith(f".{t}.weight") or name.endswith(f".{t}.bias"):
                return t
        
        # 2. Try to extract the second last part of the name as the module name
        toks = name.split(".")
        if len(toks) >= 2:
            candidate = toks[-2]
            # Simple filtering to avoid extracting numbers or "layers"
            if not candidate.isdigit() and candidate != "layers":
                return candidate
        return None

    def attach(self, model: torch.nn.Module):
        """Register parameter gradient hooks and build per-module index."""
        if not self.enabled:
            return
        found = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            lid = self._infer_layer_id(n)
            mod = self._infer_target_module(n)
            
            if lid is not None:
                self.layer_ids.add(lid)
            if mod is not None:
                self.modules.add(mod)
            
            # Skip hook if neither layer ID nor module can be identified
            if lid is None and mod is None:
                continue

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
            if self.is_master:
                print(f"[PerModuleGradClipper] Warning: No parameters hooked.")
            self.enabled = False
            return

        # Build per-module parameter index
        self._index_params_by_module(model)

        # Write CSV
        if self.is_master:
            with open(self.csv_layer, "w", encoding="utf-8") as f:
                f.write("step,layer_id,grad_l2\n")
            with open(self.csv_module, "w", encoding="utf-8") as f:
                f.write("step,module,grad_l2\n")
            with open(self.csv_clip, "w", encoding="utf-8") as f:
                f.write("step,module,norm,lower,upper,scale\n")

    def _index_params_by_module(self, model: torch.nn.Module):
        self.mod2params.clear()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mod = self._infer_target_module(n)
            if mod is None:
                continue
            self.mod2params[mod].append(p)

    # ---------- Computation and clipping ----------
    def _get_current_multipliers(self, step: int):
        base_low = self.opt.low_mult
        base_high = self.opt.high_mult
        if self.opt.bounds_schedule != "linear" or self._total_steps <= 0:
            return base_low, base_high
        late_low = self.opt.low_mult_late if self.opt.low_mult_late > 0 else base_low
        late_high = self.opt.high_mult_late if self.opt.high_mult_late > 0 else base_high
        prog = max(0.0, min(1.0, float(step) / max(1, self._total_steps)))
        s = self.opt.bounds_switch_ratio
        w = max(1e-6, self.opt.bounds_transition_ratio)
        if prog <= s:
            return base_low, base_high
        if prog >= s + w:
            return late_low, late_high
        alpha = (prog - s) / w
        return base_low * (1 - alpha) + late_low * alpha, base_high * (1 - alpha) + late_high * alpha

    @torch.no_grad()
    def _compute_global_module_norms(self) -> Dict[str, float]:
        mods = sorted(list(self.modules))
        if len(mods) == 0:
            return {}
        local_vals = [float(self.module_sums.get(m, 0.0)) for m in mods]
        if not (self.opt.sync_dist and self._dist_inited and self._world_size > 1):
            return {m: float(math.sqrt(v + 1e-16)) for m, v in zip(mods, local_vals)}
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
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
        if not (self.opt.sync_dist and self._dist_inited and self._world_size > 1):
            return {l: float(math.sqrt(v + 1e-16)) for l, v in zip(lids, local_vals)}
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        vec = torch.tensor(local_vals, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.SUM)
        vec = vec / float(self._world_size)
        reduced = vec.tolist()
        return {l: float(math.sqrt(v + 1e-16)) for l, v in zip(lids, reduced)}

    @torch.no_grad()
    def _clip_per_module(self, precomputed_norms: Optional[Dict[str, float]] = None, step: int = 0):
        if not self.enabled or len(self.mod2params) == 0:
            return []
        eps = 1e-12
        results = []
        low_mult, high_mult = self._get_current_multipliers(step)
        for mod, params in self.mod2params.items():
            if self.opt.auto_bounds:
                ema = float(self.module_ema.get(mod, 0.0))
                lower = max(self.opt.min_norm, ema * low_mult)
                upper = max(0.0, ema * high_mult)
            else:
                lower = self.opt.min_norm
                upper = self.opt.max_norm

            # Compute the norm for this module
            if precomputed_norms is not None and mod in precomputed_norms:
                norm = float(precomputed_norms[mod])
            else:
                if self.opt.norm_type != 2.0:
                    total = 0.0
                    for p in params:
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        total += g.abs().pow(self.opt.norm_type).sum().item()
                    norm = total ** (1.0 / self.opt.norm_type)
                else:
                    total = 0.0
                    for p in params:
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        total += g.float().pow(2).sum().item()
                    norm = math.sqrt(total)

            applied_scale = 1.0
            if upper > 0.0 and norm > (upper + eps):
                scale = upper / (norm + eps)
                if self.opt.max_down_scale > 0.0:
                    scale = max(scale, self.opt.max_down_scale)
                for p in params:
                    if p.grad is not None:
                        p.grad.detach().mul_(scale)
                applied_scale = scale
            elif self.opt.raise_small and lower > 0.0 and norm < (lower - eps):
                scale = lower / (norm + eps) if norm > 0 else 1.0
                if self.opt.max_up_scale > 0.0:
                    scale = min(scale, self.opt.max_up_scale)
                for p in params:
                    if p.grad is not None and scale != 1.0:
                        p.grad.detach().mul_(scale)
                applied_scale = scale

            results.append({
                "module": mod,
                "norm": float(norm),
                "lower": float(lower),
                "upper": float(upper),
                "scale": float(applied_scale),
            })

        return results


    def on_train_begin(self, total_steps: int = 0):
        self._total_steps = int(total_steps or 0)

    def on_before_optimizer_step(self, step: int = 0):
        if not self.enabled:
            return []
        cur_norms = self._compute_global_module_norms()
        results = self._clip_per_module(precomputed_norms=cur_norms, step=step)
        # EMA update (effective norm after clipping)
        if len(cur_norms) > 0 and results:
            for r in results:
                mod = r["module"]
                base = float(cur_norms.get(mod, r["norm"]))
                eff = base * float(r["scale"])
                self.module_ema[mod] = self.opt.ema_beta * self.module_ema[mod] + (1.0 - self.opt.ema_beta) * eff
        if self.is_master and results:
            with open(self.csv_clip, "a", encoding="utf-8") as f:
                for r in results:
                    f.write(f"{step},{r['module']},{r['norm']},{r['lower']},{r['upper']},{r['scale']}\n")
        return results

    def on_step_end(self, step: int = 0):
        if not self.enabled:
            self.layer_sums.clear(); self.module_sums.clear()
            return
        try:
            global_module = self._compute_global_module_norms()
            global_layer = self._compute_global_layer_norms()
        except Exception:
            global_module, global_layer = {}, {}

        layer_vals = global_layer if len(global_layer) > 0 else {int(k): math.sqrt(v + 1e-16) for k, v in self.layer_sums.items()}
        module_vals = global_module if len(global_module) > 0 else {str(k): math.sqrt(v + 1e-16) for k, v in self.module_sums.items()}

        if self.is_master:
            for lid in sorted(self.layer_ids):
                val = float(layer_vals.get(lid, 0.0))
                with open(self.csv_layer, "a", encoding="utf-8") as f:
                    f.write(f"{step},{lid},{val}\n")
            for mod in sorted(self.modules):
                val = float(module_vals.get(mod, 0.0))
                with open(self.csv_module, "a", encoding="utf-8") as f:
                    f.write(f"{step},{mod},{val}\n")

        # Clear buffers
        self.layer_sums.clear(); self.module_sums.clear()

    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
