#!/usr/bin/env python3
"""Run the downstream SKA next-token retrofit experiment.

The script replaces selected attention layers with a teacher-blended SKA
module, warm-initializes SKA using one of the tested spectral strategies,
and trains the SKA path on WikiText next-token loss.
"""

import os
import sys
import math
import argparse
import random
import time
import json
import inspect
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from ska_distill.warm_init import warm_init_ska_from_attention

# Operator consistency check

def verify_operator():
    ska_path = os.path.join(os.path.dirname(__file__), "ska_distill", "ska.py")
    if not os.path.exists(ska_path):
        print("WARNING: Cannot find ska_distill/ska.py")
        return
    with open(ska_path) as f:
        code = f.read()
    if "cholesky_solve(M_flat" in code:
        old = (
            "    Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L_flat)\n"
            "    A_w = Aw_T.transpose(-1, -2)"
        )
        new = (
            "    # Whitened Koopman: A_w = L^{-1} M L^{-T}\n"
            "    Z = torch.linalg.solve_triangular(L_flat, M_flat, upper=False)\n"
            "    A_w = torch.linalg.solve_triangular(L_flat, Z.mT, upper=False).mT"
        )
        if old in code:
            code = code.replace(old, new)
            with open(ska_path, "w") as f:
                f.write(code)
            print("Fixed: A_w = L^{-1} M L^{-T}")
    else:
        print("OK: operator verified")


class QwenSKARetrofitWrapper(torch.nn.Module):
    """Teacher-blended SKA wrapper for gated-attention models."""

    def __init__(self, orig_attn, ska_module, rho_init=0.0):
        super().__init__()
        self.orig_attn = orig_attn
        self.ska = ska_module
        self.register_buffer("rho", torch.tensor(float(rho_init)))

        # Qwen-style attention stores the gate in the second half of q_proj.
        q_weight = orig_attn.q_proj.weight.data
        q_out = q_weight.shape[0]
        half = q_out // 2
        d_in = orig_attn.q_proj.in_features

        self.gate_proj = torch.nn.Linear(d_in, half, bias=False)
        self.gate_proj.weight.data.copy_(q_weight[half:].clone())
        device = orig_attn.q_proj.weight.device
        dtype = orig_attn.q_proj.weight.dtype
        self.gate_proj = self.gate_proj.to(device=device, dtype=dtype)

        self.last_distill = None

    def set_rho(self, new_rho):
        self.rho.fill_(new_rho)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, position_embeddings=None, **kwargs):
        rho_val = self.rho.item()

        # Keep the teacher path differentiable for checkpointing compatibility.
        if rho_val < 1.0:
            attn_result = self.orig_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if isinstance(attn_result, tuple):
                y_attn = attn_result[0]
            else:
                y_attn = attn_result
        else:
            y_attn = None

        y_core = self.ska(hidden_states)  # [B, T, d_model] (includes SKA's out_proj)

        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, T, d_model]
        y_ska = y_core * gate

        if y_attn is not None:
            rho = self.rho.to(y_ska.dtype)
            y = (1.0 - rho) * y_attn + rho * y_ska
            self.last_distill = (y_ska.float(), y_attn.detach().float())
        else:
            y = y_ska
            self.last_distill = None

        return y, None


class PythiaSKARetrofitWrapper(torch.nn.Module):
    """Teacher-blended SKA wrapper for GPT-NeoX/Pythia attention."""

    def __init__(self, orig_attn, ska_module, rho_init=0.0):
        super().__init__()
        self.orig_attn = orig_attn
        self.ska = ska_module
        self.register_buffer("rho", torch.tensor(float(rho_init)))
        self.last_distill = None

    def set_rho(self, new_rho):
        self.rho.fill_(new_rho)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                head_mask=None, layer_past=None, use_cache=False,
                output_attentions=False, cache_position=None, **kwargs):
        rho_val = self.rho.item()

        if rho_val < 1.0:
            attn_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "head_mask": head_mask,
                "layer_past": layer_past,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
                "cache_position": cache_position,
                **kwargs,
            }
            sig = inspect.signature(self.orig_attn.forward)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                attn_kwargs = {k: v for k, v in attn_kwargs.items() if k in sig.parameters}
            attn_result = self.orig_attn(**attn_kwargs)
            y_attn = attn_result[0] if isinstance(attn_result, tuple) else attn_result
        else:
            attn_result = None
            y_attn = None

        y_ska = self.ska(hidden_states)

        if y_attn is not None:
            rho = self.rho.to(y_ska.dtype)
            y = (1.0 - rho) * y_attn + rho * y_ska
            self.last_distill = (y_ska.float(), y_attn.detach().float())
        else:
            y = y_ska
            self.last_distill = None

        if output_attentions:
            return y, None, None
        return y, None


def _is_pythia_like(model):
    return hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers")


def _get_transformer_layers(model):
    if _is_pythia_like(model):
        return model.gpt_neox.layers
    return model.model.layers


def _get_layer_attention(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    raise ValueError(f"Cannot find attention module in {type(layer).__name__}")


def _set_layer_attention(layer, attn_module):
    if hasattr(layer, "self_attn"):
        layer.self_attn = attn_module
    elif hasattr(layer, "attention"):
        layer.attention = attn_module
    else:
        raise ValueError(f"Cannot set attention module in {type(layer).__name__}")


def _get_retrofit_wrapper(model, idx):
    return _get_layer_attention(_get_transformer_layers(model)[idx])


def _set_retrofit_wrapper(model, idx, wrapper):
    _set_layer_attention(_get_transformer_layers(model)[idx], wrapper)


def _get_model_dims(model):
    cfg = model.config
    d_model = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (d_model // n_heads)
    return d_model, n_heads, head_dim


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_metrics_row(args, stage_name, step, total_steps, lm_loss, ppl, distill, rho):
    path = os.path.join(args.output_dir, "metrics.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "init_strategy", "scale_a", "power_b", "seed",
                "mlp_state_path", "mlp_hidden_dim", "mlp_residual_scale",
                "rank", "n_ska_layers", "stage", "step", "total_steps",
                "lm_loss", "ppl", "distill_loss", "rho",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "model": args.model,
            "init_strategy": args.init_strategy,
            "scale_a": args.scale_a,
            "power_b": args.power_b,
            "mlp_state_path": args.mlp_state_path,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "mlp_residual_scale": args.mlp_residual_scale,
            "seed": args.seed,
            "rank": args.ska_rank,
            "n_ska_layers": args.n_ska_layers,
            "stage": stage_name,
            "step": step,
            "total_steps": total_steps,
            "lm_loss": lm_loss,
            "ppl": ppl,
            "distill_loss": distill,
            "rho": rho,
        })


# Dataset

class LMRetrievalDataset(IterableDataset):
    """Iterable WikiText LM stream with optional synthetic retrieval examples."""

    def __init__(self, tokenizer, max_len=2048, seed=42,
                 retrieval_ratio=0.3, n_facts=2, distractor_len=256,
                 n_queries=2):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.retrieval_ratio = retrieval_ratio
        self.n_facts = n_facts
        self.distractor_len = distractor_len
        self.n_queries = min(n_queries, n_facts)

    def _load_wikitext(self, seed_offset=0):
        from datasets import load_dataset
        configs = ["wikitext-103-raw-v1", "wikitext-2-raw-v1"]
        for config in configs:
            try:
                ds = load_dataset("wikitext", config, split="train")
                return ds.shuffle(seed=self.seed + seed_offset)
            except Exception:
                continue
        return None

    def _make_retrieval(self, rng, dist_buf, doc_iter):
        facts = [(rng.randint(1000, 9999), rng.randint(10000, 99999))
                 for _ in range(self.n_facts)]

        fact_text = "".join(
            f"Fact {i+1}: The code for item {k} is {v}.\n"
            for i, (k, v) in enumerate(facts)
        )
        fact_ids = self.tokenizer(fact_text, add_special_tokens=False)["input_ids"]

        if doc_iter:
            while len(dist_buf) < self.distractor_len:
                try:
                    doc = next(doc_iter)
                    text = doc.get("text", "")
                    if text:
                        toks = self.tokenizer(text, truncation=False,
                                              add_special_tokens=False)["input_ids"]
                        dist_buf.extend(toks)
                except StopIteration:
                    break
        dist_ids = dist_buf[:self.distractor_len]
        del dist_buf[:self.distractor_len]

        queried = rng.sample(facts, self.n_queries)
        query_text = "".join(
            f"Question: What is the code for item {k}?\nAnswer: {v}\n"
            for k, v in queried
        )
        query_ids = self.tokenizer(query_text, add_special_tokens=False)["input_ids"]

        all_ids = fact_ids + dist_ids + query_ids
        if len(all_ids) > self.max_len:
            all_ids = all_ids[:self.max_len]

        ids = torch.tensor(all_ids, dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

    def __iter__(self):
        ds1 = self._load_wikitext(0)
        ds2 = self._load_wikitext(100)
        it1 = iter(ds1) if ds1 else None
        it2 = iter(ds2) if ds2 else None
        rng = random.Random(self.seed)
        lm_buf, dist_buf = [], []

        while True:
            if rng.random() < self.retrieval_ratio:
                yield self._make_retrieval(rng, dist_buf, it2)
            else:
                if it1:
                    while len(lm_buf) < self.max_len:
                        try:
                            doc = next(it1)
                            text = doc.get("text", "")
                            if text:
                                toks = self.tokenizer(text, truncation=False,
                                                      add_special_tokens=False)["input_ids"]
                                lm_buf.extend(toks)
                                lm_buf.append(self.tokenizer.eos_token_id)
                        except StopIteration:
                            break
                if len(lm_buf) >= self.max_len:
                    chunk = lm_buf[:self.max_len]
                    lm_buf = lm_buf[self.max_len:]
                    ids = torch.tensor(chunk, dtype=torch.long)
                    yield {
                        "input_ids": ids,
                        "labels": ids.clone(),
                    }


# Batch collation

def collate_fn(batch, pad_token_id=0):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attention_mask[i, :L] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# Checkpointing

def save_checkpoint(model, ska_indices, step, output_dir, prefix="step"):
    path = f"{output_dir}/{prefix}_{step}"
    os.makedirs(path, exist_ok=True)

    ska_state = {}
    for idx in ska_indices:
        wrapper = _get_retrofit_wrapper(model, idx)
        for name, param in wrapper.ska.named_parameters():
            ska_state[f"ska.{idx}.{name}"] = param.data.cpu()
        if hasattr(wrapper, "gate_proj"):
            ska_state[f"gate_proj.{idx}.weight"] = wrapper.gate_proj.weight.data.cpu()
        ska_state[f"rho.{idx}"] = wrapper.rho.cpu()

    torch.save(ska_state, f"{path}/ska_weights.pt")

    config = {
        "step": step, "prefix": prefix,
        "ska_indices": ska_indices,
    }
    with open(f"{path}/ska_config.json", "w") as f:
        json.dump(config, f, indent=2)

    sz = os.path.getsize(f"{path}/ska_weights.pt") / 1e6
    print(f"  Saved: {path} ({sz:.0f}MB)")


def load_ska_checkpoint(model, ska_indices, checkpoint_path):
    ska_path = f"{checkpoint_path}/ska_weights.pt"
    if not os.path.exists(ska_path):
        print(f"  No checkpoint at {ska_path}")
        return
    state = torch.load(ska_path, map_location="cpu")
    device = next(model.parameters()).device

    for idx in ska_indices:
        wrapper = _get_retrofit_wrapper(model, idx)
        for name, param in wrapper.ska.named_parameters():
            key = f"ska.{idx}.{name}"
            if key in state:
                param.data.copy_(state[key].to(device).to(param.dtype))
        gkey = f"gate_proj.{idx}.weight"
        if gkey in state and hasattr(wrapper, "gate_proj"):
            wrapper.gate_proj.weight.data.copy_(state[gkey].to(device).to(wrapper.gate_proj.weight.dtype))
        rkey = f"rho.{idx}"
        if rkey in state:
            wrapper.rho.fill_(state[rkey].item())
            print(f"  Layer {idx}: rho={wrapper.rho.item():.3f}")

    print(f"  Loaded SKA checkpoint from {checkpoint_path}")


# Training loop

def train_loop(model, loader, param_groups, args, device, ska_indices,
               total_steps, warmup_steps, rho_start, rho_end,
               distill_weight=0.0, stage_name=""):
    print(f"\n{'=' * 60}")
    print(f"STAGE {stage_name}: {total_steps} steps, rho {rho_start:.2f} -> {rho_end:.2f}")
    print(f"  distill_weight={distill_weight}")
    print(f"{'=' * 60}")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    step = micro = 0
    log_lm_loss = 0.0
    log_distill = 0.0
    log_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        progress = min(step / max(total_steps, 1), 1.0)
        current_rho = rho_start + (rho_end - rho_start) * progress
        for idx in ska_indices:
            _get_retrofit_wrapper(model, idx).set_rho(current_rho)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            out = model(input_ids=input_ids, labels=labels,
                        attention_mask=attn_mask)
            loss = out.loss / args.grad_accum

            d_loss = torch.tensor(0.0, device=device)
            if distill_weight > 0:
                for idx in ska_indices:
                    wrapper = _get_retrofit_wrapper(model, idx)
                    if wrapper.last_distill is not None:
                        y_ska, y_attn = wrapper.last_distill
                        mse = F.mse_loss(y_ska, y_attn)
                        norm = (y_attn.detach() ** 2).mean() + 1e-6
                        d_loss = d_loss + mse / norm
                d_loss = d_loss / len(ska_indices)
                loss = loss + distill_weight * d_loss / args.grad_accum

        loss.backward()

        log_lm_loss += out.loss.item()
        log_distill += d_loss.item()
        log_count += 1
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % args.logging_steps == 0:
                avg_lm = log_lm_loss / max(log_count, 1)
                avg_dist = log_distill / max(log_count, 1)
                ppl = math.exp(min(avg_lm, 20))

                info = []
                for idx in ska_indices:
                    w = _get_retrofit_wrapper(model, idx)
                    info.append(f"{idx}:rho={w.rho.item():.2f}")

                print(f"[{stage_name}] step {step}/{total_steps} | "
                      f"loss {avg_lm:.4f} ppl {ppl:.1f} "
                      f"distill {avg_dist:.4f} | "
                      f"rho={current_rho:.3f} | "
                      f"{' '.join(info)}")
                append_metrics_row(
                    args=args,
                    stage_name=stage_name,
                    step=step,
                    total_steps=total_steps,
                    lm_loss=avg_lm,
                    ppl=ppl,
                    distill=avg_dist,
                    rho=current_rho,
                )

                log_lm_loss = 0.0
                log_distill = 0.0
                log_count = 0

            if step > 0 and step % args.save_steps == 0:
                save_checkpoint(model, ska_indices, step, args.output_dir,
                                f"stage{stage_name}")

            if step >= total_steps:
                break

    save_checkpoint(model, ska_indices, step, args.output_dir,
                    f"stage{stage_name}_final")
    return step


# Model setup

def setup_model(args, device):
    from ska_distill.ska import SKAModule

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _get_transformer_layers(model)
    is_pythia = _is_pythia_like(model)

    if is_pythia:
        ska_indices = list(range(len(layers)))
    else:
        ska_indices = []
        for i, layer in enumerate(layers):
            lt = getattr(layer, "layer_type", None)
            if lt == "full_attention":
                ska_indices.append(i)
        if not ska_indices:
            ska_indices = [3, 7, 11, 15, 19, 23, 27, 31]
    print(f"Attention layers detected: {ska_indices}")

    if args.ska_layer_indices:
        requested = [int(x) for x in args.ska_layer_indices.split(",") if x.strip()]
        missing = [idx for idx in requested if idx not in ska_indices]
        if missing:
            raise ValueError(
                f"Requested SKA layers {missing} are not valid attention layers. "
                f"Available layers: {ska_indices}"
            )
        ska_indices = requested
    elif args.n_ska_layers < len(ska_indices):
        step_size = max(1, len(ska_indices) // args.n_ska_layers)
        ska_indices = ska_indices[::step_size][:args.n_ska_layers]
    print(f"SKA layers: {ska_indices}")

    d_model, n_heads, head_dim = _get_model_dims(model)
    wrapper_cls = PythiaSKARetrofitWrapper if is_pythia else QwenSKARetrofitWrapper
    print(f"Model dims: d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")

    print(f"Creating SKA modules (rank={args.ska_rank}, K={args.power_K}, "
          f"chunk={args.ska_chunk_size})")

    ska_modules = nn.ModuleDict()
    orig_attns = {}

    for idx in ska_indices:
        layer = layers[idx]
        orig_attn = _get_layer_attention(layer)

        ska = SKAModule(
            d_model=d_model,
            n_heads=n_heads,
            rank=args.ska_rank,
            head_dim=head_dim,
            chunk_size=args.ska_chunk_size,
            power_K=args.power_K,
        ).to(device).to(torch_dtype)

        ska_modules[str(idx)] = ska
        orig_attns[idx] = orig_attn

    warm_init_ska_from_attention(
        teacher_model=model,
        ska_modules=ska_modules,
        distill_indices=ska_indices,
        rank=args.ska_rank,
        strategy=args.init_strategy,
        scale_a=args.scale_a,
        power_b=args.power_b,
        mlp_state_path=args.mlp_state_path,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_residual_scale=args.mlp_residual_scale,
        verbose=True,
        diagnostics=True,
    )

    for idx in ska_indices:
        layer = layers[idx]
        wrapper = wrapper_cls(orig_attns[idx], ska_modules[str(idx)], rho_init=0.0)
        _set_layer_attention(layer, wrapper)
        path = "attn + SKA" if is_pythia else "attn + gate*SKA"
        print(f"  Layer {idx}: {path} warm-initialized from attention")

    model.gradient_checkpointing_enable()

    if args.checkpoint:
        load_ska_checkpoint(model, ska_indices, args.checkpoint)

    total = sum(p.numel() for p in model.parameters())
    ska_n = sum(
        p.numel()
        for idx in ska_indices
        for p in _get_retrofit_wrapper(model, idx).ska.parameters()
    )
    print(f"Total: {total/1e9:.1f}B | SKA: {ska_n/1e6:.0f}M")

    return model, tokenizer, ska_indices


# Parameter groups

def build_param_groups(model, args, ska_indices, stage):
    ska_param_ids = set()
    ska_params, gate_params = [], []

    for idx in ska_indices:
        w = _get_retrofit_wrapper(model, idx)
        for p in w.ska.parameters():
            ska_param_ids.add(id(p))
            ska_params.append(p)
        if hasattr(w, "gate_proj"):
            gate_params.append(w.gate_proj.weight)
            ska_param_ids.add(id(w.gate_proj.weight))

    norm_params, dn_params, mlp_params, other_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in ska_param_ids:
            continue
        nl = name.lower()
        if "norm" in nl or "layernorm" in nl:
            norm_params.append(param)
        elif any(x in nl for x in ["delta", "conv", "recurrence"]):
            dn_params.append(param)
        elif any(x in nl for x in ["mlp", "gate_proj", "up_proj", "down_proj"]):
            mlp_params.append(param)
        else:
            other_params.append(param)

    if stage == 1:
        for name, p in model.named_parameters():
            if id(p) not in ska_param_ids:
                p.requires_grad_(False)
        groups = [
            {"params": ska_params, "lr": args.ska_lr, "name": "SKA"},
            {"params": gate_params, "lr": args.ska_lr * 0.5, "name": "Gate"},
        ]
    elif stage == 2:
        for name, p in model.named_parameters():
            if id(p) in ska_param_ids:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        groups = [
            {"params": ska_params, "lr": args.ska_lr, "name": "SKA"},
            {"params": gate_params, "lr": args.ska_lr * 0.5, "name": "Gate"},
        ]
    else:  # stage 3
        for p in model.parameters():
            p.requires_grad_(True)
        for idx in ska_indices:
            w = _get_retrofit_wrapper(model, idx)
            for p in w.orig_attn.parameters():
                p.requires_grad_(False)
        groups = [
            {"params": ska_params, "lr": args.ska_lr, "name": "SKA"},
            {"params": gate_params, "lr": args.ska_lr * 0.5, "name": "Gate"},
            {"params": [p for p in norm_params if p.requires_grad], "lr": args.norm_lr, "name": "Norms"},
            {"params": [p for p in dn_params if p.requires_grad], "lr": args.deltanet_lr, "name": "DeltaNet"},
            {"params": [p for p in mlp_params if p.requires_grad], "lr": args.mlp_lr, "name": "MLPs"},
        ]

    groups = [g for g in groups if len(g["params"]) > 0]
    for g in groups:
        print(f"  {g['name']}: {len(g['params'])} params, lr={g['lr']:.1e}")
    return groups


# Evaluation helpers

def run_stage_0(model, tokenizer, ska_indices, device, args):
    """Verify rho=0 gives same PPL as base model."""
    print("\n" + "=" * 60)
    print("STAGE 0: Sanity check (rho=0 should match base model PPL)")
    print("=" * 60)

    for idx in ska_indices:
        _get_retrofit_wrapper(model, idx).set_rho(0.0)

    model.eval()
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception as exc:
        print(f"  WikiText-2 unavailable ({type(exc).__name__}); trying WikiText-103.")
        try:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            text = "\n\n".join(ds["text"][:256])
        except Exception:
            print("  WikiText unavailable; using a small built-in sanity prompt.")
            text = (
                "The quick brown fox jumps over the lazy dog.\n"
                "Language models predict the next token from context.\n"
            ) * 16
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=2048)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        out_wrapped = model(input_ids=input_ids, labels=input_ids)
    ppl_wrapped = math.exp(out_wrapped.loss.item())

    original_attns = {}
    for idx in ska_indices:
        wrapper = _get_retrofit_wrapper(model, idx)
        original_attns[idx] = wrapper
        _set_retrofit_wrapper(model, idx, wrapper.orig_attn)

    with torch.no_grad():
        out_base = model(input_ids=input_ids, labels=input_ids)
    ppl_base = math.exp(out_base.loss.item())

    for idx in ska_indices:
        _set_retrofit_wrapper(model, idx, original_attns[idx])

    print(f"  Base model PPL:    {ppl_base:.2f}")
    print(f"  Wrapped (rho=0):   {ppl_wrapped:.2f}")
    diff = abs(ppl_wrapped - ppl_base)
    if diff < 0.1:
        print(f"  PASS — difference {diff:.4f}")
    else:
        print(f"  FAIL — difference {diff:.4f}, wrapper is not transparent")
    return ppl_base, ppl_wrapped


def run_wikitext_eval(model, tokenizer, ska_indices, device, args, rho, label):
    """Evaluate next-token loss/PPL on WikiText with the requested SKA blend rho."""
    print("\n" + "=" * 60)
    print(f"EVAL: {label} (rho={rho:.2f})")
    print("=" * 60)

    for idx in ska_indices:
        _get_retrofit_wrapper(model, idx).set_rho(rho)

    model.eval()
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(ds["text"][:512])
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
    )
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    lm_loss = out.loss.item()
    ppl = math.exp(min(lm_loss, 20))
    print(f"  loss={lm_loss:.4f}, ppl={ppl:.2f}")
    append_metrics_row(
        args=args,
        stage_name=label,
        step=0,
        total_steps=0,
        lm_loss=lm_loss,
        ppl=ppl,
        distill=0.0,
        rho=rho,
    )
    return lm_loss, ppl


def run_base_wikitext_eval(model, tokenizer, ska_indices, device, args, label="base_eval"):
    """Evaluate the original attention path by temporarily bypassing SKA wrappers."""
    print("\n" + "=" * 60)
    print(f"EVAL: {label} (original attention)")
    print("=" * 60)

    original_wrappers = {}
    for idx in ska_indices:
        wrapper = _get_retrofit_wrapper(model, idx)
        original_wrappers[idx] = wrapper
        _set_retrofit_wrapper(model, idx, wrapper.orig_attn)

    model.eval()
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(ds["text"][:512])
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
    )
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    lm_loss = out.loss.item()
    ppl = math.exp(min(lm_loss, 20))
    print(f"  loss={lm_loss:.4f}, ppl={ppl:.2f}")

    for idx, wrapper in original_wrappers.items():
        _set_retrofit_wrapper(model, idx, wrapper)

    append_metrics_row(
        args=args,
        stage_name=label,
        step=0,
        total_steps=0,
        lm_loss=lm_loss,
        ppl=ppl,
        distill=0.0,
        rho=0.0,
    )
    return lm_loss, ppl


# Main

def main():
    p = argparse.ArgumentParser(description="SKA Retrofit Training")

    p.add_argument("--model", default="EleutherAI/pythia-14m")
    p.add_argument("--init_strategy", default="svd_sqrt",
                   choices=[
                       "pca", "svd_sqrt", "svd_noscale", "svd_full",
                       "change_b", "change_a_b", "mlp_log_singular_values",
                       "random_orth",
                   ])
    p.add_argument("--scale_a", type=float, default=1.0)
    p.add_argument("--power_b", type=float, default=0.5)
    p.add_argument("--mlp_state_path", default=None)
    p.add_argument("--mlp_hidden_dim", type=int, default=32)
    p.add_argument("--mlp_residual_scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ska_rank", type=int, default=64,
                   help="Rank of the SKA key/query projection")
    p.add_argument("--ska_chunk_size", type=int, default=64)
    p.add_argument("--power_K", type=int, default=0,
                   help="Number of Koopman power-filter steps")
    p.add_argument("--n_ska_layers", type=int, default=8)
    p.add_argument("--ska_layer_indices", default=None,
                   help="Optional comma-separated explicit attention layer indices, e.g. '2' or '0,2,4'. Overrides --n_ska_layers.")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3, 4, 5])

    p.add_argument("--stage1_steps", type=int, default=3000)
    p.add_argument("--stage2_steps", type=int, default=5000)
    p.add_argument("--stage3_steps", type=int, default=5000)
    p.add_argument("--single_stage_steps", type=int, default=300)
    p.add_argument("--single_stage_distill_weight", type=float, default=0.0)

    p.add_argument("--ska_lr", type=float, default=1e-4)
    p.add_argument("--norm_lr", type=float, default=1e-5)
    p.add_argument("--deltanet_lr", type=float, default=1e-5)
    p.add_argument("--mlp_lr", type=float, default=5e-6)
    p.add_argument("--grad_clip", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--retrieval_ratio", type=float, default=0.0,
                   help="Fraction of synthetic retrieval examples in the training stream")
    p.add_argument("--n_facts", type=int, default=2)
    p.add_argument("--distractor_len", type=int, default=256)
    p.add_argument("--n_queries", type=int, default=2)

    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=1000)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    verify_operator()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, ska_indices = setup_model(args, device)

    pad_id = tokenizer.pad_token_id or 0
    collate = lambda batch: collate_fn(batch, pad_id)

    if args.stage == 0:
        run_stage_0(model, tokenizer, ska_indices, device, args)
        return

    if args.stage == 1:
        print("\nStage 1: Distillation (SKA learns to mimic attention)")
        print("  rho=0, distill_weight=1.0, pure WikiText data")

        groups = build_param_groups(model, args, ska_indices, stage=1)

        dataset = LMRetrievalDataset(
            tokenizer, args.max_seq_len, seed=args.seed,
            retrieval_ratio=0.0,  # pure LM for distillation
        )
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=collate, num_workers=args.num_workers)

        train_loop(model, loader, groups, args, device, ska_indices,
                   total_steps=args.stage1_steps,
                   warmup_steps=args.warmup_steps,
                   rho_start=0.0, rho_end=0.0,
                   distill_weight=1.0,
                   stage_name="1")

    # ---- Stage 2: blended + retrieval (rho 0 -> 0.5) ----
    elif args.stage == 2:
        print("\nStage 2: Blended training with retrieval")
        print(f"  rho: 0.0 -> 0.5, retrieval_ratio={args.retrieval_ratio}")

        if args.retrieval_ratio == 0.0:
            args.retrieval_ratio = 0.3
            print(f"  Auto-set retrieval_ratio to {args.retrieval_ratio}")

        groups = build_param_groups(model, args, ska_indices, stage=2)

        dataset = LMRetrievalDataset(
            tokenizer, args.max_seq_len, seed=args.seed + 1,
            retrieval_ratio=args.retrieval_ratio,
            n_facts=args.n_facts,
            distractor_len=args.distractor_len,
            n_queries=args.n_queries,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=collate, num_workers=args.num_workers)

        train_loop(model, loader, groups, args, device, ska_indices,
                   total_steps=args.stage2_steps,
                   warmup_steps=args.warmup_steps,
                   rho_start=0.0, rho_end=0.5,
                   distill_weight=0.25,
                   stage_name="2")

    # ---- Stage 3: anneal attention away (rho 0.5 -> 1.0) ----
    elif args.stage == 3:
        print("\nStage 3: Anneal attention away")
        print("  rho: 0.5 -> 1.0")

        if args.retrieval_ratio == 0.0:
            args.retrieval_ratio = 0.2

        groups = build_param_groups(model, args, ska_indices, stage=3)

        dataset = LMRetrievalDataset(
            tokenizer, args.max_seq_len, seed=args.seed + 2,
            retrieval_ratio=args.retrieval_ratio,
            n_facts=args.n_facts,
            distractor_len=args.distractor_len,
            n_queries=args.n_queries,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=collate, num_workers=args.num_workers)

        train_loop(model, loader, groups, args, device, ska_indices,
                   total_steps=args.stage3_steps,
                   warmup_steps=args.warmup_steps,
                   rho_start=0.5, rho_end=1.0,
                   distill_weight=0.05,
                   stage_name="3")

    # ---- Stage 4: remove attention, verify ----
    elif args.stage == 4:
        print("\nStage 4: Remove attention, verify pure-SKA PPL")
        run_wikitext_eval(model, tokenizer, ska_indices, device, args, rho=1.0, label="4_pure_ska_eval")

    # ---- Stage 5: single-stage pure-SKA next-token training ----
    elif args.stage == 5:
        print("\nStage 5: Single-stage pure-SKA next-token experiment")
        print("  rho=1.0 from the start")
        print("  loss=WikiText next-token LM loss")
        print("  trainable=SKA parameters only")

        run_base_wikitext_eval(model, tokenizer, ska_indices, device, args, label="base_eval")
        run_wikitext_eval(model, tokenizer, ska_indices, device, args, rho=1.0, label="5_init_eval")

        groups = build_param_groups(model, args, ska_indices, stage=2)

        dataset = LMRetrievalDataset(
            tokenizer, args.max_seq_len, seed=args.seed,
            retrieval_ratio=0.0,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=collate, num_workers=args.num_workers)

        train_loop(model, loader, groups, args, device, ska_indices,
                   total_steps=args.single_stage_steps,
                   warmup_steps=args.warmup_steps,
                   rho_start=1.0, rho_end=1.0,
                   distill_weight=args.single_stage_distill_weight,
                   stage_name="5")

        run_wikitext_eval(model, tokenizer, ska_indices, device, args, rho=1.0, label="5_final_eval")

    print("\nDone.")


if __name__ == "__main__":
    main()
