"""Fine-tune a backbone with a single linear classification head for the 14-class
next-action task. Metric: Macro-F1 (competition metric).

This run: FULL fine-tuning — all backbone weights + the standard single linear head
are trained (AutoModelForSequenceClassification's built-in head is one linear layer).
Higher ceiling than LoRA but more VRAM/time; uses a small LR (2e-5) and gradient
checkpointing. (The LoRA variant lives on the probe-improvements branch.)

Usage:
  python -m src.finetune --model Qwen/Qwen3-Embedding-0.6B
  python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 --epochs 3
Outputs: output/ft_<model>/ (best checkpoint + metrics), appended to output/ft_results.csv.
"""
import argparse
import csv as _csv
import json
import os

# reduce CUDA fragmentation OOMs (must be set before torch initializes CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import (ACTION_GROUPS, ALL_CLASSES, CLASS_TO_ID, GROUP_ID,
                      SERIALIZE_VARIANTS, build_texts, load_samples, serialize,
                      session_fold_indices, split_indices)


def build_dataset(tok, texts, labels, max_len, desc="tokenizing", teacher=None,
                  weights=None):
    """Tokenize once; return a torch Dataset yielding input_ids/attention_mask/labels.

    Tokenizes in chunks with a tqdm bar (works in a terminal and prints periodic
    updates to an sbatch log file).
    teacher: optional (N, num_classes) float array of teacher logits (distillation);
    added to each item as "teacher_logits".
    weights: optional (N,) float array of per-sample loss weights ("weight" key).
    """
    from tqdm.auto import tqdm

    enc = {"input_ids": [], "attention_mask": []}
    chunk = 1000
    for i in tqdm(range(0, len(texts), chunk), desc=desc, unit="k-rows",
                  mininterval=5.0):  # mininterval keeps sbatch logs sparse
        e = tok(texts[i:i + chunk], truncation=True, max_length=max_len, padding=False)
        enc["input_ids"].extend(e["input_ids"])
        enc["attention_mask"].extend(e["attention_mask"])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            item = {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": int(labels[i]),
            }
            if teacher is not None:
                item["teacher_logits"] = teacher[i]
            if weights is not None:
                item["weight"] = float(weights[i])
            return item

    return DS()


def build_dynamic_dataset(tok, samples_sub, labels, max_len, max_hist, hist_dropout, seed,
                          variant="v1"):
    """Train dataset that serializes + tokenizes per __getitem__, so each epoch sees a
    FRESH random history-event drop (static pre-tokenization would corrupt once).
    Per-sample tokenization is ~1ms — negligible next to a full-FT train step."""
    rng = np.random.default_rng(seed)
    var_kw = SERIALIZE_VARIANTS[variant]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            text = serialize(samples_sub[i], max_hist=max_hist,
                             hist_dropout=hist_dropout, rng=rng, **var_kw)
            e = tok(text, truncation=True, max_length=max_len, padding=False)
            return {
                "input_ids": e["input_ids"],
                "attention_mask": e["attention_mask"],
                "labels": int(labels[i]),
            }

    return DS()


def build_ids_dataset(ids_list, labels):
    """Dataset from PRE-tokenized input_ids (E24 method-A --reduced_ids path). Additive:
    used only when --reduced_ids is set; the plain build_dataset path is untouched."""
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            ids = ids_list[i]
            return {"input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": int(labels[i])}

    return DS()


def make_compute_metrics(n_classes):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        mf1 = f1_score(labels, preds, labels=list(range(n_classes)),
                       average="macro", zero_division=0)
        return {"macro_f1": mf1}
    return compute_metrics


def make_dynamics_logger(train_ds, collator, tr_idx, labels, out_path, bs=64):
    """TrainerCallback: after each epoch, dump the model's softmax over the TRAIN set
    (eval mode, no grad, tr order) → npz(probs=(E,N,C), tr, labels). The source for
    cartography (conf×variability) / AUM (margin) / forgetting / EL2N. Assumes no
    hist_dropout (a static train_ds), so the per-epoch predictions are comparable."""
    from transformers import TrainerCallback

    class DynamicsLogger(TrainerCallback):
        def __init__(self):
            self.epoch_probs = []

        def on_epoch_end(self, args, state, control, model=None, **kw):
            from torch.utils.data import DataLoader
            was_training = model.training
            model.eval()
            probs = np.zeros((len(train_ds), model.config.num_labels), dtype=np.float32)
            k = 0
            with torch.no_grad():
                for batch in DataLoader(train_ds, batch_size=bs, shuffle=False,
                                        collate_fn=collator):
                    ins = {kk: v.to(model.device) for kk, v in batch.items() if kk != "labels"}
                    p = torch.softmax(model(**ins).logits.float(), -1).cpu().numpy()
                    probs[k:k + len(p)] = p
                    k += len(p)
            self.epoch_probs.append(probs)
            if was_training:
                model.train()

        def on_train_end(self, args, state, control, **kw):
            arr = np.stack(self.epoch_probs)  # (E, N, C)
            np.savez(out_path, probs=arr, tr=np.asarray(tr_idx),
                     labels=np.asarray(labels))
            logger.info(f"train dynamics -> {out_path}  {arr.shape}")

    return DynamicsLogger()


def make_best_snapshot(metric="eval_macro_f1"):
    """RAM replacement for save_strategy='epoch' + load_best_model_at_end: after each
    epoch eval, if the metric improved, snapshot the state_dict to CPU (~1-2s PCIe
    copy). The training loop never writes checkpoints to the (slow, shared) disk;
    the best weights are loaded back from RAM after train() and written once."""
    from transformers import TrainerCallback

    class BestSnapshot(TrainerCallback):
        def __init__(self):
            self.best_f1 = None
            self.best_epoch = None
            self.best_state = None

        def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
            f1 = (metrics or {}).get(metric)
            if f1 is None or (self.best_f1 is not None and f1 <= self.best_f1):
                return
            self.best_f1, self.best_epoch = f1, state.epoch
            self.best_state = {k: v.detach().to("cpu", copy=True)
                               for k, v in model.state_dict().items()}
            logger.info(f"new best {metric}={f1:.4f} @ epoch {state.epoch:.1f} "
                        "-> weights snapshotted to RAM (no disk write)")

    return BestSnapshot()


def _macro_f1(logits, labels, bias):
    preds = np.argmax(logits + bias, axis=1)
    return f1_score(labels, preds, labels=list(range(logits.shape[1])),
                    average="macro", zero_division=0)


def calibrate_logit_bias(logits, labels, rounds=50, grid=None):
    """Post-hoc per-class logit bias tuned on val to maximize Macro-F1 (73.07 trick).

    Coordinate ascent: for each class, try a grid of additive biases and keep the
    value that improves val Macro-F1. Returns (bias_vector, base_f1, tuned_f1).
    rounds=50/step .02: best honest performer in analysis/calibration_methods.py
    (2-fold: 0.7385 vs 0.7376 current, vs 0.7340 for overfit-prone matrix scaling).
    Converges early via the no-improvement break, so cost stays ~1-2 min.
    """
    if grid is None:
        grid = np.round(np.arange(-2.0, 2.001, 0.02), 3)
    n = logits.shape[1]
    bias = np.zeros(n, dtype=np.float32)
    base = _macro_f1(logits, labels, bias)
    best = base
    for _ in range(rounds):
        improved = False
        for c in range(n):
            cur = bias[c]
            best_b, best_f1 = cur, best
            for b in grid:
                bias[c] = b
                f1 = _macro_f1(logits, labels, bias)
                if f1 > best_f1:
                    best_f1, best_b = f1, b
            bias[c] = best_b
            if best_f1 > best:
                best, improved = best_f1, True
        if not improved:
            break
    return bias, base, best


import re as _re

_LAYER_RE = _re.compile(r"\.layers?\.(\d+)\.")


def _layer_depth(model):
    """Number of encoder layers, inferred from parameter names."""
    idxs = [int(m.group(1)) for n, _ in model.named_parameters() if (m := _LAYER_RE.search(n))]
    return max(idxs) + 1 if idxs else 0


def build_llrd_optimizer(model, base_lr, decay, optim_name, weight_decay):
    """Optimizer with layer-wise LR decay: layer i gets base_lr * decay^(depth-1-i);
    embeddings get one step lower; head/pooler get full base_lr."""
    depth = _layer_depth(model)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = _LAYER_RE.search(n)
        if m:
            lr = base_lr * decay ** (depth - 1 - int(m.group(1)))
        elif "embed" in n:
            lr = base_lr * decay ** depth
        else:                       # head / pooler / final norms
            lr = base_lr
        groups.setdefault(round(lr, 12), []).append(p)
    param_groups = [{"params": ps, "lr": lr} for lr, ps in groups.items()]
    logger.info(f"LLRD: {len(param_groups)} LR groups, "
                f"min={min(g['lr'] for g in param_groups):.2e} max={base_lr:.2e}")
    if optim_name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(param_groups, lr=base_lr, scale_parameter=False,
                         relative_step=False, warmup_init=False, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def reinit_top_layers(model, n):
    """Re-initialize the top n encoder layers (fresh start for task-specific tops)."""
    layer_lists = [mod for name, mod in model.named_modules()
                   if name.endswith(("encoder.layer", "encoder.layers", "model.layers"))]
    assert layer_lists, "could not locate the encoder layer list for --reinit_layers"
    layers = layer_lists[0]
    for layer in list(layers)[-n:]:
        layer.apply(model._init_weights)
    logger.info(f"re-initialized top {n} encoder layers")


def prune_layers(model, keep):
    """Structured depth pruning. `keep` is either:
      - an int N: keep N evenly-spaced encoder layers (always incl. first and last), or
      - a list/sequence of ints: keep exactly those layer indices (e.g. a ShortGPT
        Block-Influence selection). Order is sorted; duplicates dropped.
    config.num_hidden_layers is updated so the pruned checkpoint reloads with
    from_pretrained. Pair with --init_from to prune a trained model and recover-FT."""
    hit = next(((name, mod) for name, mod in model.named_modules()
                if name.endswith(("encoder.layer", "encoder.layers", "model.layers"))), None)
    assert hit, "could not locate the encoder layer list for --keep_layers"
    list_name, layers = hit
    depth = len(layers)
    if isinstance(keep, (list, tuple, np.ndarray)):
        idx = sorted(set(int(i) for i in keep))
        assert idx and idx[0] >= 0 and idx[-1] < depth, \
            f"--keep_layer_idx {idx} out of range for depth {depth}"
    else:
        assert keep < depth, f"--keep_layers {keep} >= model depth {depth}"
        idx = sorted(set(np.round(np.linspace(0, depth - 1, keep)).astype(int).tolist()))
    parent = model.get_submodule(list_name.rsplit(".", 1)[0])
    setattr(parent, list_name.rsplit(".", 1)[1],
            torch.nn.ModuleList([layers[i] for i in idx]))
    model.config.num_hidden_layers = len(idx)
    logger.info(f"pruned encoder depth {depth} -> {len(idx)} (kept layers {idx})")


class ExtrasCollator:
    """Wrap DataCollatorWithPadding: non-text keys (teacher_logits, weight) can't go
    through tokenizer.pad, so pop them, pad the rest, and re-attach as tensors."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        teacher = [f.pop("teacher_logits", None) for f in features]
        weight = [f.pop("weight", None) for f in features]
        batch = self.base(features)
        if teacher[0] is not None:
            batch["teacher_logits"] = torch.tensor(np.stack(teacher), dtype=torch.float32)
        if weight[0] is not None:
            batch["weight"] = torch.tensor(weight, dtype=torch.float32)
        return batch


def prune_ffn(model, keep_ratio):
    """Structured width pruning of every FFN: keep the top keep_ratio intermediate
    neurons by |W_in row| * |W_out col| (throughput proxy), slice both matrices.
    config.intermediate_size is updated so the checkpoint reloads cleanly."""
    import torch.nn as nn

    k = None
    n_pruned = 0
    for name, layer in model.named_modules():
        if not (hasattr(layer, "intermediate") and hasattr(layer, "output")):
            continue
        if not (hasattr(layer.intermediate, "dense") and hasattr(layer.output, "dense")):
            continue
        w_in, w_out = layer.intermediate.dense, layer.output.dense
        inter = w_in.out_features
        k = max(1, int(round(inter * keep_ratio)))
        score = w_in.weight.data.norm(dim=1) * w_out.weight.data.norm(dim=0)
        idx = torch.topk(score, k).indices.sort().values
        new_in = nn.Linear(w_in.in_features, k)
        new_in.weight.data = w_in.weight.data[idx].clone()
        new_in.bias.data = w_in.bias.data[idx].clone()
        new_out = nn.Linear(k, w_out.out_features)
        new_out.weight.data = w_out.weight.data[:, idx].clone()
        new_out.bias.data = w_out.bias.data.clone()
        layer.intermediate.dense, layer.output.dense = new_in, new_out
        n_pruned += 1
    assert n_pruned, "no intermediate/output FFN pairs found for --ffn_keep"
    model.config.intermediate_size = k
    logger.info(f"FFN width pruned in {n_pruned} layers: {inter} -> {k} neurons")


def prune_attn_heads(model, keep_ratio):
    """Drop the least important attention heads per layer (|W_v head| * |W_o head|
    proxy) via HF's model.prune_heads; config.pruned_heads records the surgery so
    from_pretrained reloads correctly. Call AFTER prune_layers (indices re-read)."""
    n_heads = model.config.num_attention_heads
    k = max(1, int(round(n_heads * keep_ratio)))
    to_prune = {}
    for name, mod in model.named_modules():
        if not name.endswith("attention.self"):
            continue
        layer_idx = int(name.split(".layer.")[1].split(".")[0])
        hd = mod.attention_head_size
        out = model.get_submodule(name.rsplit(".self", 1)[0] + ".output.dense")
        scores = [(mod.value.weight.data[h * hd:(h + 1) * hd].norm()
                   * out.weight.data[:, h * hd:(h + 1) * hd].norm()).item()
                  for h in range(n_heads)]
        to_prune[layer_idx] = sorted(range(n_heads), key=scores.__getitem__)[:n_heads - k]
    assert to_prune, "no attention.self modules found for --heads_keep"
    model.prune_heads(to_prune)
    logger.info(f"attention heads pruned in {len(to_prune)} layers: {n_heads} -> {k}")


def replace_head(model, n_layers, act="gelu"):
    """Swap the classification head for an n_layers MLP (act nonlinearity, dropout
    between).

    Handles both head styles:
      - Qwen3-style `model.score`: applied per-token, the model selects the last
        non-pad position afterwards -> plain Sequential works token-wise.
      - XLM-R-style `model.classifier`: called with the full sequence output and
        pools [CLS] internally -> wrapper reproduces the features[:, 0] selection.
        (NOTE: the stock XLM-R head is already dense+tanh+out_proj, i.e. ~2 layers;
        n_layers=3 is the genuinely deeper variant there.)
    """
    import torch.nn as nn

    hidden = model.config.hidden_size
    p = getattr(model.config, "classifier_dropout", None) or 0.1
    num_labels = model.config.num_labels

    act_layer = {"gelu": nn.GELU, "tanh": nn.Tanh}[act]

    def mlp():
        layers, in_dim = [], hidden
        for _ in range(n_layers - 1):
            layers += [nn.Dropout(p), nn.Linear(in_dim, hidden), act_layer()]
            in_dim = hidden
        layers += [nn.Dropout(p), nn.Linear(in_dim, num_labels)]
        return nn.Sequential(*layers)

    if hasattr(model, "score"):
        model.score = mlp()
        new = model.score
    elif hasattr(model, "classifier"):
        class CLSHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = mlp()

            def forward(self, features, **kwargs):
                x = features[:, 0, :] if features.dim() == 3 else features
                return self.mlp(x)

        model.classifier = CLSHead()
        new = model.classifier
    else:
        raise AssertionError("no .score or .classifier head found for --head_layers")
    new.apply(model._init_weights)
    # recorded so downstream loaders can reconstruct: plain from_pretrained rebuilds
    # the STOCK head and silently drops classifier.mlp.* weights. To reload a custom-
    # head checkpoint: build the base model, call replace_head(m, **config.custom_head
    # values), then load the safetensors state dict.
    model.config.custom_head = {"layers": n_layers, "act": act}
    logger.info(f"replaced classification head with {n_layers}-layer {act} MLP (dropout {p})")


def make_distill_trainer(trainer_cls, alpha, temperature, ce_mode="ce",
                         label_smoothing=0.1):
    import torch.nn.functional as F

    class _DistillTrainer(trainer_cls):
        """loss = (1-alpha)*CE(student, y) + alpha*T^2*KL(teacher_T || student_T).
        Eval batches carry no teacher_logits -> plain CE there.
        ce_mode="ls" (E30 fix): the CE term uses label smoothing — E26 path-1B lost
        the LS gain (~0.0107) because KD replaced the smoothed CE wholesale."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            teacher_logits = inputs.pop("teacher_logits", None)
            outputs = model(**inputs)
            if ce_mode == "ls":
                loss = F.cross_entropy(outputs.logits, inputs["labels"],
                                       label_smoothing=label_smoothing)
            else:
                loss = outputs.loss
            if teacher_logits is not None:
                kd = F.kl_div(
                    F.log_softmax(outputs.logits.float() / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature ** 2
                loss = (1.0 - alpha) * loss + alpha * kd
            return (loss, outputs) if return_outputs else loss

    return _DistillTrainer


def make_aux_trainer(base_cls, rdrop=0.0, supcon=None):
    """Trainer with optional boundary-sharpening losses (all train-only, eval is
    plain CE):

      per-sample weights  — "weight" batch key (from --hard_boundary mining):
                            loss = sum(w_i * CE_i) / sum(w_i)
      rdrop > 0           — second forward with fresh dropout masks;
                            loss = mean CE of both + rdrop * symmetric KL
      supcon dict         — SupCon aux on the pooled embedding via model.supcon_proj
                            (lam, tau, group_w, queue): cross-batch memory queue
                            supplies positives, so no batch-composition surgery;
                            group_w > 1 upweights SAME-GROUP negatives (read vs grep)
                            in the denominator — the boundaries we actually lose on.
    """
    import torch.nn.functional as F

    class _AuxTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weight = inputs.pop("weight", None)
            # hidden states only during training — in eval they'd leak into the
            # Trainer's prediction tuple and break compute_metrics
            need_h = supcon is not None and model.training
            outputs = model(**inputs, output_hidden_states=need_h)
            loss = outputs.loss
            if not model.training:               # eval path: plain CE
                return (loss, outputs) if return_outputs else loss
            labels = inputs["labels"]
            if weight is not None:
                ce = F.cross_entropy(outputs.logits.float(), labels, reduction="none")
                loss = (ce * weight).sum() / weight.sum()
            if rdrop:
                out2 = model(**inputs)           # fresh dropout masks
                lp1 = F.log_softmax(outputs.logits.float(), dim=-1)
                lp2 = F.log_softmax(out2.logits.float(), dim=-1)
                kl = 0.5 * (F.kl_div(lp1, lp2, log_target=True, reduction="batchmean")
                            + F.kl_div(lp2, lp1, log_target=True, reduction="batchmean"))
                loss = 0.5 * (loss + out2.loss) + rdrop * kl
            if supcon is not None:
                loss = loss + supcon["lam"] * self._supcon_loss(model, outputs, inputs)
            return (loss, outputs) if return_outputs else loss

        def _supcon_loss(self, model, outputs, inputs):
            h = outputs.hidden_states[-1]
            if hasattr(model, "score"):          # Qwen-style: last non-pad token
                last = inputs["attention_mask"].sum(1) - 1
                pooled = h[torch.arange(h.shape[0], device=h.device), last]
            else:                                # XLM-R-style: [CLS]
                pooled = h[:, 0]
            with torch.autocast(pooled.device.type, enabled=False):  # fp32 for stability
                z = F.normalize(model.supcon_proj(pooled.float()), dim=-1)
            y = inputs["labels"]
            if not hasattr(self, "_xbm_z"):      # lazy cross-batch memory init
                self._xbm_z = z.new_zeros((0, z.shape[1]))
                self._xbm_y = y.new_zeros((0,))
                self._gid = torch.tensor(GROUP_ID, device=y.device)
            bank_z = torch.cat([z.detach(), self._xbm_z])
            bank_y = torch.cat([y, self._xbm_y])
            sim = z @ bank_z.T / supcon["tau"]                       # (B, B+Q)
            b = z.shape[0]
            self_mask = torch.zeros_like(sim, dtype=torch.bool)
            self_mask[:, :b] = torch.eye(b, dtype=torch.bool, device=sim.device)
            pos = (bank_y[None, :] == y[:, None]) & ~self_mask
            w = torch.ones_like(sim)
            same_group = self._gid[bank_y][None, :] == self._gid[y][:, None]
            w[same_group & ~pos] = supcon["group_w"]                 # hard negatives
            w[self_mask] = 0.0
            smax = sim.max(1, keepdim=True).values
            denom = (w * torch.exp(sim - smax)).sum(1, keepdim=True)
            log_prob = sim - smax - torch.log(denom + 1e-12)
            n_pos = pos.sum(1)
            per_anchor = -(log_prob * pos).sum(1) / n_pos.clamp(min=1)
            loss = per_anchor[n_pos > 0].mean() if (n_pos > 0).any() else sim.sum() * 0.0
            # FIFO queue update (detached)
            self._xbm_z = torch.cat([z.detach(), self._xbm_z])[: supcon["queue"]]
            self._xbm_y = torch.cat([y, self._xbm_y])[: supcon["queue"]]
            return loss

    return _AuxTrainer


def classification_loss(logits, labels, mode="ce", focal_gamma=2.0, label_smoothing=0.1,
                        class_weight=None, log_prior=None, la_tau=1.0):
    """Head classification loss for the 14-class problem, computed in fp32 from the
    head logits (N, C) + integer labels (N,). This REPLACES the model's internal CE
    (it is NOT an added aux term).

      mode="ce"    plain cross-entropy — numerically equal to F.cross_entropy and to
                   the HF model's stock CrossEntropyLoss (the default path never
                   routes here; see make_loss_trainer).
      mode="focal" multiclass focal loss, mean-reduced:
                   FL = mean[ -(1 - p_t)^gamma * log p_t ].
      mode="ls"    cross-entropy with label smoothing (torch >= 1.10).
      mode="wce"   class-weighted CE — weighted CE with per-class weight
                   class_weight (K,) = N / (K * count_k) ('balanced', sklearn); the
                   most literal macro-F1 surrogate (every class weighted equally).
      mode="la"    logit-adjusted loss (Menon et al., 2021, ICLR): CE on
                   logits + la_tau * log_prior, where log_prior (K,) = log(count_k/N).
                   The consistent surrogate for balanced/macro error. NOT calibration:
                   priors come from the TRAIN split, inference stays raw argmax.
    """
    import torch.nn.functional as F
    logits = logits.float()
    if mode == "ce":
        return F.cross_entropy(logits, labels)
    if mode == "ls":
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    if mode == "focal":
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)   # log p_t
        pt = logpt.exp()
        return (-((1.0 - pt) ** focal_gamma) * logpt).mean()
    if mode == "wce":
        assert class_weight is not None, "wce needs class_weight"
        w = class_weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, labels, weight=w)
    if mode == "la":
        assert log_prior is not None, "la needs log_prior"
        lp = log_prior.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits + la_tau * lp, labels)
    raise ValueError(f"unknown --loss mode: {mode!r}")


def make_loss_trainer(base_cls, mode, focal_gamma=2.0, label_smoothing=0.1,
                      class_weight=None, log_prior=None, la_tau=1.0):
    """Trainer that swaps the head classification loss for a macro-F1-targeted
    variant (focal / label-smoothing / class-weighted-CE / logit-adjusted). mode='ce'
    is never wrapped by the caller, so the stock Trainer loss path stays byte-identical
    to before. class_weight / log_prior are precomputed from the TRAIN split."""

    class _LossTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = classification_loss(outputs.logits, inputs["labels"],
                                       mode, focal_gamma, label_smoothing,
                                       class_weight, log_prior, la_tau)
            return (loss, outputs) if return_outputs else loss

    return _LossTrainer


def mine_boundary_weights(model, tok, texts_tr, max_len, weight, margin, device):
    """One inference pass over the train split with the --init_from checkpoint:
    samples whose top-2 logits are BOTH in the same confusion group and closer
    than `margin` get loss weight `weight` (they sit on the boundary we lose on);
    everything else keeps 1.0."""
    gid = np.array(GROUP_ID)
    weights = np.ones(len(texts_tr), dtype=np.float32)
    order = sorted(range(len(texts_tr)), key=lambda i: -len(texts_tr[i]))  # pad less
    model.to(device).eval()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad():
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            enc = tok([texts_tr[i] for i in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                lg = model(**enc).logits.float().cpu().numpy()
            top2 = np.argsort(lg, axis=1)[:, -2:]                    # [second, first]
            marg = np.take_along_axis(lg, top2, 1)
            flag = (gid[top2[:, 0]] == gid[top2[:, 1]]) & ((marg[:, 1] - marg[:, 0]) < margin)
            for j, i in enumerate(idx):
                if flag[j]:
                    weights[i] = weight
    frac = float((weights > 1).mean())
    logger.info(f"boundary mining: {frac:.1%} of train samples flagged "
                f"(weight {weight}, margin {margin})")
    return weights


def add_action_tokens(tok, model):
    """Add the 14 action names + serialization markers as ATOMIC tokens.

    Without this, bge-m3's sentencepiece splits every action name into 3-7 pieces
    ('lint_or_typecheck' -> 7). Each new token gets ONE embedding row, initialized
    to the MEAN of the piece embeddings it replaces — training starts from the
    compositional representation instead of random. The tokenizer must be saved
    with the run (save_pretrained) and shipped at inference, or the ids shift.
    """
    new_tokens = list(ALL_CLASSES) + ["ACTION", "USER:", "PROMPT:"]
    piece_ids = {t: tok(t, add_special_tokens=False)["input_ids"] for t in new_tokens}
    n_added = tok.add_tokens(new_tokens)
    # when continuing from a checkpoint that ALREADY has the extra rows
    # (--init_from a spectok run), resize is a no-op and the trained embeddings
    # must NOT be re-initialized — only init rows that did not exist before.
    old_rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    inited = 0
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        for t in new_tokens:
            new_id = tok.convert_tokens_to_ids(t)
            if new_id >= old_rows:
                w[new_id] = w[piece_ids[t]].mean(dim=0)
                inited += 1
    logger.info(f"atomic action/marker tokens: {n_added} added to tokenizer "
                f"(vocab -> {len(tok)}), {inited} embedding rows mean-of-pieces "
                f"initialized ({n_added - inited} pre-trained rows kept)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="context", choices=["context", "prompt"])
    ap.add_argument("--max_hist", type=int, default=0,
                    help="cap history to last N events; 0 = full history (no cap)")
    ap.add_argument("--serialize", default="v1", choices=sorted(SERIALIZE_VARIANTS),
                    help="input serialization variant (see data.SERIALIZE_VARIANTS); "
                         "v1 = the hist0-baseline format")
    ap.add_argument("--special_tokens", action="store_true",
                    help="add the 14 action names + ACTION/USER:/PROMPT: markers as "
                         "single tokens (mean-of-pieces embedding init); the run-dir "
                         "tokenizer must then be used at inference")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)          # full-FT needs a small LR
    ap.add_argument("--batch_size", type=int, default=4)       # full-FT is VRAM-heavy (~11GB GPU)
    ap.add_argument("--grad_accum", type=int, default=4)       # effective batch 16
    ap.add_argument("--grad_checkpointing", default="auto", choices=["auto", "on", "off"],
                    help="recompute activations in backward to save VRAM (~30-40%% slower). "
                         "auto = on only for large (>400M) or long-seq (>512) configs, off "
                         "otherwise (e.g. granite-311m@512 fits a 3090 fine without it — "
                         "result-neutral, pure recompute). on/off force it.")
    ap.add_argument("--attn_impl", default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager"],
                    help="attention kernel. auto = flash_attention_2 if flash-attn is "
                         "installed else sdpa. ModernBERT (granite) gains most from flash "
                         "(unpadding); needs `pip install flash-attn`.")
    ap.add_argument("--group_by_length", action="store_true",
                    help="batch similar-length samples together to cut padding waste "
                         "(big speedup at small batch). NOTE: changes batch composition -> "
                         "not byte-identical to a non-grouped baseline (effect on macro-F1 "
                         "is typically <0.001, but keep it consistent within a comparison).")
    ap.add_argument("--optim", default="adamw_torch",
                    help="optimizer. adamw_torch (default, best quality) or sgd "
                         "(zero optimizer state -> fits bigger models like Qwen3 full-FT). "
                         "SGD usually needs a higher --lr.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="LR warmup fraction of total steps (teammate's recipe uses 0.1)")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                    help="AdamW weight decay (was hardcoded 0.01 before E28; default keeps "
                         "the old behavior)")
    ap.add_argument("--session_fold", type=int, default=-1,
                    help=">=0: replace the row-stratified split with the teammate's "
                         "session-grouped StratifiedGroupKFold protocol (leakage-free; "
                         "val = this fold's ~20%% of sessions). Incompatible with "
                         "--full_data, which would re-leak sessions into eval.")
    ap.add_argument("--session_splits", type=int, default=5,
                    help="fold count for --session_fold (his N_SPLITS=5)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16"],
                    help="mixed-precision dtype; auto = bf16 when supported else fp16. "
                         "fp16 forces the teammate's recipe (his notebook trains fp16)")
    ap.add_argument("--init_seed", type=int, default=-1,
                    help="seed for init/shuffle/dropout ONLY (head init, dataloader order, "
                         "dropout); the train/val SPLIT stays pinned to --seed. -1 = use "
                         "--seed (behavior unchanged). Set it to vary training stochasticity "
                         "while keeping the split FIXED — for clean same-split model soups (E12).")
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--results_name", default="ft_results.csv",
                    help="results CSV filename; give each array task a unique one to "
                         "avoid concurrent-append races")
    ap.add_argument("--limit", type=int, default=0, help="cap train+val size (0=all); for quick tests")
    ap.add_argument("--init_from", default="",
                    help="path to a fine-tuned checkpoint dir to CONTINUE training from "
                         "(weights load from here; --model still names the tokenizer/base)")
    ap.add_argument("--full_data", action="store_true",
                    help="fold 75%% of the val split into training; the remaining 25%% "
                         "(stratified) stays held out for eval + calibration. Safe to combine "
                         "with --init_from: the checkpoint only ever saw the train split")
    ap.add_argument("--all_data", action="store_true",
                    help="train on EVERY sample with NOTHING held out (final-submission "
                         "retrains, E28). No honest eval exists: per-epoch eval is skipped, "
                         "best-epoch selection is disabled, the LAST epoch is the final "
                         "artifact, val_macro_f1 is recorded as nan — epochs must come from "
                         "a prior search. Not combinable with --full_data/--session_fold/"
                         "--calibrate")
    ap.add_argument("--keep_indices", default="",
                    help="path to a .npy of ABSOLUTE sample indices (into the 70k) — restrict "
                         "TRAINING to tr ∩ these (drop-noisy coreset). Val slice unchanged")
    ap.add_argument("--log_dynamics", default="",
                    help="path to write per-epoch TRAIN dynamics npz (softmax over the train "
                         "set each epoch) for cartography / AUM / forgetting / EL2N scoring")
    ap.add_argument("--reduced_ids", default="",
                    help="E24 method-A ONLY (default off). Path to a_reduced_*.npz (ids_flat + "
                         "lengths, aligned to the 70k). When set, train/val consume these "
                         "PRE-tokenized reduced input_ids instead of serializing+tokenizing texts; "
                         "everything else (split/recipe/init) is unchanged. Plain static path only "
                         "(no hist_dropout/distill/weights).")
    ap.add_argument("--tag", default="",
                    help="suffix for the run dir (ft_<model>_<tag>) so reruns don't clobber "
                         "earlier checkpoints of the same model")
    ap.add_argument("--keep_checkpoints", type=int, default=1,
                    help="1 (default): NO per-epoch disk checkpoints — the best epoch's "
                         "weights are snapshotted to CPU RAM and written ONCE at the end "
                         "(the shared disk is slow; epoch saves used to stall the GPU). "
                         ">1: legacy per-epoch disk checkpoints (save_total_limit=N, "
                         "model-only) for post-hoc SWA averaging (analysis/swa_average.py)")
    ap.add_argument("--save_dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                    help="dtype of the FINAL saved checkpoint (training/eval stay fp32). "
                         "fp16 default = the DACON submission runtime precision, and half "
                         "the write volume; fp32 = the old byte-exact behavior")
    ap.add_argument("--calibrate", action="store_true",
                    help="run post-hoc logit-bias calibration on val and write "
                         "logit_bias.json (legacy default). OFF by default — project "
                         "decision is raw logits; run pre-submission if ever needed")
    ap.add_argument("--llrd", type=float, default=0.0,
                    help="layer-wise LR decay factor (e.g. 0.9): layer i gets lr*decay^(depth-i); "
                         "embeddings lowest, head full lr. 0 = off (uniform lr)")
    ap.add_argument("--reinit_layers", type=int, default=0,
                    help="re-initialize the top N encoder layers before training (retrieval-"
                         "pretrained tops may transfer worse than a fresh start)")
    ap.add_argument("--hist_dropout", type=float, default=0.0,
                    help="training-time augmentation: drop each history event with this "
                         "probability, re-drawn every epoch (val is never dropped)")
    ap.add_argument("--keep_layers", type=int, default=0,
                    help="depth-prune the encoder to N evenly-spaced layers before training "
                         "(0 = off). Pair with --init_from for prune-then-recover")
    ap.add_argument("--keep_layer_idx", default="",
                    help="depth-prune to an EXPLICIT comma-separated set of layer indices "
                         "(e.g. ShortGPT Block-Influence pick '0,1,2,...'). Overrides "
                         "--keep_layers. Pair with --init_from for prune-then-recover")
    ap.add_argument("--distill_from", default="",
                    help="path to a .npz with key 'logits' of shape (n_samples, 14) in "
                         "load_samples order — teacher logits for distillation "
                         "(dump with src/dump_logits.py)")
    ap.add_argument("--distill_alpha", type=float, default=0.5,
                    help="weight of the KD term: loss = (1-a)*CE + a*T^2*KL")
    ap.add_argument("--distill_T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--head_layers", type=int, default=0,
                    help="replace the classification head with an N-layer GELU MLP "
                         "(0 = keep the model's stock head)")
    ap.add_argument("--head_act", default="gelu", choices=["gelu", "tanh"],
                    help="nonlinearity for the --head_layers MLP")
    ap.add_argument("--ffn_keep", type=float, default=1.0,
                    help="structured width pruning: keep this fraction of FFN "
                         "intermediate neurons per layer (1.0 = off)")
    ap.add_argument("--heads_keep", type=float, default=1.0,
                    help="structured width pruning: keep this fraction of attention "
                         "heads per layer (1.0 = off)")
    ap.add_argument("--factor_ffn", type=int, default=0,
                    help="low-rank factorize every FFN projection (gate/up/down) to "
                         "this rank via whitened-SVD init (0 = off; E4 recovery-FT). "
                         "Pair with --init_from to factorize a trained checkpoint and "
                         "recover; records config.factored_ffn for reload")
    ap.add_argument("--factor_calib", type=int, default=256,
                    help="#train samples for the whitened-SVD calibration pass "
                         "(--factor_ffn); E3 used 256")
    ap.add_argument("--rdrop", type=float, default=0.0,
                    help="R-Drop: second forward pass with fresh dropout, symmetric-KL "
                         "consistency penalty with this weight (0 = off). Noise-"
                         "compatible boundary sharpening; ~doubles train compute")
    ap.add_argument("--supcon", type=float, default=0.0,
                    help="weight of a supervised-contrastive auxiliary loss on the "
                         "pooled embedding (0 = off). Uses a cross-batch memory queue "
                         "for positives; projection head is train-only (not shipped)")
    ap.add_argument("--supcon_tau", type=float, default=0.1, help="SupCon temperature")
    ap.add_argument("--supcon_group_w", type=float, default=2.0,
                    help="upweight same-group negatives in the SupCon denominator "
                         "(read-vs-grep style hard negatives); 1.0 = uniform")
    ap.add_argument("--supcon_queue", type=int, default=4096,
                    help="cross-batch memory size (embeddings kept as extra pos/neg)")
    ap.add_argument("--hard_boundary", type=float, default=0.0,
                    help="hard-example mining (needs --init_from): upweight train "
                         "samples whose top-2 checkpoint logits are same-group and "
                         "closer than --boundary_margin to this loss weight (0 = off)")
    ap.add_argument("--boundary_margin", type=float, default=2.0,
                    help="logit-margin threshold for --hard_boundary mining")
    ap.add_argument("--pair", default="",
                    help="subset specialist mode: comma-list of 2+ classes — train "
                         "only on their samples with an N-class head (init the "
                         "encoder from --init_from). 2 classes = deferral pair; a "
                         "full group = hierarchical per-group specialist")
    ap.add_argument("--group_task", action="store_true",
                    help="4-class router task: labels collapsed to confusion groups "
                         "(explore/edit/execute/noncode); all samples kept")
    ap.add_argument("--zero_history", action="store_true",
                    help="first-step specialist: keep only zero-history samples "
                         "(step==1, no prior actions). Full 14-class head; filters "
                         "train+val to the no-history slice after the split")
    ap.add_argument("--strip_history", action="store_true",
                    help="augmented first-step specialist: train on ALL samples with "
                         "their history STRIPPED (10x data, but labels were chosen "
                         "given history). Eval stays on the REAL zero-history val, so "
                         "it's directly comparable to --zero_history")
    ap.add_argument("--truncation_side", default="right", choices=["right", "left"],
                    help="left = drop the OLDEST tokens when over max_len, keeping "
                         "recent history + the PROMPT line (which sits at the end). "
                         "For 512-cap models whose group signal is the prompt")
    ap.add_argument("--lora", type=int, default=0,
                    help="train a LoRA adapter of this rank instead of full FT "
                         "(alpha=2r, dropout .05, q/v projections + head). Pair with "
                         "--init_from: the base stays frozen, checkpoints hold only "
                         "adapter+head — built for shared-backbone specialist packs")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal", "ls", "wce", "la"],
                    help="head classification loss — REPLACES cross-entropy (not an "
                         "aux term). ce = plain CE (default; path unchanged); focal = "
                         "multiclass focal loss (see --focal_gamma); ls = CE with "
                         "label smoothing (see --label_smoothing); wce = class-weighted "
                         "CE ('balanced' inverse-freq weights); la = logit-adjusted loss "
                         "(train-prior log-shift, see --la_tau). Not combinable "
                         "with --distill_from/--rdrop/--supcon/--hard_boundary")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="focusing parameter gamma for --loss focal")
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="smoothing epsilon for --loss ls")
    ap.add_argument("--la_tau", type=float, default=1.0,
                    help="logit-adjustment strength tau for --loss la "
                         "(logits + tau*log_prior; 1.0 = the consistent setting)")
    ap.add_argument("--ltp_final_threshold", type=float, default=0.0,
                    help="E18 Learned Token Pruning (granite/ModernBERT only): "
                         "final_token_threshold for the absolute-threshold pruner "
                         "(0 = off). >0 enables SOFT-mask LTP training — per-layer "
                         "learnable thresholds ramp base=final*i/L, soft mask "
                         "sigmoid((attn_received - thr)/T) scales each layer output, plus "
                         "--ltp_lambda*sum(mask.mean()) sparsity term. Forces "
                         "attn_impl=eager + grad_checkpointing off. Not combinable with "
                         "--lora/--distill_from/--rdrop/--supcon/--hard_boundary/--llrd. "
                         "See src/ltp_modeling.py")
    ap.add_argument("--ltp_lambda", type=float, default=0.0,
                    help="LTP sparsity regularizer weight (higher = prune more tokens)")
    ap.add_argument("--ltp_temperature", type=float, default=1e-3,
                    help="LTP soft-mask sigmoid temperature")
    ap.add_argument("--ltp_lr_threshold", type=float, default=0.0,
                    help="separate LR for the LTP threshold params (0 = use --lr); the "
                         "LTP reference trains thresholds with their own optimizer LR")
    ap.add_argument("--ltp_temp_start", type=float, default=None,
                    help="LTP soft-mask temperature at the START of training (default = "
                         "--ltp_temperature, i.e. no anneal). LINEARLY annealed to "
                         "--ltp_temp_end over the run. Start soft (e.g. 0.05 ~10x the "
                         "attention-score scale) for gentle masks + O(1/T)~20 gradients so "
                         "a trained init eases in instead of exploding at fixed small T")
    ap.add_argument("--ltp_temp_end", type=float, default=None,
                    help="LTP soft-mask temperature at the END of training (default = "
                         "--ltp_temperature). Set ~the attention-score scale (e.g. 0.005) "
                         "for sharp near-hard masks that match hard-mode deployment. When "
                         "temp_start==temp_end the temperature is fixed (back-compat)")
    ap.add_argument("--ltp_hard_recover", action="store_true",
                    help="E18 fixed-threshold HARD-DROP recovery-FT (recommended over soft): "
                         "train in HARD mode (actually drop tokens below the fixed ramp "
                         "threshold base[i]=final*i/L), FREEZE the thresholds (no learnable "
                         "deltas, no --ltp_lr_threshold group, no sparsity regularizer), and "
                         "recover the WEIGHTS only. No temperature/soft-mask -> no 1/T "
                         "gradient explosion. Train/test consistent (E4/E16 recipe). Pair "
                         "with --init_from a trained ckpt + --lr ~1e-5 + --loss ls")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments,
    )

    # --init_seed decouples init/shuffle stochasticity from the train/val SPLIT (which
    # stays on --seed via explicit random_state). Set it (>=0) to vary head-init/dataloader
    # order while keeping the split fixed — clean same-split model soups (E12). -1 = unchanged.
    init_seed = args.init_seed if args.init_seed >= 0 else args.seed
    if args.init_seed >= 0:
        from transformers import set_seed
        set_seed(init_seed)  # head init + shuffle + dropout; split stays on random_state=args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  model={args.model}  method=full-finetune+linear-head")

    # ---- data ----
    samples, y = load_samples(args.data_dir)
    max_hist = args.max_hist or None            # 0 -> None (full history)
    texts = build_texts(samples, input_mode=args.input, max_hist=max_hist,
                        variant=args.serialize, strip_history=args.strip_history)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    if args.session_fold >= 0:
        assert not args.full_data, \
            "--session_fold is the leakage-free k-fold protocol; --full_data would re-leak"
        tr, va = session_fold_indices(samples, y, args.session_fold,
                                      n_splits=args.session_splits, seed=args.seed)
        logger.info(f"session-grouped split: fold {args.session_fold}/{args.session_splits} "
                    f"(StratifiedGroupKFold by session, leakage asserted 0)")
    else:
        tr, va = split_indices(y, seed=args.seed)
    if args.full_data:
        # grow training with 75% of the old val; eval/calibrate on the untouched 25%.
        # Same seed as split_indices -> an --init_from checkpoint trained on `tr`
        # has never seen ANY of the surviving eval samples.
        from sklearn.model_selection import train_test_split
        va_train, va_eval = train_test_split(
            va, test_size=0.25, stratify=y_ids[va], random_state=args.seed)
        tr = np.concatenate([tr, va_train])
        va = va_eval
    if args.all_data:
        # E28 final-submission mode: every sample trains, nothing is held out, so no
        # honest eval exists — eval/best-epoch machinery is disabled further down and
        # the last epoch is what gets saved.
        assert not args.full_data and args.session_fold < 0, \
            "--all_data already uses every sample; drop --full_data/--session_fold"
        assert not args.calibrate, "--all_data leaves no held-out rows to calibrate on"
        tr = np.arange(len(y_ids))
        va = va[:0]  # empty val: downstream dataset/eval paths see 0 rows
        logger.info(f"ALL-DATA mode: train={len(tr)} (everything), no eval split; "
                    f"last-epoch weights will be the final artifact")
    if args.keep_indices:
        keep = set(np.load(args.keep_indices).tolist())
        before = len(tr)
        tr = tr[np.array([i in keep for i in tr], dtype=bool)]
        logger.info(f"keep_indices ({args.keep_indices}): train {before} -> {len(tr)} "
                    f"(dropped {before - len(tr)} noisy)")
    classes = ALL_CLASSES
    if args.group_task:
        assert not args.pair, "--group_task and --pair are mutually exclusive"
        classes = list(ACTION_GROUPS)
        y_ids = np.array([GROUP_ID[CLASS_TO_ID[a]] for a in y])
        logger.info(f"group-router task: 4 classes {classes}")
    if args.pair:
        classes = args.pair.split(",")
        assert len(classes) >= 2 and all(c in CLASS_TO_ID for c in classes), \
            f"--pair must name 2+ of {ALL_CLASSES}"
        # filter AFTER the split, keeping absolute indices: the specialist's train
        # set stays inside the 14-class model's train split, so composing the two
        # on the main val split later is uncontaminated
        pair_ids = {CLASS_TO_ID[c] for c in classes}
        y_ids = np.array([classes.index(a) if CLASS_TO_ID[a] in pair_ids else -1
                          for a in y])
        tr = tr[y_ids[tr] >= 0]
        va = va[y_ids[va] >= 0]
        logger.info(f"pair specialist {classes}: filtered to {len(tr)} train / {len(va)} val")
    if args.zero_history:
        zh = np.array([len(s["history"]) == 0 for s in samples])
        tr = tr[zh[tr]]
        va = va[zh[va]]
        logger.info(f"zero-history (first-step) filter: {len(tr)} train / {len(va)} val")
    if args.strip_history:
        # train on ALL samples (history already stripped in `texts`); eval only on
        # the REAL zero-history val so the number is comparable to --zero_history
        assert not args.zero_history, "--strip_history and --zero_history are exclusive"
        zh = np.array([len(s["history"]) == 0 for s in samples])
        va = va[zh[va]]
        logger.info(f"strip-history augmentation: {len(tr)} train (all, stripped) / "
                    f"{len(va)} real zero-history val")
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    logger.info(f"samples={len(texts)}  train={len(tr)}  val={len(va)}  max_hist={max_hist}  "
                f"full_data={args.full_data}")

    # ---- tokenizer + model + single linear head ----
    # trust_remote_code: some backbones (e.g. gte's model_type "new") ship custom
    # modeling code and won't load without it.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.truncation_side = args.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.init_from:
        logger.info(f"continuing from fine-tuned checkpoint: {args.init_from}")
    attn_impl = args.attn_impl
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    if args.ltp_final_threshold > 0 and attn_impl != "eager":
        logger.info(f"LTP on -> forcing attn_implementation=eager (was {attn_impl}); "
                    "flash/sdpa don't materialize the attention probs LTP scores on")
        attn_impl = "eager"
    logger.info(f"attn_implementation: {attn_impl}"
                + ("  (flash-attn not installed -> sdpa; `pip install flash-attn` to enable)"
                   if attn_impl == "sdpa" and args.attn_impl == "auto" else ""))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init_from or args.model, num_labels=len(classes),
        torch_dtype=torch.float32,   # fp32 for stable classifier training
        attn_implementation=attn_impl,
        trust_remote_code=True,
        # real action names so config.id2label maps ids -> actions (not LABEL_0...);
        # needed for logit_bias.json keys and for inference to emit action strings.
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)},
        # some backbones ship a pretrained head (e.g. gte has a 1-logit head);
        # discard it and init a fresh 14-class head for our task.
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    if args.special_tokens:
        add_action_tokens(tok, model)
    if args.reinit_layers:
        reinit_top_layers(model, args.reinit_layers)
    if args.keep_layer_idx:
        prune_layers(model, [int(x) for x in args.keep_layer_idx.split(",") if x.strip() != ""])
    elif args.keep_layers:
        prune_layers(model, args.keep_layers)
    if args.ffn_keep < 1.0:
        prune_ffn(model, args.ffn_keep)
    if args.heads_keep < 1.0:
        prune_attn_heads(model, args.heads_keep)   # after prune_layers: fresh indices
    if args.head_layers:
        replace_head(model, args.head_layers, args.head_act)
    if args.factor_ffn:
        from src.factored_ffn import build_factored_model
        # calibrate whitened-SVD on train texts (no vocab remap: a full-vocab
        # checkpoint). Move the model to the compute device first so the 256-sample
        # calibration pass runs on GPU when available.
        model.to(device)
        calib_texts = [texts[i] for i in tr[: args.factor_calib]]
        done = build_factored_model(model, tok, calib_texts, args.factor_ffn,
                                    max_len=args.max_len, n_calib=args.factor_calib,
                                    device=device)
        logger.info(f"FFN factorized to rank {args.factor_ffn} in {len(done)} "
                    f"projections (whitened-SVD init); config.factored_ffn recorded")

    if args.ltp_final_threshold > 0:
        assert not args.lora, "LTP (E18) + LoRA not supported (LTP trains full model + thresholds)"
        assert not args.llrd, "LTP + --llrd not supported (LTP uses its own threshold LR group)"
        assert getattr(model.config, "model_type", "") == "modernbert", \
            "LTP (E18) is implemented for granite/ModernBERT only"
        from src.ltp_modeling import install_ltp
        ltp_mode = "hard" if args.ltp_hard_recover else "soft"
        # resolve the temperature anneal endpoints (default = fixed --ltp_temperature)
        ltp_t_start = args.ltp_temp_start if args.ltp_temp_start is not None else args.ltp_temperature
        ltp_t_end = args.ltp_temp_end if args.ltp_temp_end is not None else args.ltp_temperature
        install_ltp(model, args.ltp_final_threshold, temperature=ltp_t_start,
                    ltp_lambda=args.ltp_lambda, mode=ltp_mode)
        if args.ltp_hard_recover:
            # fixed-threshold hard-drop recovery: freeze the per-layer thresholds at the
            # ramp base[i]=final*i/L (deltas stay 0, not trained) -> weights-only recovery,
            # no 1/T soft-mask gradient. The drop decision (score>=thr) is non-diff by
            # construction; weight grads still flow through the KEPT tokens' pathway.
            for p in model.model.ltp_delta:
                p.requires_grad_(False)
            logger.info("LTP hard-recover: thresholds FROZEN at ramp; weights-only FT")

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        head_name = "score" if hasattr(model, "score") else "classifier"
        # module names differ per architecture family
        names = [n for n, _ in model.named_modules()]
        targets = (["q_proj", "v_proj"] if any(n.endswith("q_proj") for n in names)
                   else ["query", "value"])
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora,
                         lora_alpha=2 * args.lora, lora_dropout=0.05,
                         target_modules=targets, modules_to_save=[head_name])
        model = get_peft_model(model, cfg)
        # gradient checkpointing needs grads to flow from the (frozen) embeddings
        model.enable_input_require_grads()
        logger.info(f"LoRA r={args.lora} on {targets} + {head_name} (base frozen)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {n_trainable:,} / {n_total:,} "
                f"({100 * n_trainable / n_total:.1f}%"
                f"{' — LoRA' if args.lora else ' — full fine-tune'})")

    aux_on = bool(args.rdrop or args.supcon or args.hard_boundary)
    if aux_on:
        assert not args.distill_from, \
            "--distill_from is not combinable with --rdrop/--supcon/--hard_boundary"
    assert not (args.rdrop and args.hard_boundary), \
        "--rdrop + --hard_boundary: the KL term would mix weighted/unweighted CE"
    if args.supcon:
        assert not args.pair, "--supcon group weighting assumes the 14-class space"
        import torch.nn as nn
        h = model.config.hidden_size
        # train-only projection head (SupCon standard); saved with checkpoints but
        # ignored by from_pretrained at inference — nothing ships
        model.supcon_proj = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 128))
        logger.info(f"SupCon aux: lam={args.supcon} tau={args.supcon_tau} "
                    f"group_w={args.supcon_group_w} queue={args.supcon_queue}")
    weights_tr = None
    if args.hard_boundary:
        assert args.init_from, "--hard_boundary mines boundaries with the --init_from checkpoint"
        assert not args.hist_dropout, "--hard_boundary needs static per-sample weights"
        weights_tr = mine_boundary_weights(
            model, tok, [texts[i] for i in tr], args.max_len,
            args.hard_boundary, args.boundary_margin, device)

    teacher = None
    if args.distill_from:
        teacher = np.load(args.distill_from)["logits"].astype(np.float32)
        assert len(teacher) == len(texts), \
            f"teacher logits rows ({len(teacher)}) != samples ({len(texts)})"
        logger.info(f"distilling from {args.distill_from} "
                    f"(alpha={args.distill_alpha}, T={args.distill_T})")
    if args.reduced_ids:                       # E24 method-A: PRE-tokenized reduced inputs (default off)
        assert not (args.hist_dropout or args.distill_from or weights_tr is not None), \
            "--reduced_ids uses the plain static path only (no hist_dropout/distill/weights)"
        _rd = np.load(args.reduced_ids)
        _flat, _rlen = _rd["ids_flat"], _rd["lengths"]
        _off = np.concatenate([[0], np.cumsum(_rlen)]).astype(np.int64)
        _get = lambda i: _flat[_off[i]:_off[i + 1]].tolist()
        train_ds = build_ids_dataset([_get(i) for i in tr], y_ids[tr])
        val_ds = build_ids_dataset([_get(i) for i in va], y_ids[va])
        logger.info(f"E24 --reduced_ids {args.reduced_ids}: mean reduced train len "
                    f"{np.mean([int(_rlen[i]) for i in tr]):.0f} tok "
                    f"(full-input path bypassed; split/recipe unchanged)")
    elif args.hist_dropout:
        assert not args.distill_from, "--hist_dropout + --distill_from not supported together"
        logger.info(f"history dropout p={args.hist_dropout} (fresh draw per epoch)")
        train_ds = build_dynamic_dataset(tok, [samples[i] for i in tr], y_ids[tr],
                                         args.max_len, max_hist, args.hist_dropout, init_seed,
                                         variant=args.serialize)
        val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len)
    else:
        train_ds = build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len,
                                 teacher=teacher[tr] if teacher is not None else None,
                                 weights=weights_tr)
        val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len)
    collator = DataCollatorWithPadding(tok)
    if teacher is not None or weights_tr is not None:
        collator = ExtrasCollator(collator)

    # plain cross-entropy (Trainer's default). No class weighting — the 73.07 model
    # handles imbalance via post-hoc logit-bias calibration instead.
    safe = args.model.replace("/", "__") + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(args.out_dir, f"ft_{safe}")
    # persist the tokenizer with the run: with --special_tokens the vocab differs
    # from the hub's, and inference MUST load this copy or every id past the old
    # vocab end is wrong. Saved unconditionally — harmless for stock runs.
    tok.save_pretrained(run_dir)
    # bf16 where the GPU supports it (Ampere+: 3090/4090). fp16 NaN-diverged
    # Qwen3-0.6B on pat (loss -> 0.0, grad_norm NaN mid-epoch-1) while the same
    # recipe was fine for granite — LLM-style backbones overflow fp16's range.
    # 2080 Ti (Turing) has no bf16 -> falls back to fp16 as before.
    # --precision fp16|bf16 overrides (E25: teammate's recipe trains fp16; forcing
    # bf16 on a non-bf16 card would silently train fp32 -> assert instead).
    if args.precision == "auto":
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = device == "cuda" and args.precision == "bf16"
        assert not (args.precision == "bf16" and device == "cuda"
                    and not torch.cuda.is_bf16_supported()), "--precision bf16: no bf16 on this GPU"
    logger.info(f"precision: {'bf16' if use_bf16 else 'fp16' if device == 'cuda' else 'fp32'}"
                + (f" (--precision {args.precision})" if args.precision != "auto" else ""))
    # gradient checkpointing: only worth its ~30-40% slowdown when the config is
    # actually VRAM-tight (large backbone or long sequences). granite-311m@512 uses
    # ~7GB of a 24GB card WITH it on -> pure overhead; auto turns it off. It is
    # result-neutral (recomputes the same forward), so 'off' stays comparable to an
    # 'on' baseline.
    if args.grad_checkpointing == "on":
        use_gc = True
    elif args.grad_checkpointing == "off":
        use_gc = False
    else:  # auto
        n_params = sum(p.numel() for p in model.parameters())
        use_gc = n_params > 4.0e8 or args.max_len > 512
    if args.ltp_final_threshold > 0 and use_gc:
        logger.info("LTP on -> disabling gradient_checkpointing (LTP's custom "
                    "encoder-loop forward doesn't support it; matches the reference)")
        use_gc = False
    logger.info(f"gradient_checkpointing: {use_gc} (--grad_checkpointing={args.grad_checkpointing}"
                + (f", {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, max_len={args.max_len})"
                   if args.grad_checkpointing == "auto" else ")"))
    # checkpointing: the shared disk is slow enough that per-epoch saves stall the GPU,
    # so the DEFAULT (--keep_checkpoints 1) writes NOTHING during training — best-epoch
    # weights live in a CPU-RAM snapshot (BestSnapshot callback replaces save_strategy
    # ='epoch' + load_best_model_at_end) and hit disk exactly once, after train().
    # --keep_checkpoints >1 keeps the legacy per-epoch disk checkpoints, which SWA
    # averaging genuinely needs on disk.
    ram_best = args.keep_checkpoints <= 1
    ckpt_kw = (dict(save_strategy="no")
               if ram_best else
               dict(save_strategy="epoch", load_best_model_at_end=True,
                    metric_for_best_model="macro_f1", greater_is_better=True,
                    save_total_limit=args.keep_checkpoints,
                    save_only_model=True))              # SWA needs weights only; ~3x less disk
    if args.all_data:
        # no eval -> no metric to pick a best epoch by; per-epoch disk saves
        # (--keep_checkpoints >1, SWA) still work, minus load_best_model_at_end
        ckpt_kw = (dict(save_strategy="no") if ram_best else
                   dict(save_strategy="epoch", save_total_limit=args.keep_checkpoints,
                        save_only_model=True))
    targs = TrainingArguments(
        output_dir=run_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        optim=args.optim,                              # sgd for big models (zero optimizer state)
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        logging_strategy="steps", logging_steps=50,   # periodic {loss,epoch} log lines
        disable_tqdm=False,                            # keep the bar; tqdm.auto is log-safe
        gradient_checkpointing=use_gc,                 # trade compute for VRAM (full-FT is heavy)
        group_by_length=args.group_by_length,          # cut padding waste at small batch
        eval_strategy=("no" if args.all_data else "epoch"), **ckpt_kw,
        bf16=use_bf16, fp16=(device == "cuda" and not use_bf16),
        report_to="none", seed=init_seed,
    )
    optimizers = (None, None)
    if args.llrd:
        optimizers = (build_llrd_optimizer(model, args.lr, args.llrd,
                                           args.optim, targs.weight_decay), None)
    trainer_cls = Trainer
    if teacher is not None:
        assert args.loss in ("ce", "ls"), \
            "--distill_from combines with --loss ce or ls only (ls = E30 fix: LS inside " \
            "the KD CE term); focal/wce/la still replace CE and are not combinable"
        trainer_cls = make_distill_trainer(Trainer, args.distill_alpha, args.distill_T,
                                           ce_mode=args.loss,
                                           label_smoothing=args.label_smoothing)
        if args.loss == "ls":
            logger.info(f"distill CE term uses label smoothing eps={args.label_smoothing} "
                        f"(E30 fix — E26 1B students lost the LS gain)")
    if aux_on:
        supcon_cfg = ({"lam": args.supcon, "tau": args.supcon_tau,
                       "group_w": args.supcon_group_w, "queue": args.supcon_queue}
                      if args.supcon else None)
        trainer_cls = make_aux_trainer(Trainer, rdrop=args.rdrop, supcon=supcon_cfg)
    if args.loss != "ce" and teacher is None:
        assert not aux_on, \
            "--loss focal/ls/wce/la replaces the classification CE and is not combinable " \
            "with --rdrop / --supcon / --hard_boundary"
        class_weight = log_prior = None
        if args.loss in ("wce", "la"):
            counts = np.bincount(y_ids[tr], minlength=len(classes)).astype(np.float64)
            assert (counts > 0).all(), f"--loss {args.loss}: empty class in train split: {counts}"
            if args.loss == "wce":
                w = counts.sum() / (len(classes) * counts)          # 'balanced' (sklearn)
                class_weight = torch.tensor(w, dtype=torch.float32)
                logger.info(f"wce weights: min={w.min():.3f} max={w.max():.3f} "
                            f"(rarest={classes[int(counts.argmin())]}, "
                            f"commonest={classes[int(counts.argmax())]})")
            else:
                log_prior = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
                logger.info(f"la log-prior: min={float(log_prior.min()):.3f} "
                            f"max={float(log_prior.max()):.3f} tau={args.la_tau}")
        trainer_cls = make_loss_trainer(Trainer, args.loss, args.focal_gamma,
                                        args.label_smoothing, class_weight, log_prior,
                                        args.la_tau)
    if args.ltp_final_threshold > 0 and not args.ltp_hard_recover:
        assert teacher is None and not aux_on, \
            "LTP not combinable with --distill_from/--rdrop/--supcon/--hard_boundary"
        from src.ltp_modeling import make_ltp_trainer, separate_threshold_params
        # SOFT LTP trainer = classification loss (respecting --loss, e.g. ls) + sparsity
        # regularizer. When --loss ce, loss_fn=None -> the model's built-in CE is used.
        if args.loss != "ce":
            def _ltp_loss_fn(logits, labels, _m=args.loss, _fg=args.focal_gamma,
                             _ls=args.label_smoothing, _cw=class_weight, _lp=log_prior,
                             _lt=args.la_tau):
                return classification_loss(logits, labels, _m, _fg, _ls, _cw, _lp, _lt)
        else:
            _ltp_loss_fn = None
        trainer_cls = make_ltp_trainer(Trainer, loss_fn=_ltp_loss_fn)
        if args.ltp_lr_threshold > 0:      # reference: thresholds get their own LR group
            thr_p, other_p = separate_threshold_params(model)
            opt = torch.optim.AdamW(
                [{"params": other_p, "lr": args.lr},
                 {"params": thr_p, "lr": args.ltp_lr_threshold}],
                lr=args.lr, weight_decay=targs.weight_decay)
            optimizers = (opt, None)
            logger.info(f"LTP: separate threshold LR group lr={args.ltp_lr_threshold} "
                        f"({len(thr_p)} threshold params) vs base lr={args.lr}")
    # hard-recover mode: thresholds frozen, NO regularizer/threshold-group -> the trainer
    # stays whatever --loss selected (e.g. make_loss_trainer for ls). The hard-pruned
    # forward + plain classification loss recovers weights only (E4/E16 recipe).
    elif args.ltp_final_threshold > 0 and args.ltp_hard_recover:
        assert teacher is None and not aux_on, \
            "LTP not combinable with --distill_from/--rdrop/--supcon/--hard_boundary"
        logger.info("LTP hard-recover: weights-only, no sparsity regularizer, "
                    "no threshold LR group")
    dyn_callbacks = []
    snap = None
    if ram_best and not args.all_data:   # all_data: no eval -> snapshot would never fire
        snap = make_best_snapshot()
        dyn_callbacks.append(snap)
    if args.log_dynamics:
        dyn_callbacks.append(make_dynamics_logger(
            train_ds, collator, tr, y_ids[tr], args.log_dynamics))
        logger.info(f"logging per-epoch train dynamics -> {args.log_dynamics}")
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=make_compute_metrics(len(classes)),
        optimizers=optimizers, callbacks=dyn_callbacks or None,
    )
    if (args.ltp_final_threshold > 0 and not args.ltp_hard_recover
            and ltp_t_start != ltp_t_end):
        from src.ltp_modeling import make_ltp_temp_callback
        trainer.add_callback(make_ltp_temp_callback(model, ltp_t_start, ltp_t_end))
        logger.info(f"LTP temperature anneal: {ltp_t_start} -> {ltp_t_end} "
                    f"(linear over training)")

    trainer.train()
    if args.all_data:
        # nothing held out -> no val metric; the in-memory model IS the last epoch
        val_f1 = float("nan")
    elif ram_best and snap.best_state is not None:
        # best-epoch weights come from the RAM snapshot; the per-epoch evals already
        # produced the reported metric, so no extra evaluate() pass is needed
        model.load_state_dict(snap.best_state)
        val_f1 = snap.best_f1
        logger.info(f"best-epoch weights (epoch {snap.best_epoch:.1f}) loaded from RAM")
    else:
        # legacy disk mode (HF reloaded the best checkpoint itself), or the RAM
        # snapshot never fired (e.g. --epochs <1 hit no epoch-end eval)
        val_f1 = trainer.evaluate()["eval_macro_f1"]
    logger.success(f"{args.model}: best val Macro-F1 = {val_f1:.4f}")

    # ---- optional logit-bias calibration on val (73.07 trick). OFF by default:
    # project decision 2026-07-07 is raw logits everywhere; if ever needed it can run
    # pre-submission against the saved checkpoint instead of inside training ----
    tuned_f1 = None
    if args.calibrate:
        pred_out = trainer.predict(val_ds)
        bias, base_f1, tuned_f1 = calibrate_logit_bias(pred_out.predictions,
                                                       pred_out.label_ids)
        logger.success(f"  calibrated: {base_f1:.4f} -> {tuned_f1:.4f} (+{tuned_f1-base_f1:.4f})")
        # save the bias next to the model checkpoint for inference
        id2label = model.config.id2label
        bias_map = {id2label[i]: float(bias[i]) for i in range(len(classes))}
        with open(os.path.join(run_dir, "logit_bias.json"), "w") as f:
            json.dump({"base_macro_f1": float(base_f1), "tuned_macro_f1": float(tuned_f1),
                       "bias": bias_map}, f, indent=2)

    # ---- final artifact: ONE write of the best weights to the run_dir root, in
    # --save_dtype (fp16 default = submission runtime precision, half the bytes).
    # Cast in place — nothing runs a forward pass after this point. Disk-checkpoint
    # mode gets the same root artifact on top of its epoch checkpoints ----
    save_dt = {"fp16": torch.float16, "bf16": torch.bfloat16,
               "fp32": torch.float32}[args.save_dtype]
    if save_dt != torch.float32:
        model.to(save_dt)
    model.config.torch_dtype = save_dt   # honest reload metadata
    model.save_pretrained(run_dir)
    logger.info(f"final model ({args.save_dtype}) -> {run_dir}")

    # ---- record ----
    os.makedirs(args.out_dir, exist_ok=True)
    row = {"model": args.model, "method": f"lora{args.lora}" if args.lora else "full_ft",
           "head": f"mlp{args.head_layers}_{args.head_act}" if args.head_layers else "linear",
           "epochs": args.epochs, "lr": args.lr, "val_macro_f1": round(float(val_f1), 4),
           # column kept for CSV-append compatibility; empty when --calibrate is off
           "calibrated_macro_f1": round(float(tuned_f1), 4) if tuned_f1 is not None else "",
           "tag": args.tag, "max_len": args.max_len, "serialize": args.serialize,
           "special_tokens": args.special_tokens,
           "init_from": args.init_from, "full_data": args.full_data,
           "rdrop": args.rdrop, "supcon": args.supcon,
           "hard_boundary": args.hard_boundary, "pair": args.pair}
    out_csv = os.path.join(args.out_dir, args.results_name)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info(f"appended -> {out_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
