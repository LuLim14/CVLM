# Evaluation script for CVLM with three modes:
#   cvlm          – full pipeline (embeddings → ViT → projectors → decoder)
#   baseline_llm  – bare SmolLM decoder (prompt tokens only, no embeddings)
#   baseline_proj – embeddings projected directly to LLM space (skip ViT)

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from cvlm_dataset import CvlmTrainDataset, make_collate_fn
from modeling import CVLM, ModelArguments, TrainingArguments
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CVLM")
    p.add_argument("--checkpoint_path", type=str, default="", help="Path to model_step_*.safetensors")
    p.add_argument(
        "--embeddings_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset_100_samples", "embeddings.npy"),
    )
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--vision_encoder_name", type=str, default=None)
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_vision_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=0, help="Limit eval to first N samples (0 = all).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate for generation metrics.")
    p.add_argument(
        "--mode",
        type=str,
        default="cvlm",
        choices=["cvlm", "baseline_llm", "baseline_proj"],
        help="cvlm: full pipeline; baseline_llm: prompt-only decoder; baseline_proj: linear proj (no ViT).",
    )
    p.add_argument("--compute_generation_metrics", action="store_true", help="Compute ROUGE/BLEU (slow, requires generation).")
    p.add_argument("--output_json", type=str, default="", help="Save results to JSON file.")
    p.add_argument("--tensorboard_dir", type=str, default="",
                   help="If set, log eval metrics (scalars + compression-ratio histogram) to this TB dir. "
                        "Point it at the same <output_dir>/tb used by training to see train+eval together.")
    p.add_argument("--tb_run_name", type=str, default="",
                   help="Sub-run name under tensorboard_dir. Defaults to 'eval_<mode>'.")
    p.add_argument("--global_step", type=int, default=0,
                   help="Step to tag eval scalars with in TB (e.g. the checkpoint's training step).")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Teacher-forcing metrics (fast, no generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_teacher_forcing_cvlm(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int,
) -> Dict[str, float]:
    """Full CVLM forward: embeddings → ViT → projectors → decoder."""
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0
    total_v_real = 0
    total_p_real = 0

    for batch in tqdm(loader, desc="eval (cvlm)"):
        batch = _move(batch, device, dtype)
        out = model(
            batch["input_embeds"],
            batch["prompt_ids"],
            batch["answer_ids"],
            answer_labels=batch["answer_labels"],
            attention_mask=batch["attention_mask"],
        )
        logits = out["logits"]
        answer_labels = batch["answer_labels"]
        attn = batch["attention_mask"]
        B = batch["prompt_ids"].size(0)
        P = batch["prompt_ids"].size(1)
        V = batch["input_embeds"].size(1)

        # Real (non-pad) decoder lengths per sample from the attention mask.
        real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))
        total_v_real += int(batch["vision_lens"].sum().item())
        total_p_real += int(attn[:, :P].sum().item())

        # Rebuild the full labels tensor to match the logits layout.
        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels_full = torch.cat([ignore, answer_labels], dim=1)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels_full[:, 1:].reshape(-1)
        per_tok = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_tok[mask].sum().item()
        total_tokens += int(mask.sum().item())

        # Token accuracy on answer positions.
        answer_logits = logits[:, P + V - 1 : -1, :]  # [B, A, vocab]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "vision_len_mean_seen": total_v_real / max(n_samples, 1),
        "prompt_len_mean_seen": total_p_real / max(n_samples, 1),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


@torch.no_grad()
def eval_teacher_forcing_baseline_llm(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int,
) -> Dict[str, float]:
    """Baseline: feed prompt tokens directly into decoder, no embeddings."""
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0

    for batch in tqdm(loader, desc="eval (baseline_llm)"):
        batch = _move(batch, device, dtype)
        prompt_ids = batch["prompt_ids"]
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        full_mask = batch["attention_mask"]
        V = batch["input_embeds"].size(1)
        B = prompt_ids.size(0)
        # Real (non-pad) P + A slots seen by the decoder, per sample.
        base_real_lens = (
            full_mask[:, : prompt_ids.size(1)].sum(dim=1)
            + full_mask[:, prompt_ids.size(1) + V :].sum(dim=1)
        ).tolist()
        total_dec_len += sum(base_real_lens)
        max_dec_len = max(max_dec_len, max(base_real_lens))

        embed_layer = model.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)
        answer_embs = embed_layer(answer_ids)
        decoder_input = torch.cat([prompt_embs, answer_embs], dim=1)

        P = prompt_ids.size(1)
        ignore = torch.full((B, P), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        # Drop the vision span from the combined attention mask.
        attn = torch.cat([full_mask[:, :P], full_mask[:, P + V:]], dim=1)
        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_token_loss = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_token_loss[mask].sum().item()
        total_tokens += mask.sum().item()

        answer_logits = logits[:, P - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


@torch.no_grad()
def eval_teacher_forcing_baseline_proj(
    model: CVLM,
    proj: nn.Linear,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int,
) -> Dict[str, float]:
    """Baseline: project embeddings directly to LLM space (single linear, no ViT)."""
    model.eval()
    proj.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0

    for batch in tqdm(loader, desc="eval (baseline_proj)"):
        batch = _move(batch, device, dtype)
        B = batch["prompt_ids"].size(0)
        real_lens = batch["attention_mask"].sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))

        vision_embeds = proj(batch["input_embeds"])  # [B, V, llm_dim]
        embed_layer = model.decoder.get_input_embeddings()
        prompt_embs = embed_layer(batch["prompt_ids"])
        answer_embs = embed_layer(batch["answer_ids"])
        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        P = batch["prompt_ids"].size(1)
        V = vision_embeds.size(1)
        answer_labels = batch["answer_labels"]
        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        attn = batch["attention_mask"]
        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_token_loss = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_token_loss[mask].sum().item()
        total_tokens += mask.sum().item()

        answer_logits = logits[:, P + V - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Generation metrics (slow)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answers(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    max_samples: int,
    mode: str,
    proj: Optional[nn.Linear] = None,
) -> tuple[list[str], list[str]]:
    """Generate answers and collect references. Returns (predictions, references)."""
    model.eval()
    tokenizer = model.tokenizer
    predictions: list[str] = []
    references: list[str] = []
    n_samples = 0

    for batch in tqdm(loader, desc=f"generate ({mode})"):
        batch = _move(batch, device, dtype)
        B = batch["prompt_ids"].size(0)

        P = batch["prompt_ids"].size(1)
        V = batch["input_embeds"].size(1)
        full_mask = batch["attention_mask"]
        prompt_vision_mask = full_mask[:, : P + V]
        prompt_only_mask = full_mask[:, :P]

        if mode == "cvlm":
            gen_ids = model.generate(
                batch["input_embeds"],
                batch["prompt_ids"],
                attention_mask=prompt_vision_mask,
                max_new_tokens=max_new_tokens,
            )
        elif mode == "baseline_llm":
            prompt_embs = model.decoder.get_input_embeddings()(batch["prompt_ids"])
            gen_ids = model.decoder.generate(
                inputs_embeds=prompt_embs,
                attention_mask=prompt_only_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        elif mode == "baseline_proj":
            assert proj is not None
            vision_embeds = proj(batch["input_embeds"])
            prompt_embs = model.decoder.get_input_embeddings()(batch["prompt_ids"])
            decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)
            gen_ids = model.decoder.generate(
                inputs_embeds=decoder_input,
                attention_mask=prompt_vision_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend(pred_texts)

        # Decode reference answers (non-padded tokens only)
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        for i in range(B):
            mask = answer_labels[i] != -100
            ref_ids = answer_ids[i][mask]
            references.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    return predictions[:max_samples] if max_samples > 0 else predictions, references[:max_samples] if max_samples > 0 else references


def compute_generation_metrics(predictions: list[str], references: list[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    import sacrebleu

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    bleu = sacrebleu.corpus_bleu(predictions, [references])

    n_exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())

    return {
        "rouge1": sum(r1) / len(r1) if r1 else 0.0,
        "rouge2": sum(r2) / len(r2) if r2 else 0.0,
        "rougeL": sum(rl) / len(rl) if rl else 0.0,
        "bleu4": bleu.score,
        "exact_match": n_exact / len(predictions) if predictions else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_compression_stats(
    dataset,
    tokenizer,
    max_samples: int,
) -> Dict[str, float]:
    """Per-sample source-token and vision-length stats for the CVLM pipeline.

    Definitions
    -----------
    S_i : int
        Number of tokens obtained by tokenizing ``record["input"]`` (the raw
        source text) with the *decoder's* tokenizer. Using the LLM tokenizer
        (not ModernBERT's) makes the number directly comparable with the
        token budget that a plain-LLM baseline would consume.
    V_i : int
        Number of rows in the pre-computed embedding sequence — i.e. the
        length of the compressed context the decoder actually sees.
    P_i, A_i : int
        Prompt / answer token counts (not collected here, but reported by the
        main eval loop through the mean decoder input length).

    Metrics returned
    ----------------
    source_tokens_mean, source_tokens_sum
        Mean and total of S_i across the evaluated samples. ``_sum`` is the
        denominator used by ``bits_per_source_token``.
    vision_len_mean, vision_len_median, vision_len_min, vision_len_max
        Distribution of V_i. Sanity checks that the compression stage is
        neither collapsing to 1 nor saturating at ``max_vision_len``.
    compression_ratio_mean, _median, _p10, _p90, _min, _max
        Distribution of ``S_i / V_i`` — the headline "how many original
        tokens each compressed slot represents". With
        ``embed_dataset.py --compression_rate K`` the mean should sit near K
        (it can differ because the source counts use the LLM tokenizer while
        pooling was done in ModernBERT-token space).
    n_compression_samples : int
        Number of samples the stats were computed over (≤ ``max_samples``).

    Related metrics computed inside the per-mode eval loops (not here):
    - ``decoder_input_len_mean`` / ``_max`` : mean / max of P_i + V_i + A_i,
      a direct proxy for decoder memory and attention FLOPs.
    - ``effective_context_reduction`` : S_i / (V_i + P_i), i.e. how many
      source tokens each non-answer decoder slot stands in for.
    - ``bits_per_source_token`` : joint quality/compression score, equal to
      ``(sum of answer NLL in nats / ln 2) / sum(S_i)``. Lower = the model
      needs fewer bits of answer-side cross-entropy per unit of source
      content; comparable across different compression rates.
    """
    S: list[int] = []
    V: list[int] = []
    n = len(dataset)
    limit = n if max_samples <= 0 else min(n, max_samples)
    for idx in tqdm(range(limit), desc="compression stats"):
        src_i = dataset._row_indices[idx]
        emb = dataset._get_row(src_i)
        v_len = int(emb.shape[0]) if emb.ndim == 2 else 1
        row_id = int(dataset._indexes[src_i])
        text = dataset._hf[row_id]["input"]
        s_len = len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        S.append(s_len)
        V.append(v_len)
    S_arr = np.asarray(S, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64).clip(min=1.0)
    ratio = S_arr / V_arr
    return {
        "source_tokens_mean": float(S_arr.mean()),
        "source_tokens_sum": float(S_arr.sum()),
        "vision_len_mean": float(V_arr.mean()),
        "vision_len_median": float(np.median(V_arr)),
        "vision_len_min": float(V_arr.min()),
        "vision_len_max": float(V_arr.max()),
        "compression_ratio_mean": float(ratio.mean()),
        "compression_ratio_median": float(np.median(ratio)),
        "compression_ratio_p10": float(np.percentile(ratio, 10)),
        "compression_ratio_p90": float(np.percentile(ratio, 90)),
        "compression_ratio_min": float(ratio.min()),
        "compression_ratio_max": float(ratio.max()),
        "n_compression_samples": int(len(S_arr)),
    }


def bits_per_source_token(total_answer_nll_nats: float, total_source_tokens: int) -> float:
    """bits_per_source_token = (sum answer NLL in nats / ln 2) / sum(S_i).

    Joint quality/compression metric. Expresses the model's answer-side
    cross-entropy in bits, normalised by the number of source-text tokens the
    compressed context stood in for. Lower is better, and it is directly
    comparable across runs with different compression rates (whereas raw
    perplexity is only comparable at a fixed sequence layout).
    """
    if total_source_tokens <= 0:
        return 0.0
    return (total_answer_nll_nats / math.log(2.0)) / float(total_source_tokens)


def _move(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if k == "input_embeds":
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device=device)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Build model (no training init — frozen everything)
    model_args = ModelArguments(train=False)
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path
    if args.vision_encoder_name:
        model_args.vision_encoder_name = args.vision_encoder_name
    model_args.max_vision_len = args.max_vision_len

    training_args = TrainingArguments(output_dir="/tmp/eval_cvlm_dummy")
    training_args.bf16 = bool(use_bf16)

    print(f"Loading model ({model_args.model_name_or_path})...")
    model = CVLM(model_args, training_args)
    model.to(device)

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_file(args.checkpoint_path)
        model.load_state_dict(state_dict)

    model.eval()

    # Optional: direct projection baseline (random init, untrained)
    proj: Optional[nn.Linear] = None
    if args.mode == "baseline_proj":
        embed_dim = model_args.embed_input_dim
        llm_dim = model.decoder.config.hidden_size
        proj = nn.Linear(embed_dim, llm_dim).to(device=device, dtype=dtype)
        proj.eval()
        print(f"baseline_proj: Linear({embed_dim} → {llm_dim}), random init")

    # Dataset
    tok_pad = model.tokenizer.pad_token_id
    if tok_pad is None:
        raise ValueError("Tokenizer must define pad_token")
    collate = make_collate_fn(tok_pad)

    dataset = CvlmTrainDataset(
        embeddings_path=os.path.normpath(args.embeddings_path),
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        tokenizer_name=model_args.model_name_or_path,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_vision_len=args.max_vision_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    # Teacher-forcing metrics
    print(f"\n=== Teacher-forcing evaluation (mode={args.mode}) ===")
    if args.mode == "cvlm":
        tf_metrics = eval_teacher_forcing_cvlm(model, loader, device, dtype, args.max_samples)
    elif args.mode == "baseline_llm":
        tf_metrics = eval_teacher_forcing_baseline_llm(model, loader, device, dtype, args.max_samples)
    elif args.mode == "baseline_proj":
        assert proj is not None
        tf_metrics = eval_teacher_forcing_baseline_proj(model, proj, loader, device, dtype, args.max_samples)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    tf_metrics["mode"] = args.mode
    print(f"  Perplexity:      {tf_metrics['perplexity']:.4f}")
    print(f"  Token Accuracy:  {tf_metrics['token_accuracy']:.4f}")
    print(f"  Avg Loss:        {tf_metrics['avg_loss']:.4f}")
    print(f"  Samples:         {tf_metrics['n_samples']}")
    print(f"  Decoder in-len mean/max: "
          f"{tf_metrics.get('decoder_input_len_mean', 0):.1f} / "
          f"{tf_metrics.get('decoder_input_len_max', 0)}")

    # Compression metrics (data-side; independent of the eval mode)
    print("\n=== Compression stats ===")
    comp_stats = compute_compression_stats(dataset, model.tokenizer, args.max_samples)
    for k in [
        "source_tokens_mean",
        "vision_len_mean", "vision_len_median", "vision_len_min", "vision_len_max",
        "compression_ratio_mean", "compression_ratio_median",
        "compression_ratio_p10", "compression_ratio_p90",
        "compression_ratio_min", "compression_ratio_max",
        "n_compression_samples",
    ]:
        print(f"  {k:28s}: {comp_stats[k]}")

    # Joint quality/compression scores (require both sides — computed here).
    total_nll = float(tf_metrics.get("total_answer_nll_nats", 0.0))
    total_src = int(comp_stats.get("source_tokens_sum", 0))
    bps = bits_per_source_token(total_nll, total_src)
    mean_p = float(tf_metrics.get("prompt_len_mean_seen", 0.0)) if args.mode == "cvlm" else 0.0
    mean_v = float(tf_metrics.get("vision_len_mean_seen", 0.0)) if args.mode == "cvlm" else 0.0
    eff_reduction = (
        comp_stats["source_tokens_mean"] / (mean_v + mean_p)
        if (mean_v + mean_p) > 0 else 0.0
    )
    print(f"\n  bits_per_source_token       : {bps:.6f}")
    if args.mode == "cvlm":
        print(f"  effective_context_reduction : {eff_reduction:.4f}  "
              f"(source_tokens_mean / (V_mean + P_mean))")

    # Generation metrics (optional)
    gen_metrics: Dict[str, float] = {}
    if args.compute_generation_metrics:
        print(f"\n=== Generation evaluation (mode={args.mode}) ===")
        preds, refs = generate_answers(
            model, loader, device, dtype, args.max_new_tokens, args.max_samples, args.mode, proj
        )
        gen_metrics = compute_generation_metrics(preds, refs)
        print(f"  ROUGE-1:       {gen_metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:       {gen_metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:       {gen_metrics['rougeL']:.4f}")
        print(f"  BLEU-4:        {gen_metrics['bleu4']:.2f}")
        print(f"  Exact Match:   {gen_metrics['exact_match']:.4f}")

    # Save results
    results = {
        **tf_metrics,
        **gen_metrics,
        **comp_stats,
        "bits_per_source_token": bps,
        "effective_context_reduction": eff_reduction,
    }

    # TensorBoard logging (point at the same dir training used to get one view)
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        run_name = args.tb_run_name or f"eval_{args.mode}"
        tb_path = os.path.join(args.tensorboard_dir, run_name)
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_path)
        step = int(args.global_step)
        # Scalar metrics (everything numeric in results except the mode string).
        for k, v in results.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                writer.add_scalar(f"eval/{k}", float(v), step)
        # Histogram of the per-sample compression ratios (re-tokenise once — cheap
        # compared to generation — to get the raw distribution back).
        per_sample_ratios = []
        limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
        for idx in range(limit):
            src_i = dataset._row_indices[idx]
            emb = dataset._get_row(src_i)
            v_len = max(int(emb.shape[0] if emb.ndim == 2 else 1), 1)
            row_id = int(dataset._indexes[src_i])
            s_len = len(model.tokenizer(
                dataset._hf[row_id]["input"], add_special_tokens=False, truncation=False
            )["input_ids"])
            per_sample_ratios.append(s_len / v_len)
        if per_sample_ratios:
            writer.add_histogram("eval/compression_ratio_dist", np.asarray(per_sample_ratios), step)
        writer.close()
        print(f"\nEval metrics logged to TensorBoard at {tb_path} (step={step})")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
