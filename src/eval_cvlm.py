# Evaluation script for CVLM (on-the-fly text-encoder variant).
#
# Modes:
#   cvlm          – full pipeline (text_encoder → pool → ViT → projectors → decoder)
#   baseline_llm  – bare decoder on prompt tokens only (no vision span)
#   baseline_proj – text_encoder → chunked pool → random linear projection → decoder (skip ViT)

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from cvlm_dataset import CvlmTrainDataset, make_collate_fn
from modeling import CVLM, ModelArguments, TrainingArguments, _chunked_mean_pool
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CVLM (on-the-fly encoder variant)")
    p.add_argument("--checkpoint_path", type=str, default="", help="Path to model_step_*.safetensors")
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--vision_encoder_name", type=str, default=None)
    p.add_argument("--text_encoder_name", type=str, default=None)
    p.add_argument("--compression_rate", type=int, default=4)
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_vision_len", type=int, default=512)
    p.add_argument("--max_source_len", type=int, default=0,
                   help="0 = compression_rate * max_vision_len")
    p.add_argument("--max_samples", type=int, default=0, help="Limit eval to first N samples (0 = all).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--mode",
        type=str,
        default="cvlm",
        choices=["cvlm", "baseline_llm", "baseline_proj"],
    )
    p.add_argument("--compute_generation_metrics", action="store_true")
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--tensorboard_dir", type=str, default="")
    p.add_argument("--tb_run_name", type=str, default="")
    p.add_argument("--global_step", type=int, default=0)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _encode_source_for_baseline(model: CVLM, source_ids, source_mask):
    """Run the frozen text_encoder + chunked mean pool. Used by baseline_proj."""
    with torch.no_grad():
        h = model.text_encoder(input_ids=source_ids, attention_mask=source_mask).last_hidden_state
    pooled, vmask = _chunked_mean_pool(
        h.detach(),
        source_mask,
        compression_rate=model.compression_rate,
        max_vision_len=model.max_vision_len,
    )
    return pooled, vmask


# ---------------------------------------------------------------------------
# Teacher-forcing metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_teacher_forcing_cvlm(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Dict[str, float]:
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
        batch = _move(batch, device)
        out = model(
            source_input_ids=batch["source_ids"],
            source_attention_mask=batch["source_attention_mask"],
            prompt_ids=batch["prompt_ids"],
            answer_ids=batch["answer_ids"],
            answer_labels=batch["answer_labels"],
            prompt_mask=batch["prompt_mask"],
            answer_mask=batch["answer_mask"],
        )
        logits = out["logits"]
        vision_mask = out["vision_mask"]
        full_attn = out["attention_mask"]
        answer_labels = batch["answer_labels"]
        B = batch["prompt_ids"].size(0)
        P = batch["prompt_ids"].size(1)
        V = vision_mask.size(1)

        real_lens = full_attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))
        total_v_real += int(vision_mask.sum().item())
        total_p_real += int(batch["prompt_mask"].sum().item())

        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels_full = torch.cat([ignore, answer_labels], dim=1)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels_full[:, 1:].reshape(-1)
        per_tok = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_tok[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P + V - 1:-1, :]
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
    max_samples: int,
) -> Dict[str, float]:
    """Baseline: prompt tokens + answer tokens into the frozen decoder, no vision span."""
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0

    for batch in tqdm(loader, desc="eval (baseline_llm)"):
        batch = _move(batch, device)
        prompt_ids = batch["prompt_ids"]
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        prompt_mask = batch["prompt_mask"]
        answer_mask = batch["answer_mask"]
        B = prompt_ids.size(0)
        P = prompt_ids.size(1)

        attn = torch.cat([prompt_mask, answer_mask], dim=1)
        base_real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(base_real_lens)
        max_dec_len = max(max_dec_len, max(base_real_lens))

        embed_layer = model.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)
        answer_embs = embed_layer(answer_ids)
        decoder_input = torch.cat([prompt_embs, answer_embs], dim=1)

        ignore = torch.full((B, P), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_token_loss = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_token_loss[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P - 1:-1, :]
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
    max_samples: int,
) -> Dict[str, float]:
    """Baseline: text encoder → chunked pool → random linear projection → decoder (skip ViT)."""
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
        batch = _move(batch, device)
        B = batch["prompt_ids"].size(0)
        pooled, vision_mask = _encode_source_for_baseline(
            model, batch["source_ids"], batch["source_attention_mask"]
        )
        vision_embeds = proj(pooled)  # [B, V, llm_dim]

        embed_layer = model.decoder.get_input_embeddings()
        prompt_embs = embed_layer(batch["prompt_ids"])
        answer_embs = embed_layer(batch["answer_ids"])
        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        attn = torch.cat([batch["prompt_mask"], vision_mask, batch["answer_mask"]], dim=1)
        P = batch["prompt_ids"].size(1)
        V = vision_embeds.size(1)
        answer_labels = batch["answer_labels"]
        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))

        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_token_loss = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_token_loss[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P + V - 1:-1, :]
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
# Generation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answers(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
    max_samples: int,
    mode: str,
    proj: Optional[nn.Linear] = None,
) -> tuple[list[str], list[str]]:
    model.eval()
    tokenizer = model.tokenizer
    predictions: list[str] = []
    references: list[str] = []
    n_samples = 0

    for batch in tqdm(loader, desc=f"generate ({mode})"):
        batch = _move(batch, device)
        B = batch["prompt_ids"].size(0)
        prompt_mask = batch["prompt_mask"]

        if mode == "cvlm":
            gen_ids = model.generate(
                source_input_ids=batch["source_ids"],
                source_attention_mask=batch["source_attention_mask"],
                prompt_ids=batch["prompt_ids"],
                prompt_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
            )
        elif mode == "baseline_llm":
            prompt_embs = model.decoder.get_input_embeddings()(batch["prompt_ids"])
            gen_ids = model.decoder.generate(
                inputs_embeds=prompt_embs,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        elif mode == "baseline_proj":
            assert proj is not None
            pooled, vision_mask = _encode_source_for_baseline(
                model, batch["source_ids"], batch["source_attention_mask"]
            )
            vision_embeds = proj(pooled)
            prompt_embs = model.decoder.get_input_embeddings()(batch["prompt_ids"])
            decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)
            attn = torch.cat([prompt_mask, vision_mask], dim=1)
            gen_ids = model.decoder.generate(
                inputs_embeds=decoder_input,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend(pred_texts)

        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        for i in range(B):
            m = answer_labels[i] != -100
            ref_ids = answer_ids[i][m]
            references.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    if max_samples > 0:
        return predictions[:max_samples], references[:max_samples]
    return predictions, references


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
# Compression metrics
# ---------------------------------------------------------------------------

def compute_compression_stats(
    dataset: CvlmTrainDataset,
    decoder_tokenizer,
    compression_rate: int,
    max_vision_len: int,
    max_samples: int,
) -> Dict[str, float]:
    """Per-sample source-token and vision-length stats for the on-the-fly pipeline.

    Definitions
    -----------
    S_i : int
        Number of tokens obtained by tokenizing ``record["input"]`` with the
        *decoder's* tokenizer (not ModernBERT's). This makes the number
        directly comparable with the token budget a plain-LLM baseline would
        consume.
    V_i : int
        Number of compressed vision slots the decoder actually sees for this
        sample. Derived as ``min(ceil(L_enc_i / compression_rate), max_vision_len)``
        where ``L_enc_i`` is the source length in encoder tokens (after the
        dataset filter, before truncation).
    P_i, A_i : int
        Prompt / answer decoder-token counts (tracked by the per-mode eval
        loops via ``prompt_len_mean_seen`` / ``answer`` side of the mask).

    Metrics returned
    ----------------
    source_tokens_mean, source_tokens_sum
        Mean and total of S_i across the evaluated samples. ``_sum`` is the
        denominator used by ``bits_per_source_token``.
    vision_len_mean, vision_len_median, vision_len_min, vision_len_max
        Distribution of V_i. Sanity checks that the compression stage is
        neither collapsing to 1 nor saturating at ``max_vision_len``.
    compression_ratio_mean, _median, _p10, _p90, _min, _max
        Distribution of ``S_i / V_i``. With ``--compression_rate K`` the mean
        should sit near K (it can differ because S_i uses the LLM tokenizer
        while V_i is derived from the encoder tokenizer's length).
    n_compression_samples : int
        Number of samples the stats were computed over (≤ ``max_samples``).

    Related joint metrics computed in ``main`` (not here):
    - ``decoder_input_len_mean`` / ``_max`` : mean / max of real P_i+V_i+A_i.
    - ``effective_context_reduction`` : S_i / (V_i + P_i).
    - ``bits_per_source_token`` : (answer NLL in nats / ln 2) / sum(S_i).
      Lower = fewer bits of answer-side cross-entropy per unit of source.
    """
    enc_tok = dataset._enc_tok
    S: list[int] = []
    V: list[int] = []
    n = len(dataset)
    limit = n if max_samples <= 0 else min(n, max_samples)
    cr = max(int(compression_rate), 1)
    for idx in tqdm(range(limit), desc="compression stats"):
        row_id = dataset._row_indices[idx]
        text = dataset._hf[row_id]["input"]
        s_len = len(decoder_tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        l_enc = len(enc_tok(text, add_special_tokens=False, truncation=False)["input_ids"])
        l_enc = min(l_enc, dataset.max_source_len)
        v_len = min((l_enc + cr - 1) // cr, max_vision_len)
        if v_len == 0:
            v_len = 1
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

    Joint quality/compression score. Expresses the model's answer-side
    cross-entropy in bits, normalised by the number of source-text tokens the
    compressed context stood in for. Lower is better, and it is directly
    comparable across runs with different compression rates (whereas raw
    perplexity is only comparable at a fixed sequence layout).
    """
    if total_source_tokens <= 0:
        return 0.0
    return (total_answer_nll_nats / math.log(2.0)) / float(total_source_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_bf16

    model_args = ModelArguments(train=False)
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path
    if args.vision_encoder_name:
        model_args.vision_encoder_name = args.vision_encoder_name
    if args.text_encoder_name:
        model_args.text_encoder_name = args.text_encoder_name
    model_args.max_vision_len = args.max_vision_len
    model_args.compression_rate = args.compression_rate

    training_args = TrainingArguments(output_dir="/tmp/eval_cvlm_dummy")
    training_args.bf16 = bool(use_bf16)

    print(f"Loading model ({model_args.model_name_or_path})...")
    model = CVLM(model_args, training_args)
    model.to(device)

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_file(args.checkpoint_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  loaded; missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    proj: Optional[nn.Linear] = None
    if args.mode == "baseline_proj":
        embed_dim = model.text_encoder.config.hidden_size
        llm_dim = model.decoder.config.hidden_size
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        proj = nn.Linear(embed_dim, llm_dim).to(device=device, dtype=dtype)
        proj.eval()
        print(f"baseline_proj: Linear({embed_dim} → {llm_dim}), random init")

    dec_pad = model.tokenizer.pad_token_id
    enc_pad = model.encoder_tokenizer.pad_token_id
    if dec_pad is None or enc_pad is None:
        raise ValueError("Both decoder and encoder tokenizers must define pad_token")
    collate = make_collate_fn(dec_pad_id=dec_pad, enc_pad_id=enc_pad)

    max_source_len = args.max_source_len if args.max_source_len > 0 else args.compression_rate * args.max_vision_len
    dataset = CvlmTrainDataset(
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        decoder_tokenizer_name=model_args.model_name_or_path,
        encoder_tokenizer_name=model_args.text_encoder_name,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_source_len=max_source_len,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    print(f"\n=== Teacher-forcing evaluation (mode={args.mode}) ===")
    if args.mode == "cvlm":
        tf_metrics = eval_teacher_forcing_cvlm(model, loader, device, args.max_samples)
    elif args.mode == "baseline_llm":
        tf_metrics = eval_teacher_forcing_baseline_llm(model, loader, device, args.max_samples)
    elif args.mode == "baseline_proj":
        assert proj is not None
        tf_metrics = eval_teacher_forcing_baseline_proj(model, proj, loader, device, args.max_samples)
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

    print("\n=== Compression stats ===")
    comp_stats = compute_compression_stats(
        dataset,
        model.tokenizer,
        compression_rate=args.compression_rate,
        max_vision_len=args.max_vision_len,
        max_samples=args.max_samples,
    )
    for k in [
        "source_tokens_mean",
        "vision_len_mean", "vision_len_median", "vision_len_min", "vision_len_max",
        "compression_ratio_mean", "compression_ratio_median",
        "compression_ratio_p10", "compression_ratio_p90",
        "compression_ratio_min", "compression_ratio_max",
        "n_compression_samples",
    ]:
        print(f"  {k:28s}: {comp_stats[k]}")

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

    gen_metrics: Dict[str, float] = {}
    if args.compute_generation_metrics:
        print(f"\n=== Generation evaluation (mode={args.mode}) ===")
        preds, refs = generate_answers(
            model, loader, device, args.max_new_tokens, args.max_samples, args.mode, proj
        )
        gen_metrics = compute_generation_metrics(preds, refs)
        print(f"  ROUGE-1:       {gen_metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:       {gen_metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:       {gen_metrics['rougeL']:.4f}")
        print(f"  BLEU-4:        {gen_metrics['bleu4']:.2f}")
        print(f"  Exact Match:   {gen_metrics['exact_match']:.4f}")

    results = {
        **tf_metrics,
        **gen_metrics,
        **comp_stats,
        "bits_per_source_token": bps,
        "effective_context_reduction": eff_reduction,
    }

    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        run_name = args.tb_run_name or f"eval_{args.mode}"
        tb_path = os.path.join(args.tensorboard_dir, run_name)
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_path)
        step = int(args.global_step)
        for k, v in results.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                writer.add_scalar(f"eval/{k}", float(v), step)
        # Histogram of per-sample compression ratios from comp_stats data.
        per_sample_ratios = []
        limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
        cr = max(int(args.compression_rate), 1)
        enc_tok = dataset._enc_tok
        for idx in range(limit):
            row_id = dataset._row_indices[idx]
            text = dataset._hf[row_id]["input"]
            s_len = len(model.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            l_enc = min(len(enc_tok(text, add_special_tokens=False, truncation=False)["input_ids"]),
                        dataset.max_source_len)
            v_len = max(min((l_enc + cr - 1) // cr, args.max_vision_len), 1)
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
