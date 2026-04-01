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
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0

    for batch in tqdm(loader, desc="eval (cvlm)"):
        batch = _move(batch, device, dtype)
        out = model(
            batch["input_embeds"],
            batch["prompt_ids"],
            batch["answer_ids"],
            answer_labels=batch["answer_labels"],
        )
        loss, logits = out["loss"], out["logits"]
        labels = batch["answer_labels"]

        # Token accuracy on answer positions (non-ignored)
        P = batch["prompt_ids"].size(1)
        V = batch["input_embeds"].size(1)
        answer_logits = logits[:, P + V - 1 : -1, :]  # shifted: predict next token
        preds = answer_logits.argmax(dim=-1)
        mask = labels != -100
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()
        n_samples += batch["prompt_ids"].size(0)
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
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

    for batch in tqdm(loader, desc="eval (baseline_llm)"):
        batch = _move(batch, device, dtype)
        prompt_ids = batch["prompt_ids"]
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        B = prompt_ids.size(0)

        prompt_embs = model.encoder.get_input_embeddings()(prompt_ids)
        answer_embs = model.encoder.get_input_embeddings()(answer_ids)
        decoder_input = torch.cat([prompt_embs, answer_embs], dim=1)

        P = prompt_ids.size(1)
        ignore = torch.full((B, P), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        out = model.decoder(inputs_embeds=decoder_input)
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

    for batch in tqdm(loader, desc="eval (baseline_proj)"):
        batch = _move(batch, device, dtype)
        B = batch["prompt_ids"].size(0)

        vision_embeds = proj(batch["input_embeds"])  # [B, V, llm_dim]
        prompt_embs = model.encoder.get_input_embeddings()(batch["prompt_ids"])
        answer_embs = model.encoder.get_input_embeddings()(batch["answer_ids"])
        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        P = batch["prompt_ids"].size(1)
        V = vision_embeds.size(1)
        answer_labels = batch["answer_labels"]
        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        out = model.decoder(inputs_embeds=decoder_input)
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

        if mode == "cvlm":
            gen_ids = model.generate(
                batch["input_embeds"], batch["prompt_ids"], max_new_tokens=max_new_tokens
            )
        elif mode == "baseline_llm":
            prompt_embs = model.encoder.get_input_embeddings()(batch["prompt_ids"])
            gen_ids = model.decoder.generate(
                inputs_embeds=prompt_embs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif mode == "baseline_proj":
            assert proj is not None
            vision_embeds = proj(batch["input_embeds"])
            prompt_embs = model.encoder.get_input_embeddings()(batch["prompt_ids"])
            decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)
            gen_ids = model.decoder.generate(
                inputs_embeds=decoder_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
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

def _move(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
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

    training_args = TrainingArguments(output_dir="/tmp/eval_cvlm_dummy")
    training_args.bf16 = bool(use_bf16)

    print(f"Loading model ({model_args.model_name_or_path})...")
    model = CVLM(model_args, training_args)

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_file(args.checkpoint_path)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Optional: direct projection baseline (random init, untrained)
    proj: Optional[nn.Linear] = None
    if args.mode == "baseline_proj":
        embed_dim = model_args.embed_input_dim
        llm_dim = model.encoder.config.hidden_size
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
    results = {**tf_metrics, **gen_metrics}
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
