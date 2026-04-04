import argparse
import logging
import os

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute per-document embedding sequences for CVLM.")
    p.add_argument("--embeddings_path", type=str, required=True,
                   help="Output path (.npz). Stores indexes, flat, offsets.")
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--embedder_model", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--max_length", type=int, default=2048,
                   help="Max source tokens before truncation.")
    p.add_argument("--max_samples", type=int, default=0,
                   help="If >0, subsample the first N rows.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--compression_rate", type=int, default=4,
                   help="Average-pool contiguous groups of this many real tokens into one embedding.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    set_seed(args.seed)

    print(f"Accelerator device: {accelerator.device}")

    model = AutoModel.from_pretrained(
        args.embedder_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    print(f"Embedder loaded on {model.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.embedder_model,
        use_fast=True,
        trust_remote_code=True,
        padding_side="right",
        truncation_side="right",
        model_max_length=args.max_length,
    )

    with accelerator.main_process_first():
        ds = load_dataset(args.dataset_name, split=args.dataset_split)
        if args.max_samples > 0:
            ds = ds.select(range(min(args.max_samples, len(ds))))

    ds = ds.add_column("index", list(range(len(ds))))

    def does_fit(sample):
        ids = tokenizer(sample["input"], padding=False, truncation=False)["input_ids"]
        return len(ids) <= args.max_length

    len_before = len(ds)
    ds = ds.filter(does_fit, num_proc=max(os.cpu_count() // max(torch.cuda.device_count(), 1), 1))
    ds = ds.shuffle(seed=args.seed)
    len_after = len(ds)
    print(f"Dataset size: {len_before} -> {len_after} after length filter (<= {args.max_length} tokens)")

    compression_rate = max(int(args.compression_rate), 1)

    def get_embeddings(batch):
        text_inputs = [str(x) for x in batch["input"]]
        enc = tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**enc)
            hidden = outputs.last_hidden_state  # [B, L, H]
        mask = enc["attention_mask"].to(hidden.dtype)  # [B, L]
        # Mean-pool contiguous chunks of `compression_rate` real tokens per sample.
        result = []
        for i in range(hidden.size(0)):
            L_real = int(mask[i].sum().item())
            if L_real == 0:
                # Degenerate: emit a single zero vector so shape is well-defined.
                result.append(np.zeros((1, hidden.size(-1)), dtype=np.float32))
                continue
            h = hidden[i, :L_real].to(torch.float32)  # [L_real, H]
            n_chunks = (L_real + compression_rate - 1) // compression_rate
            pooled = torch.zeros((n_chunks, h.size(-1)), dtype=torch.float32, device=h.device)
            for c in range(n_chunks):
                s = c * compression_rate
                e = min(s + compression_rate, L_real)
                pooled[c] = h[s:e].mean(dim=0)
            result.append(pooled.cpu().numpy())
        return {"embeddings": result}

    ds_with_emb = ds.map(get_embeddings, batched=True, batch_size=args.batch_size)

    if accelerator.is_main_process:
        print(f"Packing {len(ds_with_emb)} embedding sequences...")
        indexes = np.asarray(ds_with_emb["index"], dtype=np.int64)
        # embeddings is a list of variable-length float32 arrays of shape [V_i, H].
        embs = [np.asarray(e, dtype=np.float32) for e in ds_with_emb["embeddings"]]
        lengths = np.asarray([e.shape[0] for e in embs], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat = np.concatenate(embs, axis=0).astype(np.float32) if embs else np.zeros((0, 0), dtype=np.float32)

        out_path = args.embeddings_path
        if out_path.endswith(".npy"):
            out_path = out_path[:-4] + ".npz"
        elif not out_path.endswith(".npz"):
            out_path = out_path + ".npz"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(out_path, indexes=indexes, flat=flat, offsets=offsets)
        print(f"Saved {len(indexes)} entries, flat shape={flat.shape}, "
              f"offsets len={len(offsets)} -> {out_path}")


if __name__ == "__main__":
    run(parse_args())
