# Dataset + collation for CVLM (precomputed embeddings + HF text).

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def _load_embeddings(path: str):
    """Load embeddings saved by build_dataset/embed_dataset.py.

    Supports the new npz format (indexes, flat, offsets) and, for backwards
    compatibility, the legacy np.save(dict) format with per-row ragged arrays.
    Returns (indexes, get_row) where get_row(i) -> np.ndarray of shape [V_i, H].
    """
    if path.endswith(".npz") or path.endswith(".npz.npz"):
        data = np.load(path)
        indexes = np.asarray(data["indexes"], dtype=np.int64)
        flat = np.asarray(data["flat"], dtype=np.float32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)

        def get_row(i: int) -> np.ndarray:
            return flat[offsets[i]:offsets[i + 1]]

        return indexes, get_row

    # Legacy: np.save(dict(indexes=..., embeddings=list_of_arrays))
    raw = np.load(path, allow_pickle=True)
    data = raw.item()
    indexes = np.asarray(data["indexes"], dtype=np.int64)
    embeddings: List[np.ndarray] = list(data["embeddings"])

    def get_row(i: int) -> np.ndarray:
        e = np.asarray(embeddings[i], dtype=np.float32)
        if e.ndim == 1:
            e = e.reshape(1, -1)
        return e

    return indexes, get_row


def _sample_within_caps(
    vision_len: int,
    prompt_text: str,
    answer_text: str,
    tokenizer: PreTrainedTokenizer,
    max_prompt_len: int,
    max_answer_len: int,
    max_vision_len: int,
) -> bool:
    if vision_len <= 0 or vision_len > max_vision_len:
        return False
    p = tokenizer(prompt_text, add_special_tokens=False, truncation=False)
    a = tokenizer(answer_text, add_special_tokens=False, truncation=False)
    if len(p["input_ids"]) > max_prompt_len:
        return False
    if len(a["input_ids"]) > max_answer_len:
        return False
    return True


class CvlmTrainDataset(Dataset):
    """Pairs embeddings.npz rows with HuggingFace rows via indexes."""

    def __init__(
        self,
        embeddings_path: str,
        hf_dataset_name: str,
        hf_split: str,
        tokenizer_name: str,
        max_prompt_len: int,
        max_answer_len: int,
        max_vision_len: int,
    ) -> None:
        self._indexes, self._get_row = _load_embeddings(embeddings_path)
        self._hf: HFDataset = load_dataset(hf_dataset_name, split=hf_split)
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len
        self.max_vision_len = max_vision_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # padding_side is not actually used (collator pads manually left/right),
        # but set it for any ad-hoc tokenizer calls.
        self.tokenizer.padding_side = "left"

        self._row_indices: List[int] = []
        n_src = len(self._indexes)
        for i in tqdm(range(n_src), desc="Filtering CVLM samples"):
            emb_i = self._get_row(i)
            row_id = int(self._indexes[i])
            record = self._hf[row_id]
            if _sample_within_caps(
                emb_i.shape[0] if emb_i.ndim == 2 else 1,
                record["input"],
                record["answer"],
                self.tokenizer,
                self.max_prompt_len,
                self.max_answer_len,
                self.max_vision_len,
            ):
                self._row_indices.append(i)
        n_kept = len(self._row_indices)
        print(
            f"CvlmTrainDataset: kept {n_kept}/{n_src} samples "
            f"(prompt_tokens<={max_prompt_len}, answer_tokens<={max_answer_len}, "
            f"vision_len<={max_vision_len}; no truncation)"
        )
        if n_kept == 0:
            raise RuntimeError(
                "No samples passed length filters; increase caps or check data."
            )

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_i = self._row_indices[idx]
        emb = self._get_row(src_i).astype(np.float32, copy=False)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim != 2:
            raise ValueError(f"Expected embedding shape [D] or [V, D], got {emb.shape}")
        input_embeds = torch.from_numpy(np.ascontiguousarray(emb))
        row_id = int(self._indexes[src_i])
        record = self._hf[row_id]
        p = self.tokenizer(record["input"], add_special_tokens=False, truncation=False, return_tensors="pt")
        a = self.tokenizer(record["answer"], add_special_tokens=False, truncation=False, return_tensors="pt")
        return {
            "input_embeds": input_embeds,
            "prompt_ids": p["input_ids"].squeeze(0).long(),
            "answer_ids": a["input_ids"].squeeze(0).long(),
        }


def make_collate_fn(pad_token_id: int) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_cvlm_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        b = len(samples)
        # Vision: right-pad (embedding rows), record real lengths for the attn mask.
        max_v = max(s["input_embeds"].shape[0] for s in samples)
        d = samples[0]["input_embeds"].shape[1]
        input_embeds = torch.zeros(b, max_v, d, dtype=samples[0]["input_embeds"].dtype)
        vision_lens = torch.zeros(b, dtype=torch.long)
        for i, s in enumerate(samples):
            t = s["input_embeds"]
            input_embeds[i, : t.shape[0], :] = t
            vision_lens[i] = t.shape[0]

        # Prompt: LEFT-pad so real prompt tokens sit flush against the vision block.
        max_p = max(s["prompt_ids"].shape[0] for s in samples)
        # Answer: right-pad (standard).
        max_a = max(s["answer_ids"].shape[0] for s in samples)

        prompt_ids = torch.full((b, max_p), pad_token_id, dtype=torch.long)
        answer_ids = torch.full((b, max_a), pad_token_id, dtype=torch.long)
        answer_labels = torch.full((b, max_a), -100, dtype=torch.long)
        prompt_mask = torch.zeros(b, max_p, dtype=torch.long)
        vision_mask = torch.zeros(b, max_v, dtype=torch.long)
        answer_mask = torch.zeros(b, max_a, dtype=torch.long)

        for i, s in enumerate(samples):
            pids = s["prompt_ids"]
            aids = s["answer_ids"]
            pl, al = pids.shape[0], aids.shape[0]
            # left-pad prompt
            prompt_ids[i, max_p - pl:] = pids
            prompt_mask[i, max_p - pl:] = 1
            # right-pad answer
            answer_ids[i, :al] = aids
            answer_labels[i, :al] = aids
            answer_mask[i, :al] = 1
            # vision mask: first vision_lens[i] positions are valid
            vision_mask[i, : vision_lens[i]] = 1

        attention_mask = torch.cat([prompt_mask, vision_mask, answer_mask], dim=1)

        return {
            "input_embeds": input_embeds,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "answer_labels": answer_labels,
            "attention_mask": attention_mask,
            "vision_lens": vision_lens,
        }

    return collate_cvlm_batch
