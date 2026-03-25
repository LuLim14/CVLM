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


def _sample_within_caps(
    emb: np.ndarray,
    prompt_text: str,
    answer_text: str,
    tokenizer: PreTrainedTokenizer,
    max_prompt_len: int,
    max_answer_len: int,
    max_vision_len: int,
) -> bool:
    """True iff vision length and token counts fit caps (no truncation)."""
    if emb.ndim == 1:
        vision_len = 1
    elif emb.ndim == 2:
        vision_len = emb.shape[0]
    else:
        return False
    if vision_len > max_vision_len:
        return False
    p = tokenizer(prompt_text, add_special_tokens=False, truncation=False)
    a = tokenizer(answer_text, add_special_tokens=False, truncation=False)
    if len(p["input_ids"]) > max_prompt_len:
        return False
    if len(a["input_ids"]) > max_answer_len:
        return False
    return True


class CvlmTrainDataset(Dataset):
    """Pairs embeddings.npy rows with HuggingFace rows via indexes."""

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
        raw = np.load(embeddings_path, allow_pickle=True)
        data = raw.item()
        self._indexes: np.ndarray = data["indexes"]
        self._embeddings: List[np.ndarray] = data["embeddings"]
        self._hf: HFDataset = load_dataset(hf_dataset_name, split=hf_split)
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len
        self.max_vision_len = max_vision_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self._row_indices: List[int] = []
        n_src = len(self._indexes)
        for i in tqdm(range(n_src)):
            emb_i = np.asarray(self._embeddings[i])
            row_id = int(self._indexes[i])
            record = self._hf[row_id]
            if _sample_within_caps(
                emb_i,
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
        emb = np.asarray(self._embeddings[src_i], dtype=np.float32)
        # Per-sequence [V, D] from token pooling, or a single pooled vector [D]
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim != 2:
            raise ValueError(
                f"Expected embedding shape [D] or [V, D], got shape {emb.shape}"
            )
        input_embeds = torch.from_numpy(np.ascontiguousarray(emb))
        row_id = int(self._indexes[src_i])
        record = self._hf[row_id]
        p = self.tokenizer(
            record["input"],
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        a = self.tokenizer(
            record["answer"],
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        prompt_ids = p["input_ids"].squeeze(0).long()
        answer_ids = a["input_ids"].squeeze(0).long()
        return {
            "input_embeds": input_embeds,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
        }


def make_collate_fn(pad_token_id: int) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_cvlm_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_v = max(s["input_embeds"].shape[0] for s in samples)
        d = samples[0]["input_embeds"].shape[1]
        b = len(samples)
        input_embeds = torch.zeros(b, max_v, d, dtype=samples[0]["input_embeds"].dtype)
        for i, s in enumerate(samples):
            t = s["input_embeds"]
            input_embeds[i, : t.shape[0], :] = t

        max_p = max(s["prompt_ids"].shape[0] for s in samples)
        max_a = max(s["answer_ids"].shape[0] for s in samples)
        prompt_ids = torch.full((b, max_p), pad_token_id, dtype=torch.long)
        answer_ids = torch.full((b, max_a), pad_token_id, dtype=torch.long)
        answer_labels = torch.full((b, max_a), -100, dtype=torch.long)
        for i, s in enumerate(samples):
            pids = s["prompt_ids"]
            aids = s["answer_ids"]
            pl, al = pids.shape[0], aids.shape[0]
            prompt_ids[i, :pl] = pids
            answer_ids[i, :al] = aids
            answer_labels[i, :al] = aids
        return {
            "input_embeds": input_embeds,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "answer_labels": answer_labels,
        }

    return collate_cvlm_batch
