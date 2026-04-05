# Dataset + collation for CVLM.
#
# The on-the-fly variant: the dataset yields raw token ids for (source, prompt,
# answer). The model's frozen text encoder embeds `source_ids` at forward time,
# so there is no precomputed embeddings cache anywhere.

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


class CvlmTrainDataset(Dataset):
    """Raw-text dataset: yields encoder source ids + decoder prompt/answer ids."""

    def __init__(
        self,
        hf_dataset_name: str,
        hf_split: str,
        decoder_tokenizer_name: str,
        encoder_tokenizer_name: str,
        max_prompt_len: int,
        max_answer_len: int,
        max_source_len: int,
        max_samples: int = 0,
    ) -> None:
        self._hf: HFDataset = load_dataset(hf_dataset_name, split=hf_split)
        if max_samples > 0:
            self._hf = self._hf.select(range(min(max_samples, len(self._hf))))
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len
        self.max_source_len = max_source_len

        self._dec_tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            decoder_tokenizer_name, use_fast=False
        )
        if self._dec_tok.pad_token is None:
            self._dec_tok.pad_token = self._dec_tok.eos_token
        self._enc_tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            encoder_tokenizer_name, use_fast=True, trust_remote_code=True
        )
        if self._enc_tok.pad_token is None:
            self._enc_tok.pad_token = self._enc_tok.eos_token or self._enc_tok.cls_token
        self.dec_pad_id: int = self._dec_tok.pad_token_id
        self.enc_pad_id: int = self._enc_tok.pad_token_id

        # Vectorised length pass: one batched fast-tokenizer call per field.
        # PwC rows have "input" (source document) and "answer".
        inputs = list(self._hf["input"])
        answers = list(self._hf["answer"])

        print(f"Tokenising {len(inputs)} samples to compute length filter...")
        enc_lens = self._batched_lengths(self._enc_tok, inputs)       # source lens in encoder tokens
        dec_input_lens = self._batched_lengths(self._dec_tok, inputs) # prompt lens in decoder tokens
        dec_answer_lens = self._batched_lengths(self._dec_tok, answers)

        keep = []
        for i in range(len(inputs)):
            if enc_lens[i] <= 0 or enc_lens[i] > max_source_len:
                continue
            if dec_input_lens[i] > max_prompt_len:
                continue
            if dec_answer_lens[i] > max_answer_len:
                continue
            keep.append(i)
        self._row_indices: List[int] = keep
        print(
            f"CvlmTrainDataset: kept {len(keep)}/{len(inputs)} samples "
            f"(source_tokens<={max_source_len}, prompt_tokens<={max_prompt_len}, "
            f"answer_tokens<={max_answer_len})"
        )
        if not keep:
            raise RuntimeError("No samples passed length filters; loosen caps.")

    @staticmethod
    def _batched_lengths(tokenizer: PreTrainedTokenizer, texts: List[str], batch: int = 1024) -> List[int]:
        out: List[int] = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                [str(t) for t in texts[start:start + batch]],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            out.extend(len(ids) for ids in enc["input_ids"])
        return out

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_id = self._row_indices[idx]
        record = self._hf[row_id]
        src = self._enc_tok(
            str(record["input"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors=None,
        )
        prompt = self._dec_tok(
            str(record["input"]), add_special_tokens=False, truncation=False
        )
        answer = self._dec_tok(
            str(record["answer"]), add_special_tokens=False, truncation=False
        )
        return {
            "source_ids": torch.as_tensor(src["input_ids"], dtype=torch.long),
            "prompt_ids": torch.as_tensor(prompt["input_ids"], dtype=torch.long),
            "answer_ids": torch.as_tensor(answer["input_ids"], dtype=torch.long),
        }


def make_collate_fn(
    dec_pad_id: int,
    enc_pad_id: int,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_cvlm_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        b = len(samples)

        # Source (encoder input): right-pad with encoder pad id.
        max_s = max(s["source_ids"].shape[0] for s in samples)
        source_ids = torch.full((b, max_s), enc_pad_id, dtype=torch.long)
        source_attention_mask = torch.zeros(b, max_s, dtype=torch.long)

        # Prompt: LEFT-pad so real tokens sit flush against the vision block.
        max_p = max(s["prompt_ids"].shape[0] for s in samples)
        prompt_ids = torch.full((b, max_p), dec_pad_id, dtype=torch.long)
        prompt_mask = torch.zeros(b, max_p, dtype=torch.long)

        # Answer: right-pad (standard).
        max_a = max(s["answer_ids"].shape[0] for s in samples)
        answer_ids = torch.full((b, max_a), dec_pad_id, dtype=torch.long)
        answer_labels = torch.full((b, max_a), -100, dtype=torch.long)
        answer_mask = torch.zeros(b, max_a, dtype=torch.long)

        for i, s in enumerate(samples):
            sids, pids, aids = s["source_ids"], s["prompt_ids"], s["answer_ids"]
            sl, pl, al = sids.shape[0], pids.shape[0], aids.shape[0]
            source_ids[i, :sl] = sids
            source_attention_mask[i, :sl] = 1
            prompt_ids[i, max_p - pl:] = pids
            prompt_mask[i, max_p - pl:] = 1
            answer_ids[i, :al] = aids
            answer_labels[i, :al] = aids
            answer_mask[i, :al] = 1

        return {
            "source_ids": source_ids,
            "source_attention_mask": source_attention_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "answer_ids": answer_ids,
            "answer_labels": answer_labels,
            "answer_mask": answer_mask,
        }

    return collate_cvlm_batch
