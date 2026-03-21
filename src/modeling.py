# ICAE that supports multi span concat

import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers
from peft import get_peft_model
from safetensors.torch import load_file
from torch.nn.functional import gelu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="HuggingFaceTB/SmolLM-135M-Instruct")
    vision_encoder_name: str = field(default="google/vit-base-patch16-224")
    embed_input_dim: int = field(default=768, metadata={"help": "Dimension of input embeddings (e.g., ModernBERT=768)"})
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    fixed_mem_size: int = field(
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={
            "help": "Add a special token for the prompt of language modeling; default: False"
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "Enable bfloat16 precision for the model"
        },
    )


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class Projector(torch.nn.Module):
    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.model_args = model_args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_proj = torch.nn.Linear(
            in_features=self.input_dim, out_features=4 * self.input_dim
        )
        self.down_proj = torch.nn.Linear(
            in_features=4 * self.input_dim, out_features=self.output_dim
        )
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        return x


class CVLM(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.vision_encoder_name = model_args.vision_encoder_name
        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
            if training_args.bf16 is False
            else torch.bfloat16,
            # attn_implementation="flash_attention_2", # TODO: need to use
        )
        self.vision_encoder = ViTModel.from_pretrained(self.vision_encoder_name)
        self.text_projector = Projector(
            model_args,
            model_args.embed_input_dim,
            self.vision_encoder.config.hidden_size,
        )
        self.vision_projector = Projector(
            model_args,
            self.vision_encoder.config.hidden_size,
            self.encoder.config.hidden_size,
        )
        print(f"!!!======training_args.bf16: {training_args.bf16}")
        if training_args.bf16:
            self.text_projector.to(torch.bfloat16)
            self.vision_projector.to(torch.bfloat16)
            self.vision_encoder.to(torch.bfloat16)

        self.training = self.model_args.train

        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
            if training_args.bf16 is False
            else torch.bfloat16,
            # attn_implementation="flash_attention_2", # TODO: need to use
        )

        self.vocab_size = self.encoder.config.vocab_size + 1  # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = (
            self.vocab_size + self.mem_size
        )  # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2

        self.encoder.resize_token_embeddings(self.vocab_size_with_mem + 3)

        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2

        self.dim = self.encoder.config.hidden_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_token_embed = nn.Embedding(
            self.mem_size + 3, self.dim, padding_idx=None
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.append_sequence = torch.arange(
            self.vocab_size,
            self.vocab_size + self.mem_size,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)  # mem tokens

        if self.training:
            self.init()

    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(
            total_length / (self.mem_size * self.mean_compression_rate)
        )
        return num_segments

    def _encode_vision(self, input_embeds):
        """Route pre-computed embeddings through text_projector -> ViT encoder -> vision_projector."""
        vit_input = self.text_projector(input_embeds)                           # [B, seq, vit_dim]
        vit_output = self.vision_encoder.encoder(vit_input).last_hidden_state   # [B, seq, vit_dim]
        vit_output = self.vision_encoder.layernorm(vit_output)
        return self.vision_projector(vit_output)                                # [B, seq, llm_dim]

    def forward(
        self,
        input_embeds: torch.FloatTensor = None,
        prompt_ids: torch.LongTensor = None,
        answer_ids: torch.LongTensor = None,
    ):
        batch_size = prompt_ids.size(0)

        # Vision pipeline: pre-computed embeddings -> ViT -> LLM space
        vision_embeds = self._encode_vision(input_embeds)  # [B, V, llm_dim]

        # Token embeddings
        prompt_embs = self.encoder.get_input_embeddings()(prompt_ids)   # [B, P, llm_dim]
        answer_embs = self.encoder.get_input_embeddings()(answer_ids)   # [B, A, llm_dim]

        # [prompt, vision, answer] — causal decoder attends left-to-right
        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        # Labels: -100 for prompt + vision positions, answer_ids for answer
        P = prompt_ids.size(1)
        V = vision_embeds.size(1)
        ignore = torch.full((batch_size, P + V), -100, dtype=answer_ids.dtype, device=answer_ids.device)
        labels = torch.cat([ignore, answer_ids], dim=1)

        # Decoder forward + shifted cross-entropy loss
        decoder_outputs = self.decoder(inputs_embeds=decoder_input, output_hidden_states=True)
        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        target = labels[:, 1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_embeds, prompt_ids, max_new_tokens=512, temperature=0.0):
        vision_embeds = self._encode_vision(input_embeds)
        prompt_embs = self.encoder.get_input_embeddings()(prompt_ids)
        decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)

        gen_kwargs = dict(
            inputs_embeds=decoder_input,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        return self.decoder.generate(**gen_kwargs)

    def tokens_to_embeddings(
        self, token_ids
    ):  # input_tokens can be either normal tokens and special tokens
        embeddings = self.encoder.get_input_embeddings()(token_ids)
        return embeddings

    def _compress(
        self, input_ids: torch.LongTensor = None
    ):  # for inference; compress a fixed length of input into memory slots
        # TODO: how to do inference, how to use this compress function?

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim))

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat(
                [segment_input_ids, self.append_sequence], dim=1
            )
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(
                segment_input_ids
            )
            segment_input_embedding[mem_flag] = self.memory_token_embed(
                segment_input_ids[mem_flag] - self.vocab_size
            ).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(
                inputs_embeds=segment_input_embedding, output_hidden_states=True
            )
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[
                segment_idx * self.mem_size : self.mem_size * (segment_idx + 1)
            ] = segment_compress_outputs[mem_flag]

            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()

        return compress_outputs
