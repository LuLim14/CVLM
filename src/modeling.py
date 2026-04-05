from dataclasses import dataclass, field

import torch
import torch.nn as nn
import transformers
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, ViTModel


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="HuggingFaceTB/SmolLM-135M-Instruct")
    vision_encoder_name: str = field(default="google/vit-base-patch16-224")
    text_encoder_name: str = field(
        default="answerdotai/ModernBERT-base",
        metadata={"help": "HF name of the frozen text encoder that turns source tokens into the vision sequence."},
    )
    embed_input_dim: int = field(default=768, metadata={"help": "Hidden size of the text encoder output (ModernBERT-base=768)."})
    max_vision_len: int = field(default=512, metadata={"help": "Max compressed-vision-sequence length; sizes the learned positional embedding."})
    compression_rate: int = field(default=4, metadata={"help": "Average-pool this many source tokens into one vision token."})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    restore_from: str = field(
        default="",
        metadata={"help": "Safetensors checkpoint to preload trainable sub-modules from."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Enable bfloat16 precision for the model"},
    )


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    pct = 100 * trainable_parameters / all_param if all_param else 0.0
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {pct:.4f}"
    )


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class Projector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.up_proj = torch.nn.Linear(in_features=input_dim, out_features=4 * input_dim)
        self.down_proj = torch.nn.Linear(in_features=4 * input_dim, out_features=output_dim)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        return x


def _chunked_mean_pool(
    hidden: torch.Tensor,
    source_attention_mask: torch.Tensor,
    compression_rate: int,
    max_vision_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool contiguous chunks of `compression_rate` real tokens per sample.

    Args:
        hidden: [B, L, H] encoder output (any dtype).
        source_attention_mask: [B, L] long/int mask (1 for real, 0 for pad).
        compression_rate: pool size.
        max_vision_len: hard cap on output V per sample (truncates longer docs).

    Returns:
        pooled: [B, V_max, H] where V_max = max V_i across batch.
        vision_attention_mask: [B, V_max] long (1 for real pooled slots, 0 for pad).
    """
    B, L, H = hidden.shape
    cr = max(int(compression_rate), 1)
    lens = source_attention_mask.sum(dim=1).tolist()

    V_list = []
    for l_real in lens:
        v = (l_real + cr - 1) // cr
        if v > max_vision_len:
            v = max_vision_len
        V_list.append(v)
    V_max = max(V_list) if V_list else 1
    V_max = max(V_max, 1)

    pooled = torch.zeros((B, V_max, H), dtype=hidden.dtype, device=hidden.device)
    v_mask = torch.zeros((B, V_max), dtype=torch.long, device=hidden.device)

    for b in range(B):
        L_real = int(lens[b])
        V_i = int(V_list[b])
        if L_real == 0 or V_i == 0:
            continue
        # If the sample would need more than max_vision_len chunks, truncate.
        L_use = min(L_real, V_i * cr)
        h = hidden[b, :L_use]                       # [L_use, H]
        pad = V_i * cr - L_use
        if pad > 0:
            h = torch.nn.functional.pad(h, (0, 0, 0, pad))  # zero-pad tail
        chunked = h.view(V_i, cr, H).mean(dim=1)    # [V_i, H]
        if pad > 0:
            # Last chunk's mean was divided by cr, should be divided by (cr - pad).
            true_last = cr - pad
            chunked[-1] = chunked[-1] * cr / true_last
        pooled[b, :V_i] = chunked
        v_mask[b, :V_i] = 1

    return pooled, v_mask


class CVLM(torch.nn.Module):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        # Decoder (frozen LLM)
        self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=compute_dtype)

        # Text encoder (frozen; runs in forward pass instead of offline preprocess).
        self.text_encoder = AutoModel.from_pretrained(
            model_args.text_encoder_name,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        # Stash the encoder's hidden size for the text_projector input dim.
        enc_hidden = self.text_encoder.config.hidden_size

        self.vision_encoder = ViTModel.from_pretrained(model_args.vision_encoder_name)
        self.text_projector = Projector(
            enc_hidden,
            self.vision_encoder.config.hidden_size,
        )
        self.vision_projector = Projector(
            self.vision_encoder.config.hidden_size,
            self.decoder.config.hidden_size,
        )
        # Learned positional embedding for the compressed vision sequence.
        self.max_vision_len = int(model_args.max_vision_len)
        self.compression_rate = int(model_args.compression_rate)
        self.vision_pos_embed = nn.Embedding(self.max_vision_len, self.vision_encoder.config.hidden_size)
        nn.init.normal_(self.vision_pos_embed.weight, std=0.02)

        if training_args.bf16:
            self.text_projector.to(torch.bfloat16)
            self.vision_projector.to(torch.bfloat16)
            self.vision_encoder.to(torch.bfloat16)
            self.vision_pos_embed.to(torch.bfloat16)

        self.dim = self.decoder.config.hidden_size
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_encoder_name, use_fast=True, trust_remote_code=True
        )

        self._init_for_training = model_args.train
        if self._init_for_training:
            self._init_training()
        else:
            # Eval: freeze + eval everything that is frozen during training too, so
            # behaviour matches regardless of how the model is constructed.
            self._freeze_frozen_submodules()

    def _freeze_frozen_submodules(self):
        freeze_model(self.decoder)
        self.decoder.eval()
        freeze_model(self.text_encoder)
        self.text_encoder.eval()

    def _init_training(self):
        print("Freezing the decoder and text encoder...")
        self._freeze_frozen_submodules()
        # Sanity assertion: the two landmines from the plan.
        assert all(not p.requires_grad for p in self.decoder.parameters()), \
            "decoder must be frozen"
        assert all(not p.requires_grad for p in self.text_encoder.parameters()), \
            "text_encoder must be frozen"
        print_trainable_parameters(self)
        if self.training_args.restore_from:
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            # Allow partial loads (trainable-only checkpoints).
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"  restored; missing={len(missing)} unexpected={len(unexpected)}")

    def _encode_vision(self, source_input_ids, source_attention_mask):
        """source text ids -> frozen text encoder -> chunked pool -> ViT encoder (+pos) -> vision_projector."""
        # Encoder runs inside no_grad so the 12-layer activation chain is not kept
        # alive for backward. Its params have requires_grad=False, but downstream
        # trainable modules would otherwise force autograd to build the full graph.
        with torch.no_grad():
            enc_out = self.text_encoder(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
            ).last_hidden_state                     # [B, L, H_enc] in compute_dtype
        enc_out = enc_out.detach()

        pooled, vision_attention_mask = _chunked_mean_pool(
            enc_out,
            source_attention_mask,
            compression_rate=self.compression_rate,
            max_vision_len=self.max_vision_len,
        )                                           # [B, V_max, H_enc], [B, V_max]

        vit_input = self.text_projector(pooled)     # [B, V_max, vit_dim]
        V = vit_input.size(1)
        if V > self.max_vision_len:
            raise ValueError(
                f"Vision sequence length {V} exceeds max_vision_len={self.max_vision_len}; "
                "increase max_vision_len or compression_rate."
            )
        pos = torch.arange(V, device=vit_input.device)
        vit_input = vit_input + self.vision_pos_embed(pos).unsqueeze(0).to(vit_input.dtype)
        vit_output = self.vision_encoder.encoder(vit_input).last_hidden_state   # [B, V, vit_dim]
        vit_output = self.vision_encoder.layernorm(vit_output)
        return self.vision_projector(vit_output), vision_attention_mask         # [B, V, llm_dim], [B, V]

    def forward(
        self,
        source_input_ids: torch.LongTensor = None,
        source_attention_mask: torch.LongTensor = None,
        prompt_ids: torch.LongTensor = None,
        answer_ids: torch.LongTensor = None,
        answer_labels: torch.LongTensor = None,
        prompt_mask: torch.LongTensor = None,
        answer_mask: torch.LongTensor = None,
    ):
        batch_size = prompt_ids.size(0)

        vision_embeds, vision_mask = self._encode_vision(source_input_ids, source_attention_mask)
        # [B, V, llm_dim], [B, V]

        embed_layer = self.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)   # [B, P, llm_dim]
        answer_embs = embed_layer(answer_ids)   # [B, A, llm_dim]

        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        # Combined attention mask: prompt_mask | vision_mask | answer_mask.
        # (Collator provides the three sub-masks so we don't need to reconstruct them.)
        if prompt_mask is None or answer_mask is None:
            raise ValueError("prompt_mask and answer_mask are required in forward().")
        attention_mask = torch.cat([prompt_mask, vision_mask, answer_mask], dim=1)

        labels_src = answer_labels if answer_labels is not None else answer_ids
        P = prompt_ids.size(1)
        V = vision_embeds.size(1)
        ignore = torch.full((batch_size, P + V), -100, dtype=labels_src.dtype, device=labels_src.device)
        labels = torch.cat([ignore, labels_src], dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        target = labels[:, 1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target)
        return {
            "loss": loss,
            "logits": logits,
            "vision_mask": vision_mask,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        source_input_ids,
        source_attention_mask,
        prompt_ids,
        prompt_mask,
        max_new_tokens=512,
        temperature=0.0,
    ):
        vision_embeds, vision_mask = self._encode_vision(source_input_ids, source_attention_mask)
        prompt_embs = self.decoder.get_input_embeddings()(prompt_ids)
        decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, vision_mask], dim=1)

        gen_kwargs = dict(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        return self.decoder.generate(**gen_kwargs)
