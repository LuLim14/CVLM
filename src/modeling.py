from dataclasses import dataclass, field

import torch
import torch.nn as nn
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, ViTModel


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="HuggingFaceTB/SmolLM-135M-Instruct")
    vision_encoder_name: str = field(default="google/vit-base-patch16-224")
    embed_input_dim: int = field(default=768, metadata={"help": "Dimension of input embeddings (e.g., ModernBERT=768)"})
    max_vision_len: int = field(default=512, metadata={"help": "Max vision-sequence length used to size the learned positional embedding."})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    fixed_mem_size: int = field(
        default=128,
        metadata={"help": "Enabling the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
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


class CVLM(torch.nn.Module):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=compute_dtype)
        self.vision_encoder = ViTModel.from_pretrained(model_args.vision_encoder_name)
        self.text_projector = Projector(
            model_args.embed_input_dim,
            self.vision_encoder.config.hidden_size,
        )
        self.vision_projector = Projector(
            self.vision_encoder.config.hidden_size,
            self.decoder.config.hidden_size,
        )
        # Learned positional embedding for the compressed vision sequence.
        # ViT.encoder has no intrinsic position signal, so without this the whole
        # vision block would be permutation-invariant over V.
        self.max_vision_len = int(model_args.max_vision_len)
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

        self._init_for_training = model_args.train
        if self._init_for_training:
            self._init_training()

    def _init_training(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if self.training_args.restore_from:
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def _encode_vision(self, input_embeds):
        """Route pre-computed embeddings through text_projector -> ViT encoder (+pos) -> vision_projector."""
        vit_input = self.text_projector(input_embeds)                           # [B, V, vit_dim]
        V = vit_input.size(1)
        if V > self.max_vision_len:
            raise ValueError(
                f"Vision sequence length {V} exceeds max_vision_len={self.max_vision_len}; "
                "increase ModelArguments.max_vision_len or tighten the dataset filter."
            )
        pos = torch.arange(V, device=vit_input.device)
        vit_input = vit_input + self.vision_pos_embed(pos).unsqueeze(0).to(vit_input.dtype)
        vit_output = self.vision_encoder.encoder(vit_input).last_hidden_state   # [B, V, vit_dim]
        vit_output = self.vision_encoder.layernorm(vit_output)
        return self.vision_projector(vit_output)                                # [B, V, llm_dim]

    def forward(
        self,
        input_embeds: torch.FloatTensor = None,
        prompt_ids: torch.LongTensor = None,
        answer_ids: torch.LongTensor = None,
        answer_labels: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        batch_size = prompt_ids.size(0)

        # Vision pipeline: pre-computed embeddings -> ViT -> LLM space
        vision_embeds = self._encode_vision(input_embeds)  # [B, V, llm_dim]

        # Token embeddings come from the decoder's own table, otherwise we feed
        # vectors the frozen decoder does not recognise.
        embed_layer = self.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)   # [B, P, llm_dim]
        answer_embs = embed_layer(answer_ids)   # [B, A, llm_dim]

        # [prompt, vision, answer] — causal decoder attends left-to-right
        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        # Labels: -100 for prompt + vision positions, answer_labels for answer.
        labels_src = answer_labels if answer_labels is not None else answer_ids
        P = prompt_ids.size(1)
        V = vision_embeds.size(1)
        ignore = torch.full((batch_size, P + V), -100, dtype=labels_src.dtype, device=labels_src.device)
        labels = torch.cat([ignore, labels_src], dim=1)

        # Decoder forward + shifted cross-entropy loss
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        target = labels[:, 1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_embeds, prompt_ids, attention_mask=None, max_new_tokens=512, temperature=0.0):
        vision_embeds = self._encode_vision(input_embeds)
        prompt_embs = self.decoder.get_input_embeddings()(prompt_ids)
        decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)

        gen_kwargs = dict(
            inputs_embeds=decoder_input,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        return self.decoder.generate(**gen_kwargs)
