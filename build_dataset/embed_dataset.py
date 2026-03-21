import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_name: str = field(init=False)
    data_path: str = field(init=False)
    embeddings_path: str = field(init=False)
    seed: int = field(init=False)
    batch_size: int = field(init=False)
    max_length: int = field(init=False)

    def __post_init__(self):
        self.model_name = "answerdotai/ModernBERT-base"
        self.data_path = "sggetao/PwC"
        self.embeddings_path = "/workspace-SR004.nfs2/acherepanov/compress_project/compress_project/data/dataset_100_samples/embeddings.npy"
        self.seed = 42
        self.batch_size = 1024
        self.max_length = 2048

        self.num_proc = (
            os.cpu_count() // torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1
        )


def run(config: Config):
    accelerator = Accelerator()
    set_seed(config.seed)
    rng = np.random.default_rng(seed=config.seed)

    print(f"Accelerator: {accelerator.device}")

    model = AutoModel.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        # attention_implementation="flash_attention_2", # TODO: may be i dont need this for embeddings and simplify
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"model loaded on {model.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
        padding_side="right",
        truncation_side="right",
        model_max_length=config.max_length,
    )

    def does_fit(sample):
        inputs = tokenizer(
            sample["input"],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        # print(inputs[0])
        # print(inputs[0].ids)
        return len(inputs["input_ids"]) <= config.max_length

    with accelerator.main_process_first():
        ds = load_dataset(config.data_path, split="train")
        # ds = ds.select(range(100_000))  # TODO: for test, need to remove
        ds = ds.select(range(100))  # TODO: for test, need to remove

    ds = ds.add_column("index", range(len(ds)))

    len_before_train = len(ds)
    ds = ds.filter(does_fit, num_proc=config.num_proc)
    ds = ds.shuffle(seed=config.seed)
    len_after_train = len(ds)

    if accelerator.is_main_process:
        logger.info(f"Length before train: {len_before_train}")
        logger.info(f"Length after train: {len_after_train}")

    def get_embeddings(batch):
        model.eval()
        text_inputs = [str(item) for item in batch["input"]]

        inputs = tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Take [CLS] token at index 0
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        return {"embeddings": batch_embeddings.to(torch.float32).cpu().numpy()}

    ds_with_embeddings = ds.map(
        get_embeddings, batched=True, batch_size=config.batch_size
    )

    if accelerator.is_main_process:
        logger.info(f"Length of embeddings: {len(ds)}")
        logger.info(f"Columns after embeddings: {ds.column_names}")
        logger.info(f"Sample of embeddings: {ds[0]}")

        logger.info("Packing data into single dictionary...")

        # Create a dictionary holding both arrays
        # This preserves types: indexes stay Int, embeddings stay Float
        data_payload = {
            "indexes": np.array(ds_with_embeddings["index"]),
            "embeddings": np.array(ds_with_embeddings["embeddings"]),
        }

        logger.info(f"Embeddings saved to {config.embeddings_path}")
        logger.info(f"Saving to {config.embeddings_path}...")

        # Save the dictionary object
        np.save(config.embeddings_path, data_payload)

        logger.info("Done.")


if __name__ == "__main__":
    config = Config()
    run(config)
