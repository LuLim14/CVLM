import numpy as np
from datasets import load_dataset

# 1. Load the file (allow_pickle=True is required for dictionaries)
data = np.load(
    "/home/jovyan/shares/SR008.fs2/acherepanov/compress_project/data/test_dataset/embeddings.npy",
    allow_pickle=True,
).item()

saved_indexes = data[
    "indexes"
]  # e.g., [0, 2, 5, ...] (skipped rows that were filtered)
saved_embeddings = data["embeddings"]  # Corresponding vectors

# 2. Load the ORIGINAL dataset (same split, same config)
# IMPORTANT: Do not apply filter() here. We need the raw data to match indices.
original_ds = load_dataset("sggetao/PwC", split="train")

array_pos = 0

# 1. Get the Original Index from your saved array
original_db_id = saved_indexes[array_pos]

# 2. Retrieve from the original dataset using standard list indexing
# Because we added the index column BEFORE shuffling/filtering,
# dataset index matches the row number.
record = original_ds[int(original_db_id)]

print(f"ID: {original_db_id}")
print(f"Input Text: {record['input']}")
print(f"Saved Embedding: {saved_embeddings[array_pos]}")
