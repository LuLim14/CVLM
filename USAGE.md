# Usage

## Skip training and run evaluation only

`scripts/run_full_pipeline.sh` now honours a `SKIP_TRAIN` env var. Set it to `1`
to skip Step 1/2 and jump straight to checkpoint discovery + evaluation. The
script still needs `OUTPUT_DIR` to point at a run directory that already
contains `model_step_*.safetensors`.

```bash
SKIP_TRAIN=1 \
OUTPUT_DIR=/path/to/existing/run \
bash CVLM/scripts/run_full_pipeline.sh
```

Other eval knobs (`EVAL_MODES`, `EVAL_SPLIT`, `EVAL_MAX_SAMPLES`,
`EVAL_BATCH_SIZE`, `EVAL_COMPUTE_GEN`) behave as before.

## Live PNG training plots (no TensorBoard required)

Both `train_cvlm.py` and `train_sft.py` now write a CSV of training scalars and
render PNG dashboards to disk while training is running. View them with `scp`,
a file browser, or `feh` — no TB process needed.

Outputs in `<output_dir>/`:

| file              | content                                        |
|-------------------|------------------------------------------------|
| `metrics.csv`     | append-only: `step,loss,loss_avg,lr,grad_norm,batch_time` |
| `loss.png`        | loss + running-average loss vs step            |
| `lr.png`          | learning rate vs step                          |
| `grad_norm.png`   | gradient norm vs step (sync-step rows only)    |
| `batch_time.png`  | per-step wall time vs step                     |
| `dashboard.png`   | 2x2 grid of all four plots                     |

New flags (both trainers):

```
--plot_interval N    refresh PNGs every N optimizer steps (default 100; 0 = disable PNGs)
--csv_path PATH      override CSV location (default <output_dir>/metrics.csv)
```

The CSV is always written. Only PNG generation is gated by `--plot_interval`.

## SFT baseline pipeline

Train the same decoder used by CVLM with standard supervised fine-tuning on
`document + "\n\n" + question -> answer`, then evaluate with the same metric
suite as CVLM for direct comparison.

End-to-end:

```bash
OUTPUT_DIR=/path/to/sft_run \
  bash CVLM/scripts/run_sft_pipeline.sh
```

The script: trains `train_sft.py`, saves an HF-format checkpoint to
`OUTPUT_DIR`, then runs `eval_cvlm.py --mode sft --sft_model_path OUTPUT_DIR`
producing `OUTPUT_DIR/eval_sft.json`. JSON keys match a CVLM
`eval_baseline_llm_full.json` so the two are diffable.

Manual training:

```bash
python CVLM/src/train_sft.py \
  --output_dir /path/to/sft_run \
  --dataset_name sggetao/PwC \
  --model_name_or_path HuggingFaceTB/SmolLM-135M-Instruct \
  --epochs 2 --batch_size 2 --lr 1e-5 \
  --max_prompt_len 512 --max_answer_len 1024 \
  --plot_interval 100
```

Manual eval:

```bash
python CVLM/src/eval_cvlm.py \
  --mode sft \
  --sft_model_path /path/to/sft_run \
  --dataset_name sggetao/PwC --dataset_split test \
  --text_encoder_name answerdotai/ModernBERT-base \
  --max_prompt_len 512 --max_answer_len 1024 \
  --max_vision_len 256 --max_source_len 0 --compression_rate 4 \
  --output_json /path/to/sft_run/eval_sft.json \
  --compute_generation_metrics
```

The `--text_encoder_name` and vision-related flags are needed only because the
eval reuses `CvlmTrainDataset` for sample filtering — the SFT decoder itself
ignores the encoder/vision side.

`run_sft_pipeline.sh` env vars:

| var | default | meaning |
|---|---|---|
| `OUTPUT_DIR` | timestamped path | where the HF model + eval JSON go |
| `MODEL_NAME` | `HuggingFaceTB/SmolLM-135M-Instruct` | base decoder to fine-tune |
| `EPOCHS` | 2 | train epochs |
| `BATCH_SIZE` | 2 | per-rank batch |
| `LR` | 1e-5 | AdamW learning rate |
| `MAX_PROMPT_LEN` / `MAX_ANSWER_LEN` | 512 / 1024 | dataset-filter caps |
| `MAX_SAMPLES` | 0 | cap rows; 0 = all |
| `GRAD_ACCUM` | 1 | gradient accumulation |
| `LOG_INTERVAL` | 10 | optimizer steps per CSV/TB log |
| `SAVE_INTERVAL_STEPS` | 500 | HF Trainer save_steps; 0 = end only |
| `PLOT_INTERVAL` | 100 | optimizer steps between PNG refreshes |
| `NPROC` | 1 | torchrun GPUs (≥2 enables DDP) |
| `ENABLE_WARMUP` / `WARMUP_STEPS` | 1 / 100 | linear LR warmup |
| `EVAL_SPLIT` | `test` | split to evaluate on |
| `SKIP_TRAIN` | 0 | set to 1 to skip training and re-eval an existing dir |
