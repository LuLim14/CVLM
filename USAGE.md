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

## Trackio logging

`SummaryWriter` has been replaced by [trackio](https://github.com/gradio-app/trackio)
for the live dashboard. CSV (`metrics.csv`) and PNG (`dashboard.png`) outputs are
unchanged and continue to work even when trackio is disabled.

### Install

```bash
pip install trackio
```

### Run a training with trackio

```bash
TRACKIO_PROJECT=cvlm \
TRACKIO_RUN_NAME=my-run \
OUTPUT_DIR=/path/to/run \
bash CVLM/scripts/run_full_pipeline.sh
```

Defaults: `TRACKIO_PROJECT=cvlm`, `TRACKIO_RUN_NAME=$(basename "$OUTPUT_DIR")`.
The same env vars are honoured by `scripts/run_sft_pipeline.sh`.

Per-binary CLI flags (equivalent to env vars) are available on
`train_cvlm.py`, `train_sft.py`, and `eval_cvlm.py`:

```
--trackio_project NAME      # default: cvlm
--trackio_run_name NAME     # default: output dir basename
--trackio_space_id user/repo  # optional HF Space to host the dashboard
--trackio_disable           # turn trackio off entirely
```

### View the dashboard

Local UI:

```bash
trackio show --project cvlm
```

Host on HF Spaces (auto-creates the Space the first time):

```bash
TRACKIO_SPACE_ID=username/cvlm-runs bash CVLM/scripts/run_full_pipeline.sh
```

### Disable trackio

```bash
TRACKIO_DISABLE=1 bash CVLM/scripts/run_full_pipeline.sh
# or per-binary:
python CVLM/src/train_cvlm.py --trackio_disable ...
```

CSV + PNG outputs remain available when trackio is disabled.

### Metrics keys

| Phase | Keys |
| --- | --- |
| Train (CVLM) | `train/loss`, `train/loss_avg`, `train/lr`, `train/grad_norm`, `train/batch_time` |
| Train (SFT) | `train/loss`, `train/lr`, `train/grad_norm`, `train/epoch` |
| Eval | `eval/<metric>` (loss, ppl, ROUGE, BLEU-4, EM, compression stats) + `eval/compression_ratio_dist` (histogram if supported, else `*_mean/std/min/max`) |

## Memory optimisations (Liger-Kernel + gradient checkpointing)

### Install Liger-Kernel

CVLM training requires `liger-kernel` (fused Triton kernels for the SmolLM
decoder). Install once into the conda env:

```bash
PYTHONNOUSERSITE=1 /home/jovyan/.mlspace/envs/cvlm/bin/pip install "liger-kernel>=0.8.0,<0.9"
```

Verify:

```bash
PYTHONNOUSERSITE=1 /home/jovyan/.mlspace/envs/cvlm/bin/python -c \
  "from liger_kernel.transformers import apply_liger_kernel_to_llama; \
   import importlib.metadata; \
   print('liger', importlib.metadata.version('liger-kernel'))"
```

Expected: `liger 0.8.x`.

`liger-kernel 0.8.x` requires `torch >= 2.5` (uses the public
`torch.distributed.tensor.DTensor` path). The cvlm env was upgraded to
`torch 2.5.1+cu124` / `triton 3.1.0` for compatibility.

### Gradient checkpointing flag

`scripts/run_full_pipeline.sh` exposes a `GRADIENT_CHECKPOINTING` env var
(default `1`, on). Disable with `GRADIENT_CHECKPOINTING=0` for ~25–35%
faster steps at the cost of ~50% more activation memory:

```bash
GRADIENT_CHECKPOINTING=0 OUTPUT_DIR=/tmp/baseline   bash scripts/run_full_pipeline.sh
GRADIENT_CHECKPOINTING=1 OUTPUT_DIR=/tmp/optimised bash scripts/run_full_pipeline.sh
```

Each run prints `[train] peak GPU memory: X.XX GB` at the end and logs
`system/peak_gpu_gb` to trackio for A/B comparison across runs.

### Direct CLI override

When invoking `src/train_cvlm.py` directly (e.g. via `torchrun`), use
`--gradient_checkpointing true` / `--gradient_checkpointing false`.

## Compression-rate curriculum (in-training)

When CVLM training plateaus at a high static compression rate, train under
an in-training curriculum that starts at a low cr and steps up at fixed
milestones inside one run. Optimizer state is preserved across stages.

### Schedule format

`CR_SCHEDULE` env var (or `--cr_schedule` CLI flag) is a comma-separated
list of `cr:end_step` pairs. Each entry says "use this cr until global_step
reaches end_step". The **last entry's `end_step` MUST be 0** (sentinel for
forever).

Example: cr=1 for steps 0-5999, cr=2 for steps 6000-11999, cr=4 for steps
12000-17999, cr=8 from step 18000 onward:

```bash
CR_SCHEDULE="1:6000,2:12000,4:18000,8:0" \
  OUTPUT_DIR=/path/to/run \
  bash scripts/run_full_pipeline.sh
```

Empty (default) = static cr from `COMPRESSION_RATE`.

### What to watch in trackio

`train/compression_rate` is logged every step; overlay it on `train/loss`
to see exactly when each stage transition happens. Each transition triggers
a forced checkpoint save (`model_step_<N>.safetensors`).

### Eval per-stage

Eval each stage's checkpoint with the cr that was active at its end:

```bash
python src/eval_cvlm.py \
  --checkpoint_path "${OUTPUT_DIR}/model_step_6000.safetensors" \
  --compression_rate 1 \
  --mode cvlm \
  --output_json "${OUTPUT_DIR}/eval_stage1_cr1.json"
```

### Memory note

The cr=1 stage is the heaviest (V == max_source_len, longest decoder
sequence in the curriculum). Set `BATCH_SIZE` to fit the cr=1 stage —
later stages will run with the same batch size and lower memory usage.

## Disable LR schedule (constant LR for curriculum runs)

Cosine annealing over the full step budget kills late curriculum stages —
by the time cr=8 starts, the LR is too low to relearn the bottleneck.
Disable cosine decay so the LR stays constant at `--lr` after warmup:

```bash
DISABLE_LR_SCHEDULE=1 \
  CR_SCHEDULE="1:6000,2:12000,4:18000,8:0" \
  OUTPUT_DIR=/path/to/run \
  bash scripts/run_full_pipeline.sh
```

Equivalent direct CLI: `--disable_lr_schedule true`. Warmup (if enabled)
still ramps linearly; only the post-warmup cosine decay is skipped.

## Eval all checkpoints (per-step trackio scatter)

Sweep every `model_step_*.safetensors` in `OUTPUT_DIR` and log each
checkpoint's metrics into a single per-mode trackio run, indexed by
checkpoint step. The dashboard renders the metric chart as a scatter/line
across checkpoints automatically.

```bash
SKIP_TRAIN=1 \
  EVAL_ALL_CHECKPOINTS=1 \
  OUTPUT_DIR=/path/to/existing/run \
  bash scripts/run_full_pipeline.sh
```

Per-checkpoint JSONs are written as `eval_${MODE}_step${STEP}.json`. The
trackio run name is `${TRACKIO_RUN_NAME}_${MODE}` (one run per mode, N
points per mode at the trained checkpoint steps).

For curriculum runs, evaluate each checkpoint at the cr that was active
when it was saved by also setting `COMPRESSION_RATE` per stage — or run
the sweep multiple times with different `COMPRESSION_RATE` values to
inspect cr-vs-checkpoint generalisation.

## Eval each checkpoint at its native (training-time) cr

For curriculum runs, evaluating every checkpoint at the same fixed cr
mismatches what the model actually learned at each stage. Pass
`EVAL_CR_SCHEDULE` (same format as training `CR_SCHEDULE`) and the eval
sweep will set the model's compression rate per checkpoint based on the
schedule:

```bash
SKIP_TRAIN=1 \
  EVAL_ALL_CHECKPOINTS=1 \
  EVAL_CR_SCHEDULE="1:6000,2:12000,4:18000,8:0" \
  OUTPUT_DIR=/path/to/existing/run \
  bash scripts/run_full_pipeline.sh
```

Each checkpoint at step `S` is evaluated at the cr of the schedule entry
whose `end_step >= S` (i.e. the cr that produced its weights):

| ckpt step | trained at | evaluated at |
|---|---|---|
| ≤ 6000 | cr=1 | cr=1 |
| 6001–12000 | cr=2 | cr=2 |
| 12001–18000 | cr=4 | cr=4 |
| > 18000 | cr=8 | cr=8 |

Per-checkpoint JSONs are written as `eval_${MODE}_native_step${STEP}.json`
(the `_native` suffix avoids overwriting prior fixed-cr eval JSONs in the
same directory). A new scalar `compression_rate` is added to each JSON
and logged to trackio as `eval/compression_rate`, so you can overlay it
on metric scatters to see exactly where each cr stage sits.

`EVAL_CR_SCHEDULE` is only forwarded to cr-dependent modes (`cvlm`,
`cvlm_shuffle`, `baseline_proj`); `baseline_llm` and `baseline_llm_full`
ignore the flag because they don't use compression.

Direct CLI:

```bash
python src/eval_cvlm.py \
  --all_checkpoints \
  --cr_schedule "1:6000,2:12000,4:18000,8:0" \
  --checkpoint_path "${OUTPUT_DIR}/model_step_30201.safetensors" \
  --mode cvlm --compute_generation_metrics \
  --output_json "${OUTPUT_DIR}/eval_cvlm.json"
```

Constraints: `--cr_schedule` requires `--all_checkpoints` and one of the
cr-dependent modes; otherwise eval exits with `ValueError`.

## Unfreeze top-K text-encoder layers

By default the text encoder (ModernBERT-base, 22 layers) is fully frozen.
To unfreeze the top-K transformer layers (plus `final_norm`) and let
gradients flow back into them during training, set `UNFREEZE_ENCODER_TOP_K`
or pass `--unfreeze_encoder_top_k`.

```bash
UNFREEZE_ENCODER_TOP_K=4 \
bash CVLM/scripts/run_full_pipeline.sh
```

Direct CLI:

```bash
python src/train_cvlm.py \
  --output_dir "${OUTPUT_DIR}" \
  --unfreeze_encoder_top_k 4 \
  ...other args...
```

Notes:
- `0` (default) = fully frozen, identical to the legacy code path.
- The top-K layers + `final_norm` are saved in each checkpoint
  (`requires_grad`-filtered safetensors export already handles this — no
  separate flag needed at eval time).
- Old fully-frozen checkpoints load via `strict=False` against a
  `unfreeze_encoder_top_k>0` model: missing top-layer weights will be the
  pretrained ModernBERT init (since we always `from_pretrained` the
  encoder before partial-loading the trainable state dict).
- Gradient checkpointing is enabled on the text encoder automatically
  when `unfreeze_encoder_top_k>0` and `--gradient_checkpointing=1`, to
  bound activation memory through the full encoder stack.
- The encoder is switched to `train()` mode while top-K is unfrozen so
  dropout in the unfrozen blocks is active. Frozen lower layers receive
  no parameter gradients regardless of train/eval state.
