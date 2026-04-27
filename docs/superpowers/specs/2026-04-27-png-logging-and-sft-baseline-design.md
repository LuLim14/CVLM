# PNG training plots + SFT baseline pipeline — design

Date: 2026-04-27
Project: CVLM (text compression via visual encoders)

## Problem statement

Two independent gaps in the current pipeline:

1. **Live training visibility without TensorBoard.** The trainer (`src/train_cvlm.py`) already writes TensorBoard event files to `<output_dir>/tb`, but the user cannot open the TensorBoard UI in their environment. They need to monitor loss / grad norm / lr while training is running so they can decide whether to abort.
2. **No SFT baseline for comparison.** The CVLM model trains a frozen decoder with a vision-token compression path. To claim "compression preserves accuracy" we need a comparison point: the *same* decoder, fully fine-tuned on the *same* data with standard SFT (full uncompressed source as text), evaluated under the *same* metrics.

## Goals

- Generate PNG plots (loss, lr, grad norm, batch time, plus a combined dashboard) on disk during training, refreshed every N steps. Plots must be viewable via `scp` / file browser without TensorBoard running.
- Provide an SFT training entrypoint that produces a checkpoint usable by `eval_cvlm.py`, with eval output JSON shape-compatible with CVLM's JSON.
- Reuse the PNG plotting between CVLM training and SFT training (single source of truth, same dashboard layout).

## Non-goals

- Replacing TensorBoard logging. TB stays. PNGs are additive.
- LoRA / PEFT for the SFT baseline. We want a full fine-tune as the "ceiling" comparison.
- Modifying `modeling.py` or `cvlm_dataset.py`.
- Changing the CVLM checkpoint format or its eval JSON schema.

## Architecture overview

```
src/
  train_logging.py       NEW   MetricsCsvWriter, MetricsPngPlotter
  train_cvlm.py          EDIT  wire CSV+PNG calls into existing log block
  train_sft.py           NEW   TRL SFTTrainer entrypoint + plotting callback
  eval_cvlm.py           EDIT  add --mode sft (loads bare HF decoder)
scripts/
  run_sft_pipeline.sh    NEW   train + eval wrapper, mirrors run_full_pipeline.sh
docs/superpowers/specs/
  2026-04-27-png-logging-and-sft-baseline-design.md  (this file)
```

## Component 1 — `src/train_logging.py`

New module. Two classes, master-rank-only usage enforced by the caller.

### `MetricsCsvWriter`

- `__init__(self, csv_path: str)`: opens line-buffered append-mode file. Writes header `step,loss,loss_avg,lr,grad_norm,batch_time` if file is empty.
- `append(self, step, loss, loss_avg, lr, grad_norm, batch_time)`: one row, fsync-free, line-buffered (so a `tail -f` works).
- `close()`.

CSV is the source of truth. Append-only. Survives partial-run crashes — the next plot run still produces a usable graph.

### `MetricsPngPlotter`

- `__init__(self, csv_path: str, out_dir: str)`: stores paths. Uses matplotlib `Agg` backend (set on import).
- `refresh(self)`: re-reads the CSV via `numpy.genfromtxt` (skip header), produces:
  - `loss.png` — `loss` and `loss_avg` vs step (two lines).
  - `lr.png` — `lr` vs step.
  - `grad_norm.png` — `grad_norm` vs step (skip rows where `grad_norm == 0`).
  - `batch_time.png` — `batch_time` vs step.
  - `dashboard.png` — 2x2 grid of the above.
  Each PNG is overwritten atomically (write to `*.tmp` then `os.replace`).
- Failure mode: any matplotlib error logs once and is suppressed for subsequent calls (don't kill training).

Design note: plotting reads the CSV every refresh rather than keeping in-memory state. This means the same `MetricsPngPlotter` works for the CVLM trainer (called inside the loop) and for the SFT TRL callback (no shared state with the trainer's running averages).

## Component 2 — `train_cvlm.py` edits

Single, narrow change. The existing logging block (`train_cvlm.py:367-373`) already runs only on master and only every `--log_interval` steps. We extend it.

### CLI additions

- `--plot_interval` (int, default 100): refresh PNGs every N optimizer steps. `0` disables PNG plotting.
- `--csv_path` (str, default `<output_dir>/metrics.csv`): override CSV location.

### Wiring

Near the `SummaryWriter` setup (`train_cvlm.py:268-274`), if master:

```
csv_writer = MetricsCsvWriter(csv_path)
plotter = MetricsPngPlotter(csv_path, out_dir=output_dir) if args.plot_interval > 0 else None
```

In the log block (currently inside `if is_master and global_step % args.log_interval == 0:`), after the existing TB `add_scalar` calls, also call:

```
csv_writer.append(global_step, curr_loss, running_avg_loss_value.avg,
                  curr_lrs[0] if curr_lrs else args.lr,
                  grad_norm if grad_norm > 0 else 0.0,
                  batch_time)
```

Add a separate block (still master-only) that triggers PNG refresh:

```
if plotter is not None and global_step % args.plot_interval == 0:
    plotter.refresh()
```

At the end (next to `writer.close()`):

```
csv_writer.close()
if plotter is not None:
    plotter.refresh()
```

## Component 3 — `src/train_sft.py`

New entrypoint, ~150 LOC. Uses TRL `SFTTrainer` (which wraps HF `Trainer`).

### CLI

Mirrors `train_cvlm.py` where possible (same flag names) for muscle memory:

- `--output_dir` (required)
- `--dataset_name` (default `sggetao/PwC`)
- `--dataset_split` (default `train`)
- `--model_name_or_path` (default `HuggingFaceTB/SmolLM-135M-Instruct`)
- `--max_prompt_len` (default 512)
- `--max_answer_len` (default 2048)
- `--max_samples` (default 0 = all)
- `--epochs` (default 1)
- `--batch_size` (default 2)
- `--gradient_accumulation_steps` (default 1)
- `--lr` (default 1e-5)
- `--grad_clip` (default 1.0)
- `--enable_warmup`, `--warmup_ratio`, `--warmup_steps`
- `--log_interval` (default 10), `--save_interval_steps` (default 500)
- `--no_bf16`
- `--seed` (default 42)
- `--plot_interval` (default 100), `--csv_path` (default `<output_dir>/metrics.csv`)

### Data preparation

Inline (no shared collator with CVLM). Per record `{"input": doc, "instruction": question, "output": answer}`:

1. Tokenize `input` with the *decoder* tokenizer; tokenize `instruction` with same; tokenize `output` with same.
2. Filter rows to match CVLM's training-data acceptance set sample-for-sample:
   - Drop rows where `len(decoder_tokenizer(question_chat_templated))` > `max_prompt_len`. (Question-only — same as CVLM's filter; the document length is unbounded by CVLM's filter, so SFT also accepts arbitrarily long documents.)
   - Drop rows where `len(decoder_tokenizer(answer))` > `max_answer_len`.
3. Format strings:
   - If decoder tokenizer has a chat template: build messages `[{role:user, content: doc + "\n\n" + question}, {role:assistant, content: answer}]` and apply the template.
   - Else: insert an explicit unique sentinel `<|sft_response|>` between question and answer: `f"{doc}\n\n{question}\n{tokenizer.bos_token or ''}<|sft_response|>{answer}{tokenizer.eos_token}"`. The sentinel is needed for the response-only collator (see step 4).
4. For loss masking: use TRL's `DataCollatorForCompletionOnlyLM`.
   - With chat template: `response_template` is the assistant turn's start token sequence (e.g. `<|im_start|>assistant\n` for ChatML-style; resolved at runtime from the template's rendered output).
   - Without chat template: `response_template` is the literal `<|sft_response|>` sentinel inserted in step 3. This guarantees the marker is unique within the rendered string (a `\n\n` sentinel would collide with paragraph breaks in the document).
   The collator masks every label position before the response template to `-100`, so loss is computed only on answer tokens (and any post-answer EOS).

### Training loop

- `TrainingArguments(...)` from transformers, configured with:
  - `output_dir`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `lr_scheduler_type="cosine"`, `warmup_steps` / `warmup_ratio`, `bf16`, `max_grad_norm`, `save_steps`, `logging_steps`, `report_to=["tensorboard"]`, `logging_dir=<output_dir>/tb`, `ddp_find_unused_parameters=False`, `gradient_checkpointing=False`, `remove_unused_columns=False`.
- `SFTTrainer(model, args=…, train_dataset=…, data_collator=…, processing_class=tokenizer)`.
- Add a custom `TrainerCallback` (`PngLoggingCallback`):
  - `on_log`: pulls `logs["loss"]`, `logs["learning_rate"]`, `logs.get("grad_norm", 0.0)` and the wall-time delta, calls `MetricsCsvWriter.append(...)`. Maintains its own running average for `loss_avg`.
  - `on_step_end`: every `--plot_interval` steps, calls `plotter.refresh()`. Master-rank-only via `state.is_world_process_zero`.
  - `on_train_end`: final flush + plot.
- DDP/optimizer/scheduler are all handled by `Trainer` — no manual setup.

### Output

- HF-format checkpoint dir written by `trainer.save_model(output_dir)`. Standard layout: `pytorch_model.bin` (or sharded), `config.json`, `tokenizer*`. This is what `--mode sft` will load.

## Component 4 — `eval_cvlm.py` edits

Add `sft` to `--mode` choices. New CLI flag: `--sft_model_path` (HF directory).

### Behavior

When `--mode sft`:

- **Skip building `CVLM(...)`.** Instead load `AutoModelForCausalLM.from_pretrained(args.sft_model_path)` and `AutoTokenizer.from_pretrained(args.sft_model_path)` directly. Move to `device`, eval mode, bf16 if enabled.
- Reuse the *existing* `CvlmTrainDataset` to get `full_prompt_ids` / `full_prompt_mask` / `answer_ids` / `answer_labels` / `answer_mask` (already what `baseline_llm_full` consumes).
- Run the same logic body as `eval_teacher_forcing_baseline_llm_full` but against the bare HF model. Factor that body into a small helper `_run_full_prompt_eval(model_or_decoder, ...)` that both `baseline_llm_full` and `sft` call (decoder is `model.decoder` in the former, the bare HF model in the latter).
- Same for generation: factor a `_run_full_prompt_generate(...)`.
- Output JSON has identical keys to `baseline_llm_full`. `mode: "sft"` is the only structural diff.

This produces a directly diffable JSON pair: CVLM run vs SFT run, same metric set.

### Why not reuse `baseline_llm_full` with `--checkpoint_path`

`baseline_llm_full` builds the full `CVLM(...)` wrapper (ViT, ModernBERT, projectors) just to use `model.decoder`. For an SFT-trained decoder we'd have to rename state-dict keys with a `decoder.` prefix and waste GPU memory on unused submodules. A dedicated mode is cleaner.

## Component 5 — `scripts/run_sft_pipeline.sh`

New shell wrapper, mirrors the structure of `scripts/run_full_pipeline.sh`. Two stages:

1. `torchrun --nproc_per_node=$N src/train_sft.py …`
2. `python src/eval_cvlm.py --mode sft --sft_model_path $OUT_DIR …`

Same env-var conventions as the existing script (`OUTPUT_DIR`, `DATASET_NAME`, etc.).

## Data flow

### CVLM training (existing + plotting)

```
loader → model.forward → loss
       ↓
  optimizer.step
       ↓
  log_block (every log_interval):
     SummaryWriter.add_scalar(...)
     MetricsCsvWriter.append(...)
       ↓
  plot_block (every plot_interval):
     MetricsPngPlotter.refresh() → reads CSV → writes 5 PNGs
```

### SFT training

```
HF dataset → tokenize → DataCollatorForCompletionOnlyLM (mask prompt+pad to -100)
                                                          ↓
                                               TRL SFTTrainer
                                                          ↓
                                          PngLoggingCallback.on_log → CSV
                                          PngLoggingCallback.on_step_end → PNG
                                                          ↓
                                          trainer.save_model → HF dir
```

### SFT eval

```
sggetao/PwC test split → CvlmTrainDataset (existing) → DataLoader
                                                          ↓
                                  AutoModelForCausalLM (SFT weights)
                                                          ↓
                              _run_full_prompt_eval (shared helper)
                                                          ↓
                                  same JSON schema as baseline_llm_full
```

## Error handling

- **Missing matplotlib:** `MetricsPngPlotter.__init__` catches the import error, sets a `_disabled` flag, logs once, every `refresh()` is a no-op. CSV continues to be written.
- **Corrupt CSV row mid-write:** `np.genfromtxt(..., invalid_raise=False)` skips bad rows. Plot still renders.
- **Missing TRL:** `train_sft.py` imports trl at the top; if it's missing, fail fast with a clear message — this is a hard requirement, not optional.
- **Eval `--mode sft` without `--sft_model_path`:** argparse `error()` immediately.
- **DDP rank coordination in SFT plot callback:** all CSV/PNG writes guarded by `state.is_world_process_zero`. No barriers needed (writes are local to rank 0).

## Testing

- **`MetricsCsvWriter`:** unit-test that `append` produces a parseable CSV with the right header and N rows after N appends.
- **`MetricsPngPlotter`:** unit-test that `refresh()` on a synthetic 50-row CSV produces 5 PNG files of non-zero size in the target dir.
- **`train_cvlm.py` integration:** smoke test — run with `--max_samples 8 --epochs 1 --log_interval 1 --plot_interval 2`, assert PNGs exist and CSV has rows.
- **`train_sft.py` integration:** smoke test — same shape: `--max_samples 8 --epochs 1`, assert HF dir is produced and is loadable by `AutoModelForCausalLM`.
- **Eval `--mode sft`:** smoke test — run on the smoke-test SFT dir, assert JSON has the expected metric keys and matches the schema of `baseline_llm_full` JSON.

## Open questions / risks

- **TRL version compatibility:** `SFTTrainer`'s API changed across versions (especially around `tokenizer` vs `processing_class`, and `dataset_text_field` vs explicit collator). The plan assumes a recent TRL (≥0.10). The implementation should pin or at least probe the version and pick the right API surface.
- **Chat template existence:** `HuggingFaceTB/SmolLM-135M-Instruct` does have a chat template. If the user later swaps to a non-instruct base, the plain-text fallback path is used. Tested on the default config.
- **Document length truncation in SFT:** unlike CVLM (which truncates the source to fit `max_source_len` in encoder tokens), SFT consumes the document via the decoder tokenizer, and `Trainer` will simply truncate/skip overlength sequences. This is a known asymmetry — `eval_cvlm.py --mode baseline_llm_full` already has the same property. The comparison is still fair *within* `baseline_llm_full` semantics.

## Acceptance criteria

- Training a CVLM run for ≥200 steps with `--plot_interval 100` produces `metrics.csv` and `dashboard.png` (plus the four individual PNGs) in `<output_dir>`. The PNG content shows live-updating loss/lr/grad-norm curves.
- `bash scripts/run_sft_pipeline.sh` (with appropriate env vars) trains an SFT decoder and runs an eval, producing a JSON with `mode: "sft"` and the same metric keys as a `mode: "baseline_llm_full"` run.
- A diff of CVLM eval JSON vs SFT eval JSON is straightforward (same keys, comparable values).
- No regression in existing CVLM training: passing `--plot_interval 0` reproduces the prior trainer behavior bit-for-bit aside from the new CSV file (CSV writes are unconditional; PNG generation is gated).
