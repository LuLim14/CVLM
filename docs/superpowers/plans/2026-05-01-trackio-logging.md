# Trackio Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `torch.utils.tensorboard.SummaryWriter` with [trackio](https://github.com/gradio-app/trackio) across CVLM training, SFT training, and eval, so metrics can be inspected in a HF Spaces / local browser UI without TensorBoard. CSV + PNG dashboards stay as offline fallback.

**Architecture:** Add a thin wrapper `TrackioRun` to `src/train_logging.py` that owns `trackio.init/log/finish` and degrades to a no-op when trackio is unavailable or disabled. Call sites in `train_cvlm.py` (manual loop), `train_sft.py` (HF `TrainerCallback`), and `eval_cvlm.py` (eval scalars + compression histogram) call into the wrapper instead of `SummaryWriter`. CLI flags + env vars on both pipeline scripts let users set the project/run name/space.

**Tech Stack:** Python 3.10+, `trackio` (>=0.0.6), `transformers.Trainer` + `TrainerCallback`, PyTorch DDP. No new files except an optional smoke test.

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/train_logging.py` | Modify | Add `TrackioRun` wrapper + extend `make_logger()` to also return a `TrackioRun`. |
| `src/train_cvlm.py` | Modify | Drop `SummaryWriter` setup + `add_scalar` calls; add `--trackio_*` flags; route scalars through `TrackioRun`. |
| `src/train_sft.py` | Modify | Drop `report_to=["tensorboard"]`; add a `TrackioCallback`; add `--trackio_*` flags. |
| `src/eval_cvlm.py` | Modify | Drop the eval-time `SummaryWriter` block; add a `TrackioRun` per eval mode; add `--trackio_*` flags. |
| `scripts/run_full_pipeline.sh` | Modify | Add `TRACKIO_PROJECT`, `TRACKIO_RUN_NAME`, `TRACKIO_SPACE_ID`, `TRACKIO_DISABLE` envs and forward them to train + eval. |
| `scripts/run_sft_pipeline.sh` | Modify | Same as full pipeline, for SFT train + eval. |
| `tests/test_trackio_logging.py` | Create | Smoke test: `TrackioRun` no-op path + happy path with stub. |
| `USAGE.md` | Modify | Document trackio setup / env vars / disabling. |

---

## Conventions Used Throughout

- **Master rank only.** All trackio calls are guarded by `is_master` (manual loop) or `state.is_world_process_zero` (HF Trainer). Workers skip init entirely.
- **Disable cleanly.** `--trackio_disable` (or env `TRACKIO_DISABLE=1`) returns a no-op `TrackioRun`. Same wrapper used by every call site so trainers don't branch.
- **Run name default.** `os.path.basename(args.output_dir)` for train; `f"eval_{args.mode}"` for eval (matches the existing TB run-name convention).
- **Project default.** `cvlm` (settable via `--trackio_project` or `TRACKIO_PROJECT`).
- **Metric keys are unchanged** from the current TB layout: `train/loss`, `train/loss_avg`, `train/lr`, `train/grad_norm`, `train/batch_time`, `eval/<metric>`, `eval/compression_ratio_dist` (histogram).

---

### Task 1: Add `TrackioRun` wrapper to `src/train_logging.py`

**Files:**
- Modify: `src/train_logging.py` (append new class + extend `make_logger`)
- Test:   `tests/test_trackio_logging.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackio_logging.py`:

```python
# Smoke tests for the TrackioRun wrapper. We do NOT exercise the real trackio
# SDK (it spawns a Gradio server); we monkeypatch it with a recording stub.
from __future__ import annotations

import sys
import types

import pytest

from train_logging import TrackioRun


class _Stub:
    """Minimal trackio API surface, recording every call."""
    def __init__(self):
        self.calls = []
    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
    def log(self, metrics, step=None):
        self.calls.append(("log", dict(metrics), step))
    def finish(self):
        self.calls.append(("finish",))


@pytest.fixture
def stub_trackio(monkeypatch):
    stub = _Stub()
    fake = types.ModuleType("trackio")
    fake.init = stub.init
    fake.log = stub.log
    fake.finish = stub.finish
    monkeypatch.setitem(sys.modules, "trackio", fake)
    return stub


def test_disabled_run_is_noop(stub_trackio):
    run = TrackioRun(project="p", name="r", config={"a": 1}, disable=True)
    run.log({"x": 1.0}, step=1)
    run.log_histogram("h", [1.0, 2.0], step=1)
    run.finish()
    assert stub_trackio.calls == [], "disabled run must not touch the SDK"


def test_enabled_run_logs_scalars_and_histograms(stub_trackio):
    run = TrackioRun(project="p", name="r", config={"a": 1})
    run.log({"loss": 0.5}, step=10)
    run.log_histogram("hist", [0.1, 0.2, 0.3], step=10)
    run.finish()
    kinds = [c[0] for c in stub_trackio.calls]
    assert kinds == ["init", "log", "log", "finish"]
    init_kwargs = stub_trackio.calls[0][1]
    assert init_kwargs["project"] == "p"
    assert init_kwargs["name"] == "r"
    assert init_kwargs["config"] == {"a": 1}


def test_missing_trackio_disables_silently(monkeypatch):
    # Force ImportError by removing the module if cached.
    monkeypatch.setitem(sys.modules, "trackio", None)
    run = TrackioRun(project="p", name="r", config={})
    # Must not raise even when trackio is absent.
    run.log({"x": 1.0}, step=1)
    run.finish()
    assert run.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM
PYTHONPATH=src pytest tests/test_trackio_logging.py -v
```

Expected: `ImportError: cannot import name 'TrackioRun' from 'train_logging'`.

- [ ] **Step 3: Add the `TrackioRun` class to `src/train_logging.py`**

Append below `MetricsPngPlotter`, before `make_logger`:

```python
class TrackioRun:
    """Thin wrapper around the trackio SDK with a no-op fallback.

    The trainer never branches on `enabled`; calling .log/.log_histogram/.finish
    on a disabled run is a no-op. trackio is imported lazily so the rest of the
    pipeline still runs if the package isn't installed.
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[dict] = None,
        space_id: Optional[str] = None,
        disable: bool = False,
    ) -> None:
        self.enabled = False
        self._mod = None
        if disable:
            return
        try:
            import trackio as _tr  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging] trackio not available; logging disabled: {exc}")
            return
        if _tr is None:  # monkeypatched-to-None in tests
            return
        try:
            init_kwargs = {"project": project, "name": name, "config": config or {}}
            if space_id:
                init_kwargs["space_id"] = space_id
            _tr.init(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging] trackio.init failed; logging disabled: {exc}")
            return
        self._mod = _tr
        self.enabled = True

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            self._mod.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging] trackio.log failed (disabling): {exc}")
            self.enabled = False

    def log_histogram(self, key: str, values, step: Optional[int] = None) -> None:
        """Best-effort histogram log. trackio's histogram object varies by
        version; if unavailable we log summary statistics under <key>_mean/std.
        """
        if not self.enabled:
            return
        try:
            import numpy as np
            arr = np.asarray(values, dtype=float).ravel()
            payload: dict = {}
            try:
                Hist = getattr(self._mod, "Histogram", None)
                if Hist is not None:
                    payload[key] = Hist(arr.tolist())
                else:
                    raise AttributeError
            except Exception:
                if arr.size:
                    payload[f"{key}_mean"] = float(arr.mean())
                    payload[f"{key}_std"] = float(arr.std())
                    payload[f"{key}_min"] = float(arr.min())
                    payload[f"{key}_max"] = float(arr.max())
            if payload:
                self._mod.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging] trackio histogram log failed (disabling): {exc}")
            self.enabled = False

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._mod.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging] trackio.finish failed: {exc}")
        self.enabled = False
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_trackio_logging.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/train_logging.py tests/test_trackio_logging.py
git commit -m "feat(logging): add TrackioRun wrapper with no-op fallback"
```

---

### Task 2: Wire trackio into `src/train_cvlm.py` (manual loop)

**Files:**
- Modify: `src/train_cvlm.py` (CLI flags + writer setup + log block + close block)

- [ ] **Step 1: Add CLI flags**

In `parse_args()` after the `--csv_path` argument (currently around line 116-120), add:

```python
    p.add_argument(
        "--trackio_project",
        type=str,
        default=os.environ.get("TRACKIO_PROJECT", "cvlm"),
        help="trackio project name. Default: cvlm or $TRACKIO_PROJECT.",
    )
    p.add_argument(
        "--trackio_run_name",
        type=str,
        default=os.environ.get("TRACKIO_RUN_NAME", ""),
        help="trackio run name. Default: basename(output_dir) or $TRACKIO_RUN_NAME.",
    )
    p.add_argument(
        "--trackio_space_id",
        type=str,
        default=os.environ.get("TRACKIO_SPACE_ID", ""),
        help="Optional HF Space id (user/space) to host the dashboard.",
    )
    p.add_argument(
        "--trackio_disable",
        action="store_true",
        default=os.environ.get("TRACKIO_DISABLE", "0") == "1",
        help="Disable trackio logging (CSV/PNG still run).",
    )
```

- [ ] **Step 2: Replace the `SummaryWriter` block at lines 280-289**

Replace the existing block:

```python
    writer = None
    csv_writer = None
    plotter = None
    if is_master:
        from torch.utils.tensorboard import SummaryWriter
        from train_logging import make_logger
        tb_dir = args.tensorboard_dir.strip() or os.path.join(args.output_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        csv_writer, plotter = make_logger(args.output_dir, args.csv_path, args.plot_interval)
```

with:

```python
    trackio_run = None
    csv_writer = None
    plotter = None
    if is_master:
        from train_logging import TrackioRun, make_logger
        run_name = args.trackio_run_name.strip() or os.path.basename(args.output_dir.rstrip("/"))
        config_payload = {
            "model_name_or_path": model_args.model_name_or_path,
            "text_encoder_name": model_args.text_encoder_name,
            "vision_encoder_name": getattr(model_args, "vision_encoder_name", ""),
            "compression_rate": args.compression_rate,
            "max_vision_len": args.max_vision_len,
            "max_source_len": args.max_source_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "lr": args.lr,
            "warmup_steps": warmup_steps,
            "world_size": world_size,
        }
        trackio_run = TrackioRun(
            project=args.trackio_project,
            name=run_name,
            config=config_payload,
            space_id=args.trackio_space_id or None,
            disable=args.trackio_disable,
        )
        csv_writer, plotter = make_logger(args.output_dir, args.csv_path, args.plot_interval)
```

- [ ] **Step 3: Replace the `add_scalar` calls in the master log block (lines 382-388)**

Replace:

```python
                if writer is not None:
                    writer.add_scalar("train/loss", curr_loss, global_step)
                    writer.add_scalar("train/loss_avg", running_avg_loss_value.avg, global_step)
                    writer.add_scalar("train/lr", curr_lrs[0] if curr_lrs else args.lr, global_step)
                    if grad_norm > 0:
                        writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    writer.add_scalar("train/batch_time", batch_time, global_step)
```

with:

```python
                if trackio_run is not None:
                    metrics = {
                        "train/loss": curr_loss,
                        "train/loss_avg": running_avg_loss_value.avg,
                        "train/lr": curr_lrs[0] if curr_lrs else args.lr,
                        "train/batch_time": batch_time,
                    }
                    if grad_norm > 0:
                        metrics["train/grad_norm"] = grad_norm
                    trackio_run.log(metrics, step=global_step)
```

- [ ] **Step 4: Replace the close block at lines 458-461**

Replace:

```python
    if writer is not None:
        writer.close()
    if csv_writer is not None:
        csv_writer.close()
```

with:

```python
    if trackio_run is not None:
        trackio_run.finish()
    if csv_writer is not None:
        csv_writer.close()
```

- [ ] **Step 5: Smoke-run with trackio disabled (sanity)**

```bash
cd /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM
PYTHONPATH=src PYTHONNOUSERSITE=1 TRACKIO_DISABLE=1 python -c "
import argparse, src.train_cvlm as t
ap = t.__dict__  # ensure module imports cleanly
print('train_cvlm imports OK')
"
```

Expected: prints `train_cvlm imports OK` with no traceback.

- [ ] **Step 6: Commit**

```bash
git add src/train_cvlm.py
git commit -m "feat(train_cvlm): replace TensorBoard SummaryWriter with trackio"
```

---

### Task 3: Wire trackio into `src/train_sft.py` (HF Trainer callback)

**Files:**
- Modify: `src/train_sft.py` (CLI flags + new `TrackioCallback` + Trainer wiring)

- [ ] **Step 1: Add CLI flags**

In `parse_args()` after the `--tensorboard_dir` argument (around line 256-258), add:

```python
    p.add_argument("--trackio_project", type=str,
                   default=os.environ.get("TRACKIO_PROJECT", "cvlm"))
    p.add_argument("--trackio_run_name", type=str,
                   default=os.environ.get("TRACKIO_RUN_NAME", ""))
    p.add_argument("--trackio_space_id", type=str,
                   default=os.environ.get("TRACKIO_SPACE_ID", ""))
    p.add_argument("--trackio_disable", action="store_true",
                   default=os.environ.get("TRACKIO_DISABLE", "0") == "1")
```

- [ ] **Step 2: Add `TrackioCallback` below `PngLoggingCallback`**

Append after the `PngLoggingCallback` class (around line 224):

```python
class TrackioCallback(TrainerCallback):
    """Forwards `Trainer.log` metrics to a `TrackioRun` on master rank only."""

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        space_id: Optional[str] = None,
        disable: bool = False,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self.space_id = space_id or None
        self.disable = disable
        self._run = None  # constructed on_train_begin (master only)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return
        from train_logging import TrackioRun
        self._run = TrackioRun(
            project=self.project,
            name=self.name,
            config=self.config,
            space_id=self.space_id,
            disable=self.disable,
        )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):  # type: ignore[override]
        if self._run is None or logs is None:
            return
        # Skip eval-only logs (no "loss" key) — same filter as PngLoggingCallback.
        if "loss" not in logs:
            return
        metrics: Dict[str, float] = {}
        for src_key, dst_key in (
            ("loss", "train/loss"),
            ("learning_rate", "train/lr"),
            ("grad_norm", "train/grad_norm"),
            ("epoch", "train/epoch"),
        ):
            if src_key in logs and isinstance(logs[src_key], (int, float)):
                metrics[dst_key] = float(logs[src_key])
        if metrics:
            self._run.log(metrics, step=int(state.global_step))

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if self._run is not None:
            self._run.finish()
            self._run = None
```

- [ ] **Step 3: Drop tensorboard from `report_to` and add the callback**

In `main()`, change `TrainingArguments(...)` line `report_to=["tensorboard"]` to `report_to=[]`.

Then change the `Trainer(...)` callbacks list (currently `callbacks=[PngLoggingCallback(...)]`) to:

```python
    sft_run_name = args.trackio_run_name.strip() or os.path.basename(args.output_dir.rstrip("/"))
    sft_config = {
        "model_name_or_path": args.model_name_or_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "lr": args.lr,
        "max_prompt_len": args.max_prompt_len,
        "max_answer_len": args.max_answer_len,
        "task": "sft_baseline",
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[
            PngLoggingCallback(args.output_dir, csv_path, args.plot_interval),
            TrackioCallback(
                project=args.trackio_project,
                name=sft_run_name,
                config=sft_config,
                space_id=args.trackio_space_id,
                disable=args.trackio_disable,
            ),
        ],
    )
```

- [ ] **Step 4: Smoke-test (import only — no GPU/data needed)**

```bash
cd /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM
PYTHONPATH=src PYTHONNOUSERSITE=1 TRACKIO_DISABLE=1 python -c "
import importlib; m = importlib.import_module('train_sft')
assert hasattr(m, 'TrackioCallback'); print('train_sft imports OK')
"
```

Expected: `train_sft imports OK`.

- [ ] **Step 5: Commit**

```bash
git add src/train_sft.py
git commit -m "feat(train_sft): replace TB report_to with TrackioCallback"
```

---

### Task 4: Wire trackio into `src/eval_cvlm.py`

**Files:**
- Modify: `src/eval_cvlm.py` (CLI flags + replace SummaryWriter block)

- [ ] **Step 1: Add CLI flags near `--tensorboard_dir` (line 64)**

Append immediately after the existing `--tensorboard_dir` argument:

```python
    p.add_argument("--trackio_project", type=str,
                   default=os.environ.get("TRACKIO_PROJECT", "cvlm"))
    p.add_argument("--trackio_run_name", type=str,
                   default=os.environ.get("TRACKIO_RUN_NAME", ""))
    p.add_argument("--trackio_space_id", type=str,
                   default=os.environ.get("TRACKIO_SPACE_ID", ""))
    p.add_argument("--trackio_disable", action="store_true",
                   default=os.environ.get("TRACKIO_DISABLE", "0") == "1")
```

- [ ] **Step 2: Replace the eval-time `SummaryWriter` block (lines 966-993)**

Replace the entire `if args.tensorboard_dir:` block with:

```python
    if not args.trackio_disable:
        from train_logging import TrackioRun
        run_name = (
            args.trackio_run_name.strip()
            or args.tb_run_name
            or f"eval_{args.mode}"
        )
        eval_config = {
            "mode": args.mode,
            "model_name_or_path": args.model_name_or_path,
            "text_encoder_name": args.text_encoder_name,
            "compression_rate": args.compression_rate,
            "max_prompt_len": args.max_prompt_len,
            "max_answer_len": args.max_answer_len,
            "max_vision_len": args.max_vision_len,
            "max_source_len": args.max_source_len,
            "checkpoint_path": getattr(args, "checkpoint_path", "") or getattr(args, "sft_model_path", ""),
            "global_step": int(args.global_step),
        }
        run = TrackioRun(
            project=args.trackio_project,
            name=run_name,
            config=eval_config,
            space_id=args.trackio_space_id or None,
        )
        step = int(args.global_step)

        flat_metrics: dict = {}
        for k, v in results.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat_metrics[f"eval/{k}"] = float(v)
        if flat_metrics:
            run.log(flat_metrics, step=step)

        per_sample_ratios = []
        limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
        cr = max(int(args.compression_rate), 1)
        enc_tok = dataset._enc_tok
        dec_tok_hist = sft_tokenizer if args.mode == "sft" else model.tokenizer  # type: ignore[union-attr]
        for idx in range(limit):
            row_id = dataset._row_indices[idx]
            text = dataset._hf[row_id]["input"]
            s_len = len(dec_tok_hist(text, add_special_tokens=False, truncation=False)["input_ids"])
            l_enc = min(len(enc_tok(text, add_special_tokens=False, truncation=False)["input_ids"]),
                        dataset.max_source_len)
            v_len = max(min((l_enc + cr - 1) // cr, args.max_vision_len), 1)
            per_sample_ratios.append(s_len / v_len)
        if per_sample_ratios:
            run.log_histogram("eval/compression_ratio_dist", per_sample_ratios, step=step)
        run.finish()
        print(f"\nEval metrics logged to trackio (project={args.trackio_project} run={run_name} step={step})")
```

Note: `--tensorboard_dir` and `--tb_run_name` flags **stay in the CLI** (they're now silently ignored at log time but still accepted so the pipeline scripts don't have to change their --flag list yet). They will be removed only in Task 5 once both pipelines no longer pass them.

- [ ] **Step 3: Smoke-test imports**

```bash
PYTHONPATH=src PYTHONNOUSERSITE=1 TRACKIO_DISABLE=1 python -c "
import importlib; importlib.import_module('eval_cvlm'); print('eval_cvlm imports OK')
"
```

Expected: `eval_cvlm imports OK`.

- [ ] **Step 4: Commit**

```bash
git add src/eval_cvlm.py
git commit -m "feat(eval): replace TB SummaryWriter with trackio per-mode run"
```

---

### Task 5: Update both pipeline scripts

**Files:**
- Modify: `scripts/run_full_pipeline.sh`
- Modify: `scripts/run_sft_pipeline.sh`

- [ ] **Step 1: Add trackio block to `run_full_pipeline.sh`**

After the `# Eval` section block (around line 65 — right before `TB_DIR=...`), insert:

```bash
# trackio (replaces TensorBoard for the live UI)
TRACKIO_PROJECT="${TRACKIO_PROJECT:-cvlm}"
TRACKIO_RUN_NAME="${TRACKIO_RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-}"
TRACKIO_DISABLE="${TRACKIO_DISABLE:-0}"
export TRACKIO_PROJECT TRACKIO_RUN_NAME TRACKIO_SPACE_ID TRACKIO_DISABLE
```

In the banner echo block (`echo "  EVAL_MODES        = ..."` etc.) add:

```bash
echo "  TRACKIO_PROJECT   = ${TRACKIO_PROJECT}"
echo "  TRACKIO_RUN_NAME  = ${TRACKIO_RUN_NAME}"
echo "  TRACKIO_SPACE_ID  = ${TRACKIO_SPACE_ID:-<local-only>}"
echo "  TRACKIO_DISABLE   = ${TRACKIO_DISABLE}"
```

Replace the `TensorBoard:` line of the banner with:

```bash
echo "trackio UI:   trackio show --project ${TRACKIO_PROJECT}"
echo "(or set TRACKIO_SPACE_ID=user/space to host on HF Spaces)"
```

The existing TB-aware lines (`TB_DIR=...`, `--tensorboard_dir "${TB_DIR}/train"`, the eval `--tensorboard_dir "${TB_DIR}"`) **stay as-is** — `train_cvlm.py` / `eval_cvlm.py` still accept the flag but ignore it post-Task 4. Removing it would be a breaking change for users with running scripts; defer.

- [ ] **Step 2: Same edits in `run_sft_pipeline.sh`**

Mirror the trackio block (project/run/space/disable + banner echo). Run name default is `$(basename "${OUTPUT_DIR}")` which already encodes `sft_run_<timestamp>`.

- [ ] **Step 3: Run a 10-sample dry-run end to end with trackio disabled**

```bash
cd /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM
TRACKIO_DISABLE=1 MAX_SAMPLES=10 EPOCHS=1 BATCH_SIZE=2 \
EVAL_MAX_SAMPLES=4 EVAL_MODES="cvlm" \
OUTPUT_DIR=/tmp/cvlm_smoke bash scripts/run_full_pipeline.sh
```

Expected: pipeline completes; `/tmp/cvlm_smoke/eval_cvlm.json` exists; no trackio stack traces. Watch the banner for `TRACKIO_DISABLE = 1`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_full_pipeline.sh scripts/run_sft_pipeline.sh
git commit -m "chore(pipelines): expose TRACKIO_* env vars in both pipelines"
```

---

### Task 6: Document in `USAGE.md`

**Files:**
- Modify: `USAGE.md` (append a new top-level section)

- [ ] **Step 1: Append the trackio section**

Append to `USAGE.md`:

```markdown
## Trackio logging

`SummaryWriter` has been replaced by [trackio](https://github.com/gradio-app/trackio)
for the live dashboard. CSV (`metrics.csv`) and PNG (`dashboard.png`) outputs are
unchanged.

### Install

```bash
pip install trackio
```

### Run a training with trackio

```bash
TRACKIO_PROJECT=cvlm \
TRACKIO_RUN_NAME=my-run \
OUTPUT_DIR=/path/to/run \
bash scripts/run_full_pipeline.sh
```

### View the dashboard

Local UI:

```bash
trackio show --project cvlm
```

Host on HF Spaces (auto-creates the Space the first time):

```bash
TRACKIO_SPACE_ID=username/cvlm-runs bash scripts/run_full_pipeline.sh
```

### Disable trackio

```bash
TRACKIO_DISABLE=1 bash scripts/run_full_pipeline.sh
# or per-binary:
python src/train_cvlm.py --trackio_disable ...
```

### Metrics keys

| Phase | Keys |
| --- | --- |
| Train (CVLM + SFT) | `train/loss`, `train/loss_avg`, `train/lr`, `train/grad_norm`, `train/batch_time` |
| Eval | `eval/<metric>` (loss, ppl, ROUGE, BLEU-4, EM, compression stats) + `eval/compression_ratio_dist` (histogram if supported, else `*_mean/std/min/max`) |
```

- [ ] **Step 2: Commit**

```bash
git add USAGE.md
git commit -m "docs: usage notes for trackio logging"
```

---

## Self-Review Checklist (run by implementer when finished)

1. **Spec coverage** — every TB call site identified earlier (train_cvlm:283-289, 382-388, 458-459; train_sft:310; eval_cvlm:966-993) is replaced.
2. **No placeholders** — all code blocks compile (no `...`, no "TODO").
3. **Type/name consistency** — `TrackioRun(project, name, config, space_id, disable)` matches in all four call sites and the test. `log(metrics, step=...)`, `log_histogram(key, values, step=...)`, `finish()` — single signature.
4. **DDP safety** — `is_master` (manual) and `state.is_world_process_zero` (Trainer) guard every trackio touchpoint.
5. **Failure modes** — missing `trackio` package, `init` failure, mid-run `log` failure all degrade to silent no-op (and log a single warning), so training never crashes because of telemetry.
