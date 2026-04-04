#!/usr/bin/env bash
# Full CVLM pipeline: preprocess embeddings -> train -> eval.
# All three stages log into the same TensorBoard tree at "${OUTPUT_DIR}/tb",
# so a single `tensorboard --logdir ${OUTPUT_DIR}/tb` shows train + eval.
#
# Designed to be safe under terminal disconnect: stdout/stderr are tee'd into
# "${OUTPUT_DIR}/pipeline.log" and the script exits non-zero on any failure.
#
# Typical use:
#   tmux new -s cvlm
#   OUTPUT_DIR=/path/to/run bash CVLM/scripts/run_full_pipeline.sh
#   <Ctrl-b d>   # detach — training keeps running
#   tmux attach -t cvlm   # reattach later

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# -----------------------------------------------------------------------------
# Configuration (override by exporting env vars before running the script).
# -----------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/cvlm_$(date +%Y%m%d_%H%M%S)}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-${ROOT}/data/processed/embeddings.npz}"
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"

# Preprocess
EMBEDDER_MODEL="${EMBEDDER_MODEL:-answerdotai/ModernBERT-base}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-2048}"
EMBED_MAX_SAMPLES="${EMBED_MAX_SAMPLES:-0}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
COMPRESSION_RATE="${COMPRESSION_RATE:-4}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"   # set to 1 to reuse an existing .npz

# Train
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM-135M-Instruct}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_ANSWER_LEN="${MAX_ANSWER_LEN:-256}"
MAX_VISION_LEN="${MAX_VISION_LEN:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-0}"
NPROC="${NPROC:-1}"

# Eval
EVAL_SPLIT="${EVAL_SPLIT:-train}"     # use 'test' once you have one
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_MODES="${EVAL_MODES:-cvlm baseline_llm}"   # space-separated

TB_DIR="${OUTPUT_DIR}/tb"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

mkdir -p "${OUTPUT_DIR}" "${TB_DIR}" "$(dirname "${EMBEDDINGS_PATH}")"

# Tee everything to a log file so you can `tail -f` it from another shell.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "CVLM FULL PIPELINE"
echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "  EMBEDDINGS_PATH  = ${EMBEDDINGS_PATH}"
echo "  DATASET_NAME     = ${DATASET_NAME}"
echo "  MODEL_NAME       = ${MODEL_NAME}"
echo "  MAX_VISION_LEN   = ${MAX_VISION_LEN}"
echo "  COMPRESSION_RATE = ${COMPRESSION_RATE}"
echo "  TB_DIR           = ${TB_DIR}"
echo "  LOG_FILE         = ${LOG_FILE}"
echo "======================================================================"
echo "TensorBoard:  tensorboard --logdir ${TB_DIR} --port 6006 --bind_all"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Step 1/3: Preprocess embeddings
# -----------------------------------------------------------------------------
if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  echo; echo "===== Step 1/3: Build embeddings -> ${EMBEDDINGS_PATH} ====="
  python "${ROOT}/build_dataset/embed_dataset.py" \
    --embeddings_path "${EMBEDDINGS_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --embedder_model "${EMBEDDER_MODEL}" \
    --max_length "${EMBED_MAX_LENGTH}" \
    --max_samples "${EMBED_MAX_SAMPLES}" \
    --batch_size "${EMBED_BATCH_SIZE}" \
    --compression_rate "${COMPRESSION_RATE}"
else
  echo; echo "===== Step 1/3: SKIPPED (SKIP_PREPROCESS=1), using ${EMBEDDINGS_PATH} ====="
fi

# -----------------------------------------------------------------------------
# Step 2/3: Train
# -----------------------------------------------------------------------------
echo; echo "===== Step 2/3: Train CVLM -> ${OUTPUT_DIR} ====="
TRAIN_ARGS=(
  "${ROOT}/src/train_cvlm.py"
  --output_dir "${OUTPUT_DIR}"
  --embeddings_path "${EMBEDDINGS_PATH}"
  --dataset_name "${DATASET_NAME}"
  --model_name_or_path "${MODEL_NAME}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --max_prompt_len "${MAX_PROMPT_LEN}"
  --max_answer_len "${MAX_ANSWER_LEN}"
  --max_vision_len "${MAX_VISION_LEN}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --log_interval "${LOG_INTERVAL}"
  --save_interval_steps "${SAVE_INTERVAL_STEPS}"
  --tensorboard_dir "${TB_DIR}/train"
)
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" "${TRAIN_ARGS[@]}"
else
  python "${TRAIN_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# Step 3/3: Locate latest checkpoint and evaluate
# -----------------------------------------------------------------------------
CKPT=$(ls -1 "${OUTPUT_DIR}"/model_step_*.safetensors 2>/dev/null \
       | sed -E 's|.*model_step_([0-9]+)\.safetensors$|\1 &|' \
       | sort -n \
       | tail -n1 \
       | awk '{print $2}')
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no model_step_*.safetensors found in ${OUTPUT_DIR}"; exit 1
fi
# Extract step number so eval can tag TB scalars with it.
STEP=$(basename "${CKPT}" | sed -E 's|model_step_([0-9]+)\.safetensors|\1|')
echo; echo "===== Step 3/3: Eval checkpoint ${CKPT} (step=${STEP}) ====="

for MODE in ${EVAL_MODES}; do
  echo; echo "----- eval mode=${MODE} -----"
  python "${ROOT}/src/eval_cvlm.py" \
    --checkpoint_path "${CKPT}" \
    --embeddings_path "${EMBEDDINGS_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_split "${EVAL_SPLIT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --mode "${MODE}" \
    --max_samples "${EVAL_MAX_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --max_answer_len "${MAX_ANSWER_LEN}" \
    --max_vision_len "${MAX_VISION_LEN}" \
    --tensorboard_dir "${TB_DIR}" \
    --tb_run_name "eval_${MODE}" \
    --global_step "${STEP}" \
    --output_json "${OUTPUT_DIR}/eval_${MODE}.json"
done

echo
echo "======================================================================"
echo "DONE. Artifacts:"
echo "  checkpoints : ${OUTPUT_DIR}/model_step_*.safetensors"
echo "  eval JSON   : ${OUTPUT_DIR}/eval_*.json"
echo "  full log    : ${LOG_FILE}"
echo "  tensorboard : tensorboard --logdir ${TB_DIR} --port 6006 --bind_all"
echo "======================================================================"
