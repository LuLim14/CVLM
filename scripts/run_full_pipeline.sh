#!/usr/bin/env bash
# Full CVLM pipeline: train -> eval.
#
# On-the-fly text-encoder variant: ModernBERT lives inside the CVLM model and
# runs during the forward pass, so there is no preprocessing step. Both stages
# log into the same TensorBoard tree at "${OUTPUT_DIR}/tb", so a single
# `tensorboard --logdir ${OUTPUT_DIR}/tb` shows train + eval.
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
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm/run_$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"

# Model
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM-135M-Instruct}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-answerdotai/ModernBERT-base}"

# Train
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_ANSWER_LEN="${MAX_ANSWER_LEN:-1024}"
MAX_VISION_LEN="${MAX_VISION_LEN:-256}"
MAX_SOURCE_LEN="${MAX_SOURCE_LEN:-0}"      # 0 = compression_rate * max_vision_len
COMPRESSION_RATE="${COMPRESSION_RATE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"            # cap HF dataset rows; 0 = all
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-0}"
NPROC="${NPROC:-1}"

# Eval
EVAL_SPLIT="${EVAL_SPLIT:-test}"          # use 'test' once you have one
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_MODES="${EVAL_MODES:-cvlm baseline_llm}"   # space-separated

TB_DIR="${OUTPUT_DIR}/tb"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

mkdir -p "${OUTPUT_DIR}" "${TB_DIR}"

# Tee everything to a log file so you can `tail -f` it from another shell.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "CVLM FULL PIPELINE (on-the-fly encoder)"
echo "  OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "  DATASET_NAME      = ${DATASET_NAME}"
echo "  MODEL_NAME        = ${MODEL_NAME}"
echo "  TEXT_ENCODER_NAME = ${TEXT_ENCODER_NAME}"
echo "  MAX_VISION_LEN    = ${MAX_VISION_LEN}"
echo "  MAX_SOURCE_LEN    = ${MAX_SOURCE_LEN} (0 = cr*max_vision_len)"
echo "  COMPRESSION_RATE  = ${COMPRESSION_RATE}"
echo "  MAX_SAMPLES       = ${MAX_SAMPLES} (0 = all)"
echo "  TB_DIR            = ${TB_DIR}"
echo "  LOG_FILE          = ${LOG_FILE}"
echo "======================================================================"
echo "TensorBoard:  tensorboard --logdir ${TB_DIR} --port 6006 --bind_all"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Step 1/2: Train
# -----------------------------------------------------------------------------
echo; echo "===== Step 1/2: Train CVLM -> ${OUTPUT_DIR} ====="
TRAIN_ARGS=(
  "${ROOT}/src/train_cvlm.py"
  --output_dir "${OUTPUT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --model_name_or_path "${MODEL_NAME}"
  --text_encoder_name "${TEXT_ENCODER_NAME}"
  --compression_rate "${COMPRESSION_RATE}"
  --max_samples "${MAX_SAMPLES}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --max_prompt_len "${MAX_PROMPT_LEN}"
  --max_answer_len "${MAX_ANSWER_LEN}"
  --max_vision_len "${MAX_VISION_LEN}"
  --max_source_len "${MAX_SOURCE_LEN}"
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
# Step 2/2: Locate latest checkpoint and evaluate
# -----------------------------------------------------------------------------
CKPT=$(ls -1 "${OUTPUT_DIR}"/model_step_*.safetensors 2>/dev/null \
       | sed -E 's|.*model_step_([0-9]+)\.safetensors$|\1 &|' \
       | sort -n \
       | tail -n1 \
       | awk '{print $2}')
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no model_step_*.safetensors found in ${OUTPUT_DIR}"; exit 1
fi
STEP=$(basename "${CKPT}" | sed -E 's|model_step_([0-9]+)\.safetensors|\1|')
echo; echo "===== Step 2/2: Eval checkpoint ${CKPT} (step=${STEP}) ====="

for MODE in ${EVAL_MODES}; do
  echo; echo "----- eval mode=${MODE} -----"
  python "${ROOT}/src/eval_cvlm.py" \
    --checkpoint_path "${CKPT}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_split "${EVAL_SPLIT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --text_encoder_name "${TEXT_ENCODER_NAME}" \
    --compression_rate "${COMPRESSION_RATE}" \
    --mode "${MODE}" \
    --max_samples "${EVAL_MAX_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --max_answer_len "${MAX_ANSWER_LEN}" \
    --max_vision_len "${MAX_VISION_LEN}" \
    --max_source_len "${MAX_SOURCE_LEN}" \
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
