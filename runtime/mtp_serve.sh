#!/bin/bash
# mtp_serve.sh — native qwen3_5 MTP-N head for the comparison (`--profile mtp`).
# The MTP head (1 layer, reuses target KV/embed/lm_head) is in the Intel
# checkpoint (mtp.layers.0) — no separate drafter, no unify patch needed.
#   $1 = num_speculative_tokens (default 2 = the "MTP-2" recipe); $2 = backend.
set -euo pipefail
NSPEC="${1:-3}"
BACKEND="${2:-flash_attn}"
MODEL="${MODEL:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM="${GPU_MEM:-0.82}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
PORT="${PORT:-8000}"
# Reclaim the CUDA-graph memory over-estimate to KV (see serve.sh). Set =1 to restore.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"


# FLA sm121 big-tile shmem fix (prefill/TTFT only on sm121; harmless, free).
echo "[serve] FLA sm121 big-tile shmem patch"
python3 /host/patch_fla_shmem.py || true

if [ "${INC_HYBRID:-0}" = "1" ]; then
  echo "[serve] hybrid INT4+FP8 dispatch patch (inc.py)"
  python3 /host/patch_inc_hybrid.py
fi
if [ "${INT8_LMHEAD_V3:-0}" = "1" ]; then
  echo "[serve] int8 lm-head v3 patch (batched w8a16 GEMV)"
  python3 /host/patch_int8_lmhead_v3.py
fi

python3 /host/patch_unify2.py || { [ "$NSPEC" = "0" ] && true; }

PREFIX_CACHE="${PREFIX_CACHE:-1}"
if [ "$PREFIX_CACHE" != "0" ]; then
  echo "[serve] prefix caching ON (default) — applying align-aware hash_block_size fix"
  python3 /host/patch_prefix_align.py
  PREFIX_ARG=(--enable-prefix-caching)
else
  echo "[serve] prefix caching OFF (PREFIX_CACHE=0)"
  PREFIX_ARG=(--no-enable-prefix-caching)
fi


# Auto tool-calling + reasoning split (see serve.sh). Off via TOOL_PARSER="" etc.
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_ARG=()
[ -n "$TOOL_PARSER" ] && TOOL_ARG+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && TOOL_ARG+=(--reasoning-parser "$REASONING_PARSER")
echo "[mtp] qwen3_5_mtp — backend=$BACKEND, num_speculative_tokens=$NSPEC, model=$MODEL"
exec vllm serve "$MODEL" \
  --served-model-name qwen \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM" \
  "${PREFIX_ARG[@]}" \
  --enable-chunked-prefill \
  --trust-remote-code \
        --chat-template /host/Qwen3.5-122B-A10B_ha.jinja \
        --scheduling-policy priority \
        --optimization-level 3 \
        --performance-mode throughput \
  --load-format "$LOAD_FORMAT" \
  --attention-backend "$BACKEND" \
  "${TOOL_ARG[@]}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}"
