#!/usr/bin/env bash
# Single-setting sglang.bench_one_batch run on Qwen3-4B.
# Mirrors the user-supplied "mystery" verify_algo.py JSON (topk_ratio=0.0625,
# mem=0.85, vortex_impl_backend=triton, page_size=16, workload_chunk_size=64,
# vortex_attention_backend=trtllm, layers_skip=[0], local model path), differing
# only in the swept dimensions: ALGO, TOPK_VAL, BS, IN_LEN, OUT_LEN.
#
# Required env: ALGO, BS, IN_LEN, OUT_LEN
# Required env when ALGO != dense: TOPK_VAL
# ALGO ∈ {dense, block_sparse_attention, gqa_quest_sparse_attention}
set -euo pipefail

: "${ALGO:?ALGO env var not set}"
: "${BS:?BS env var not set}"
: "${IN_LEN:?IN_LEN env var not set}"
: "${OUT_LEN:?OUT_LEN env var not set}"

MODEL_PATH=/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-4B
MEM=0.85
RESULT_DIR=/vast/projects/liuv/pennnetworks/ljm_data/vortex_torch/bench_results/Qwen3-4B
mkdir -p "$RESULT_DIR"

IN_K=$((IN_LEN / 1024))
OUT_K=$((OUT_LEN / 1024))
MAX_SEQ_LENS=$((IN_LEN + OUT_LEN))

if [ "$ALGO" = "dense" ]; then
    TAG="dense_bs${BS}_in${IN_K}k_out${OUT_K}k"
    VORTEX_FLAG=()
else
    : "${TOPK_VAL:?TOPK_VAL env var not set for ALGO=${ALGO}}"
    TAG="${ALGO}_topk${TOPK_VAL}_bs${BS}_in${IN_K}k_out${OUT_K}k"
    # Mirrors mystery JSON args exactly. compilation_cache_dir is set via
    # cluster_env.sh's $VORTEX_COMPILATION_CACHE_DIR to keep JIT off $HOME.
    VORTEX_JSON=$(printf '{"topk_val": %d, "max_topk_val": 256, "topk_ratio": 0.0625, "block_size": 16, "workload_chunk_size": 64, "module_name": "%s", "attention_backend": "trtllm", "impl_backend": "triton", "use_tensor_core": false, "layers_skip": [0], "max_seq_lens": %d, "compilation_cache_dir": "%s", "block_reserved_bos": 1, "block_reserved_eos": 2}' \
        "$TOPK_VAL" "$ALGO" "$MAX_SEQ_LENS" "${VORTEX_COMPILATION_CACHE_DIR}")
    VORTEX_FLAG=(--vortex-config "$VORTEX_JSON")
fi

OUT_FILE="$RESULT_DIR/${TAG}.jsonl"
rm -f "$OUT_FILE"

echo ">>> bench: ${TAG}"
echo ">>> result: ${OUT_FILE}"
echo ">>> vortex-config (if any): ${VORTEX_FLAG[*]:-<none>}"

python -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --tp-size 1 \
    --kv-cache-dtype auto \
    --mem-fraction-static "$MEM" \
    --attention-backend flashinfer \
    --page-size 16 \
    --context-length 40960 \
    --trust-remote-code \
    --batch-size "$BS" \
    --input-len "$IN_LEN" \
    --output-len "$OUT_LEN" \
    --result-filename "$OUT_FILE" \
    --run-name "$TAG" \
    "${VORTEX_FLAG[@]}"
