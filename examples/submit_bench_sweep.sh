#!/usr/bin/env bash
# Submit one Slurm job per (algo, topk, bs, in_len) setting in a 4-D sweep
# of sglang.bench_one_batch on Qwen3-4B. Each job runs
# `bench_one_batch_one.sh` with the corresponding env vars and writes a single
# jsonl line to bench_results/Qwen3-4B/.
#
# Defaults reproduce the sweep used to populate BENCH_ONE_BATCH.md. All can be
# overridden via env, e.g.:
#     ALGOS_SPARSE="gqa_quest_sparse_attention" TOPKS="61 125" \
#     BSS="1 4" IN_LENS="8192" \
#     bash examples/submit_bench_sweep.sh
#
# Run from the vortex_torch repo root.
set -euo pipefail

ACCOUNT="${ACCOUNT:-liuv-pennnetworks}"
SBATCH_FILE="${SBATCH_FILE:-bench_one_batch_one.sbatch}"

ALGOS_SPARSE=(${ALGOS_SPARSE:-block_sparse_attention gqa_quest_sparse_attention})
TOPKS=(${TOPKS:-29 93 253})
BSS=(${BSS:-1 4 16})
IN_LENS=(${IN_LENS:-8192 16384})
OUT_LEN="${OUT_LEN:-8192}"
INCLUDE_DENSE="${INCLUDE_DENSE:-1}"

if [ ! -f "$SBATCH_FILE" ]; then
    echo "ERROR: sbatch file '$SBATCH_FILE' not found (run from repo root or set SBATCH_FILE)." >&2
    exit 2
fi

submit() {
    local name="$1" exports="$2"
    sbatch --parsable \
        --account="$ACCOUNT" \
        --job-name="$name" \
        --export=ALL,"$exports" \
        "$SBATCH_FILE"
}

short_name() {
    case "$1" in
        block_sparse_attention) echo "block" ;;
        gqa_quest_sparse_attention) echo "quest" ;;
        *) echo "$1" ;;
    esac
}

total=0

for ALGO in "${ALGOS_SPARSE[@]}"; do
    SHORT=$(short_name "$ALGO")
    for TOPK in "${TOPKS[@]}"; do
        for BS in "${BSS[@]}"; do
            for IN in "${IN_LENS[@]}"; do
                IN_K=$((IN / 1024))
                NAME="bob-${SHORT}-tk${TOPK}-bs${BS}-in${IN_K}k"
                jid=$(submit "$NAME" \
                    "ALGO=$ALGO,TOPK_VAL=$TOPK,BS=$BS,IN_LEN=$IN,OUT_LEN=$OUT_LEN")
                echo "$NAME -> $jid"
                total=$((total + 1))
            done
        done
    done
done

if [ "$INCLUDE_DENSE" = "1" ]; then
    for BS in "${BSS[@]}"; do
        for IN in "${IN_LENS[@]}"; do
            IN_K=$((IN / 1024))
            NAME="bob-dense-bs${BS}-in${IN_K}k"
            jid=$(submit "$NAME" \
                "ALGO=dense,BS=$BS,IN_LEN=$IN,OUT_LEN=$OUT_LEN")
            echo "$NAME -> $jid"
            total=$((total + 1))
        done
    done
fi

echo "submitted: $total"
