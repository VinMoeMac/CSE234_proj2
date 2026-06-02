#!/bin/bash
# Overnight experiment runner — runs sequentially, logs all results
# Start with: tmux new -s overnight
# Inside tmux: conda activate cse234 && cd CSE234_proj2 && HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash run_overnight.sh 2>&1 | tee overnight_v2.log
# Detach: Ctrl+B D

set -e
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCHEMAS="Project2/schemas"
VAL_INPUT="Project2/validation_input.json"
VAL_GOLD="Project2/validation_gold_schema_links.json"
TRAIN="Project2/train.json"
TRAIN_AUG="augmented_data/train_with_sap.json"
VAL="Project2/validation.json"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

run_and_eval() {
    local name=$1
    shift
    echo ""
    echo "========================================"
    echo "$(date): STARTING $name"
    echo "========================================"

    python train_simple.py \
        --model $MODEL \
        --validation $VAL \
        --schemas_dir $SCHEMAS \
        --output_dir "./runs/$name" \
        "$@"

    cp -r "runs/$name/adapter/." adapter/

    python main.py \
        --input $VAL_INPUT \
        --output "preds_${name}.json" \
        --schemas_dir $SCHEMAS \
        --batch_size 1

    echo -n "$name: "
    python Project2/eval.py \
        --predictions "preds_${name}.json" \
        --gold $VAL_GOLD \
        --schemas_dir $SCHEMAS \
        --questions_input $VAL_INPUT \
        --per_question_out "per_q_${name}.csv" \
        | grep "Leaderboard"

    echo "$(date): DONE $name"
}

# Exp 1: FK + sorting, original data (ablation — how much does synthetic help?)
run_and_eval "fk-sorted-v1" \
    --train $TRAIN \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question

# Exp 2: FK + sorting + augmented + more epochs (best config so far + longer)
run_and_eval "augmented-10ep" \
    --train $TRAIN_AUG \
    --epochs 10 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question

# Exp 3: FK + sorting + augmented + higher LoRA rank
run_and_eval "augmented-r64" \
    --train $TRAIN_AUG \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 64

# Exp 4: FK + sorting + augmented + lower LR (less overfitting)
run_and_eval "augmented-lr5e5" \
    --train $TRAIN_AUG \
    --epochs 7 --batch_size 1 --grad_accum 16 --lr 5e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question

# Exp 5: FK + sorting + augmented + oversample underrepresented DBs
run_and_eval "augmented-oversample" \
    --train $TRAIN_AUG \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --oversample_factor 3

echo ""
echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"
for name in fk-sorted-v1 augmented-10ep augmented-r64 augmented-lr5e5 augmented-oversample; do
    if [ -f "preds_${name}.json" ]; then
        echo -n "$name: "
        python Project2/eval.py \
            --predictions "preds_${name}.json" \
            --gold $VAL_GOLD \
            --schemas_dir $SCHEMAS \
            --questions_input $VAL_INPUT 2>/dev/null \
            | grep "Leaderboard"
    fi
done

echo ""
echo "Best adapter from current run: augmented-fk-sorted (0.425)"
echo "Check overnight_v2.log for full output"
