#!/bin/bash
# Run three schema serialization experiments and evaluate each.
# Approach 1: table descriptions derived from training data
# Approach 2: FK relationship links shown in schema
# Approach 3: columns sorted by relevance to question
# Also runs a combined approach (all three together).

set -e
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCHEMAS="Project2/schemas"
VAL_INPUT="Project2/validation_input.json"
VAL_GOLD="Project2/validation_gold_schema_links.json"
TRAIN="Project2/train.json"
VAL="Project2/validation.json"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
EPOCHS=5
BATCH=1
GRAD_ACCUM=16
LR=1e-4
MAX_SEQ=2048

run_experiment() {
    local name=$1
    local extra_args=$2

    echo ""
    echo "========================================"
    echo "EXPERIMENT: $name"
    echo "========================================"

    python train_simple.py \
        --model $MODEL \
        --train $TRAIN \
        --validation $VAL \
        --schemas_dir $SCHEMAS \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --max_seq_len $MAX_SEQ \
        --output_dir "./runs/$name" \
        $extra_args

    echo "Running inference for $name..."
    cp -r "runs/$name/adapter/." adapter/
    python main.py \
        --input $VAL_INPUT \
        --output "preds_${name}.json" \
        --schemas_dir $SCHEMAS \
        --batch_size 1

    echo "Evaluating $name..."
    python Project2/eval.py \
        --predictions "preds_${name}.json" \
        --gold $VAL_GOLD \
        --schemas_dir $SCHEMAS \
        --questions_input $VAL_INPUT \
        --per_question_out "per_q_${name}.csv" \
        | grep -E "Leaderboard|Table Score|Column Score"
}

# Approach 1: table descriptions
run_experiment "approach1-table-desc" "--use_table_desc"

# Approach 2: FK links
run_experiment "approach2-fk-links" "--show_fk_links"

# Approach 3: column relevance sorting
run_experiment "approach3-col-sort" "--sort_by_question"

# Combined: all three
run_experiment "approach-combined" "--use_table_desc --show_fk_links --sort_by_question"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
for name in approach1-table-desc approach2-fk-links approach3-col-sort approach-combined; do
    echo -n "$name: "
    python Project2/eval.py \
        --predictions "preds_${name}.json" \
        --gold $VAL_GOLD \
        --schemas_dir $SCHEMAS \
        --questions_input $VAL_INPUT 2>/dev/null \
        | grep "Leaderboard"
done
