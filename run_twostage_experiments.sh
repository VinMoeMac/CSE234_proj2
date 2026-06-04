#!/bin/bash
# Test three two-stage variants sequentially
# Run: tmux new -s ts-exp && conda activate cse234 && cd ~/CSE234_proj2 && bash run_twostage_experiments.sh 2>&1 | tee ts_exp.log

set -e
SCHEMAS="Project2/schemas"
VAL_INPUT="Project2/validation_input.json"
VAL_GOLD="Project2/validation_gold_schema_links.json"

eval_preds() {
    local name=$1
    local pred_file=$2
    echo -n "$name: "
    python Project2/eval.py \
        --predictions "$pred_file" \
        --gold $VAL_GOLD \
        --schemas_dir $SCHEMAS \
        --questions_input $VAL_INPUT 2>/dev/null \
        | grep "Leaderboard"
}

echo "$(date): Starting two-stage experiments"

# ---------------------------------------------------------------
# Exp 1: original 0.425 adapter + two-stage (no camelCase mismatch)
# ---------------------------------------------------------------
echo ""
echo "=== Exp 1: 0.425 adapter (augmented-fk-sorted) + two-stage ==="
cp -r runs/augmented-fk-sorted/adapter/. adapter/
python main.py --input $VAL_INPUT --output preds_ts_425.json \
    --schemas_dir $SCHEMAS --batch_size 1 --two_stage
eval_preds "ts_425" preds_ts_425.json

# ---------------------------------------------------------------
# Exp 2: camel-split-v1 + two-stage with top-3 tables fallback
# The idea: if stage1 predicts 0-1 tables, also pass the top-2
# keyword-matched tables to stage2 to catch missed tables
# ---------------------------------------------------------------
echo ""
echo "=== Exp 2: camel-split-v1 + two-stage + top-k table fallback ==="
cp -r runs/camel-split-v1/adapter/. adapter/
python main.py --input $VAL_INPUT --output preds_ts_topk.json \
    --schemas_dir $SCHEMAS --batch_size 1 --two_stage --two_stage_topk 3
eval_preds "ts_topk" preds_ts_topk.json

# ---------------------------------------------------------------
# Exp 3: train with two-stage format
# Train the model to output single-table column lists
# Then use two-stage at inference
# ---------------------------------------------------------------
echo ""
echo "=== Exp 3: Retrain with two-stage column format ==="
python train_simple.py \
    --train augmented_data/train_with_sap.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --two_stage_train \
    --output_dir ./runs/two-stage-trained

cp -r runs/two-stage-trained/adapter/. adapter/
python main.py --input $VAL_INPUT --output preds_ts_trained.json \
    --schemas_dir $SCHEMAS --batch_size 1 --two_stage
eval_preds "ts_trained" preds_ts_trained.json

echo ""
echo "=== SUMMARY ==="
eval_preds "current_best (camel-split-v1 + two-stage)" preds_2stage.json
eval_preds "ts_425" preds_ts_425.json
eval_preds "ts_topk" preds_ts_topk.json
eval_preds "ts_trained" preds_ts_trained.json

echo "$(date): Done"
