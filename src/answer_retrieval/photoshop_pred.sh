OUTPUT_DIR=../../yeonseonwoo/code/adb_int/outputs/kgqa
CHECKPOINT_DIR=../../yeonseonwoo/code/adb_int/checkpoints

python photoshopquia_pred.py \
    --eval_file $OUTPUT_DIR/codeqa/q_test.jsonl \
    --corpus_file $OUTPUT_DIR/codeqa/corpus.jsonl \
    --dataset photoshopquia \
    --bm25_file $OUTPUT_DIR/codeqa/bm25_scores.jsonl \
    --max_length 512 \
    --backbone_model dpr \
    --init_checkpoint $CHECKPOINT_DIR/kgqa/bi_encoder/codeqa_bi_encoder_last_checkpoint \
    --per_device_eval_batch_size 1 \
    --eval_steps 500
