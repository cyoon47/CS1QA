OUTPUT_DIR=/home/lama/nas/code/adb_int/outputs/kgqa
CHECKPOINT_DIR=/home/lama/nas/code/adb_int/checkpoints

python photoshopquia_train.py \
    --train_file $OUTPUT_DIR/codeqa/q_train.jsonl \
    --eval_file $OUTPUT_DIR/codeqa/q_dev.jsonl \
    --corpus_file $OUTPUT_DIR/codeqa/corpus.jsonl \
    --max_length 512 \
    --n_hard_negs 2 \
    --init_checkpoint facebook/dpr-question_encoder-single-nq-base\
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --n_epochs 20 \
    --learning_rate 2e-5 \
    --checkpoint_save_dir $CHECKPOINT_DIR/kgqa/bi_encoder/codeqa_bi_encoder \
    --backbone_model dpr \
    --dataset photoshopquia \
    --bm25_file $OUTPUT_DIR/codeqa/bm25_scores.jsonl
