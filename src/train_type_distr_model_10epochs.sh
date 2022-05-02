for SEED in 7 42 312 
do
    for MODEL in roberta-base microsoft/codebert-base
    do
        python type_classification/classify_type.py --model_type roberta --task_name typeclassification --do_train --do_eval --eval_all_checkpoints \
            --train_file train_cleaned.jsonl --dev_file dev_cleaned.jsonl --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
            --learning_rate 1e-5 --num_train_epochs 10 --gradient_accumulation_steps 1 --overwrite_output_dir --seed $SEED \
            --data_dir ../../data/final/cleaned/equal --output_dir ../../../nas/augmented/full/seed$SEED/${MODEL} --model_name_or_path $MODEL \
            --tokenizer_name $MODEL

        python span_prediction/select_span.py --model_type roberta --task_name spanselection --do_train --do_eval --eval_all_checkpoints \
            --train_file train_cleaned_attn.jsonl --dev_file dev_cleaned_attn.jsonl --max_seq_length 4096 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
            --learning_rate 1e-5 --num_train_epochs 10 --gradient_accumulation_steps 1 --overwrite_output_dir --seed $SEED \
            --data_dir ../../data/final/cleaned/equal --output_dir ../../../nas/span_prediction/final_cleaned/equal/10epochs/$MODEL/seed$SEED --model_name_or_path $MODEL --return_type \
            --tokenizer_name $MODEL
        
    done

    for MODEL in xlm-roberta-base
    do
        python type_classification/classify_type.py --model_type xlm --task_name typeclassification --do_train --do_eval --eval_all_checkpoints \
            --train_file train_cleaned.jsonl --dev_file dev_cleaned.jsonl --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
            --learning_rate 1e-5 --num_train_epochs 10 --gradient_accumulation_steps 1 --overwrite_output_dir --seed $SEED \
            --data_dir ../../data/final/cleaned/equal --output_dir ../../../nas/augmented/full/seed$SEED/${MODEL} --model_name_or_path $MODEL \
            --tokenizer_name $MODEL

        python span_prediction/select_span.py --model_type xlm --task_name spanselection --do_train --do_eval --eval_all_checkpoints \
            --train_file train_cleaned_attn.jsonl --dev_file dev_cleaned_attn.jsonl --max_seq_length 4096 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
            --learning_rate 1e-5 --num_train_epochs 10 --gradient_accumulation_steps 1 --overwrite_output_dir --seed $SEED \
            --data_dir ../../data/final/cleaned/equal --output_dir ../../../nas/span_prediction/final_cleaned/equal/10epochs/$MODEL/seed$SEED --model_name_or_path $MODEL --return_type \
            --tokenizer_name $MODEL
    done
done