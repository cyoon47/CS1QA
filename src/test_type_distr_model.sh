for seed in 7 42 312 
do
    for DIST in equal
    do
        for MODEL in xlm-roberta-base
        do
            for CHECKPOINT in best
            do
                python type_classification/classify_type.py --model_type xlm --model_name_or_path $MODEL --task_name typeclassification --do_predict --output_dir ./models \
                    --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
                    --learning_rate 1e-5 --num_train_epochs 10 --test_file test_cleaned.jsonl  --pred_model_dir ../../../nas/augmented/full/seed$seed/${MODEL}/checkpoint-$CHECKPOINT\
                    --data_dir ../../data/final/cleaned/equal --test_result_dir ./results/augmented/type_classification/$MODEL/$DIST/pred_result_${CHECKPOINT}_${seed}_full.txt \
                    --tokenizer_name $MODEL --seed $seed

                python type_classification_text_only/classify_type.py --model_type xlm --model_name_or_path $MODEL --task_name typeclassification --do_predict --output_dir ./models \
                    --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
                    --learning_rate 1e-5 --num_train_epochs 10 --test_file test_cleaned.jsonl  --pred_model_dir ../../../nas/augmented/text_only/seed$seed/${MODEL}/checkpoint-$CHECKPOINT\
                    --data_dir ../../data/final/cleaned/equal --test_result_dir ./results/augmented/type_classification/$MODEL/$DIST/pred_result_${CHECKPOINT}_${seed}_text_only.txt \
                    --tokenizer_name $MODEL --seed $seed

                python span_prediction/select_span.py --model_type xlm --model_name_or_path $MODEL --task_name spanselection --do_predict --output_dir ./models \
                    --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
                    --learning_rate 1e-5 --num_train_epochs 10 --test_file test_cleaned.jsonl --pred_model_dir ../../../nas/span_prediction/final_cleaned/10epochs/seed$seed/$DIST/${MODEL}_en/checkpoint-$CHECKPOINT \
                    --data_dir ../../data/final/cleaned/equal --test_result_dir ./results/final_cleaned/span_prediction/$MODEL/$DIST/pred_result_code_${CHECKPOINT}_${seed}_en.txt --return_type \
                    --tokenizer_name $MODEL --seed $seed

            done
        done

        for MODEL in roberta-base microsoft/codebert-base
        do
            for CHECKPOINT in best
            do
                python type_classification/classify_type.py --model_type roberta --model_name_or_path $MODEL --task_name typeclassification --do_predict --output_dir ./models \
                    --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
                    --learning_rate 1e-5 --num_train_epochs 10 --test_file test_cleaned.jsonl  --pred_model_dir ../../../nas/augmented/full/seed$seed/${MODEL}/checkpoint-$CHECKPOINT\
                    --data_dir ../../data/final/cleaned/equal --test_result_dir ./results/augmented/type_classification/$MODEL/$DIST/pred_result_${CHECKPOINT}_${seed}_full.txt \
                    --tokenizer_name $MODEL --seed $seed

                python type_classification_text_only/classify_type.py --model_type roberta --model_name_or_path $MODEL --task_name typeclassification --do_predict --output_dir ./models \
                    --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
                    --learning_rate 1e-5 --num_train_epochs 10 --test_file test_cleaned.jsonl  --pred_model_dir ../../../nas/augmented/text_only/seed$seed/${MODEL}/checkpoint-$CHECKPOINT\
                    --data_dir ../../data/final/cleaned/equal --test_result_dir ./results/augmented/type_classification/$MODEL/$DIST/pred_result_${CHECKPOINT}_${seed}_text_only.txt \
                    --tokenizer_name $MODEL --seed $seed

                python span_prediction/select_span.py --model_type roberta --model_name_or_path $MODEL --task_name spanselection --do_predict --output_dir ./models \
                    --max_seq_length 512 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
                    --learning_rate 1e-5 --num_train_epochs 10 --test_file test_cleaned.jsonl --pred_model_dir ../../../nas/span_prediction/final_cleaned/10epochs/seed$seed/$DIST/${MODEL}_en/checkpoint-$CHECKPOINT \
                    --data_dir ../../data/final/cleaned/equal --test_result_dir ./results/final_cleaned/span_prediction/$MODEL/$DIST/pred_result_code_${CHECKPOINT}_${seed}_en.txt --return_type \
                    --tokenizer_name $MODEL --seed $seed
            done
        done
    done
done