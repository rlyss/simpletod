


export MODEL=$2
export MODEL_NAME=$3
export BATCH=$4
export OUTPUT=output/${MODEL_NAME}
# export TRAIN_FILE=./resources/gpt2/train.history_belief
# export TEST_FILE=./resources/gpt2/val.history_belief
export TRAIN_FILE=./resources/gpt2/train.history_belief_hub
export TRAIN_DICT_FILE=./resources/gpt2/train.id_to_hub
export TEST_FILE=./resources/gpt2/test.history_belief_hub
export TEST_DICT_FILE=./resources/gpt2/test.id_to_hub


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --output_dir=$OUTPUT \
    --model_type=$MODEL \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --train_dict_file=$TRAIN_DICT_FILE \
    --user_hubert \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --eval_dict_file=$TEST_DICT_FILE \
    --evaluate_during_training \
    --save_steps 10000 \
    --logging_steps 1000 \
    --per_gpu_train_batch_size $BATCH \
    --num_train_epochs 100