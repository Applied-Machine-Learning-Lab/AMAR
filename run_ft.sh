#!/bin/bash

data=$1 # 'CWQ'
llm_model=$2 # 'LLaMA-2-13b-hf'
setting=$3
mode=$4
topk=$5
soft_prompt_length=$6
cuda=$7
extra_infor_len=$8
gate_len=${9}
num_beams=${10}

if [ "$data" == "WebQSP" ]; then
    num_train_epochs=80.0
else
    num_train_epochs=8.0
fi


if [[ $mode == *"train"* ]]; then
    python -u LLMs/LLaMA/src/train_bash.py \
            --stage sft \
            --model_name_or_path ./../../LLM_checkpoint/${llm_model} \
            --do_train  \
            --dataset_dir LLMs/data_id \
            --dataset ${data}_Freebase_NQ_train \
            --template default  \
            --finetuning_type lora \
            --lora_target gate_proj,down_proj,up_proj \
            --output_dir Reading/${llm_model}/${data}_${setting}/checkpoint \
            --overwrite_cache \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 4  \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --save_strategy no \
            --learning_rate 5e-5  \
            --num_train_epochs ${num_train_epochs} \
            --plot_loss \
            --overwrite_output_dir \
            --topk ${topk} \
            --soft_prompt_length ${soft_prompt_length} \
            --extra_infor_len ${extra_infor_len} \
            --gate_len ${gate_len}
fi



if [[ $mode == *"test"* ]]; then
    python -u LLMs/LLaMA/src/beam_output_eva.py  \
            --model_name_or_path  ./../../LLM_checkpoint/${llm_model}   \
            --dataset_dir LLMs/data_id  \
            --dataset ${data}_Freebase_NQ_test  \
            --template default  \
            --finetuning_type lora  \
            --lora_target gate_proj,down_proj,up_proj \
            --checkpoint_dir Reading/${llm_model}/${data}_${setting}/checkpoint  \
            --num_beams ${num_beams}  \
            --max_new_tokens 256 \
            --topk ${topk} \
            --soft_prompt_length ${soft_prompt_length} \
            --extra_infor_len ${extra_infor_len} \
            --gate_len ${gate_len}
fi
