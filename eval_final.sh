#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA-2-7b-hf/WebQSP_webqsp_100_7_32_16/evaluation_beam/beam_test_top_k_predictions.json
CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset CWQ --pred_file Reading/LLaMA-2-13b-hf/CWQ_cwq_4_16_32_16/evaluation_beam/beam_test_top_k_predictions.json
