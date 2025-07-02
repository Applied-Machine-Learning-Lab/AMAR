#!/bin/bash

CUDA_VISIBLE_DEVICES=0 bash run_ft.sh WebQSP LLaMA-2-7b-hf webqsp_100_7_32_16 test 100 7 0 32 16 15
CUDA_VISIBLE_DEVICES=0 bash run_ft.sh CWQ LLaMA-2-13b-hf cwq_4_16_32_16 test 4 16 0 32 16 8
