#!/bin/bash
export LMUData=/data/pinci/datasets/LMUData
export GPU=$(nvidia-smi --list-gpus | wc -l)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export PYTHONWARNINGS="ignore"

WORKSPACE=../
cd $WORKSPACE

MODEL_PATH=/data/pinci/ckpt/huggingface/llava-onevision-qwen2-0.5b-ov
MODEL_NAME=llava_onevision_qwen2_0.5b_ov
RAG_MODEL_PATH=/data/pinci/ckpt/huggingface/VisRAG-Ret
VISION_TOWER_PATH=/data/pinci/ckpt/huggingface/siglip-so400m-patch14-384

dataset=HRBench4K
processed_dataset=HRBench4K_single
work_dir=./outputs/${MODEL_NAME}
PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
mkdir -p $work_dir
# torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --vision_tower_path $VISION_TOWER_PATH --max_step 200
# torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --vision_tower_path $VISION_TOWER_PATH

python run.py --data $processed_dataset \
              --model $MODEL_NAME \
              --judge chatgpt-0125 \
              --work-dir $work_dir \
              --model_path $MODEL_PATH \
              --is_process_image \
              --processed_image_path $PROCESSED_IMAGE_PATH \
              --rag_model_path $RAG_MODEL_PATH \
              --vision_tower_path $VISION_TOWER_PATH \
              --max_step 200

python run.py --data $dataset \
              --model $MODEL_NAME \
              --judge chatgpt-0125 \
              --work-dir $work_dir \
              --model_path $MODEL_PATH \
              --processed_image_path $PROCESSED_IMAGE_PATH \
              --rag_model_path $RAG_MODEL_PATH \
              --vision_tower_path $VISION_TOWER_PATH
# dataset=HRBench8K
# processed_dataset=HRBench8K_single
# work_dir=./outputs/${MODEL_NAME}
# PROCESSED_IMAGE_PATH=$work_dir/${dataset}/images
# mkdir -p $work_dir
# torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $processed_dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --is_process_image --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH --max_step 200
# torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data $dataset --model $MODEL_NAME --judge chatgpt-0125 --work-dir $work_dir --model_path $MODEL_PATH --processed_image_path $PROCESSED_IMAGE_PATH --rag_model_path $RAG_MODEL_PATH