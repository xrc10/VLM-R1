# Using VLLM with GRPO Trainer

This guide explains how to use VLLM (Very Large Language Model) inference with the GRPO (Group Relative Policy Optimization) trainer for efficient training of large language models.

## Prerequisites

### Docker Environment
```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel-vlmr1-v0.2-vllm
```

### Required Packages
- transformers
- vllm
- torch
- accelerate
- deepspeed
- peft

## Key Features

- VLLM integration for faster inference during training
- Support for multimodal models (Qwen2-VL, Qwen2.5-VL)
- DeepSpeed ZeRO-3 optimization
- Flash Attention 2 support
- Gradient checkpointing
- Distributed training support

## Example Training Script

```bash
#!/bin/bash

# Set environment variables
export EXP_NAME="your_experiment_name"
export BATCH_SIZE=4
export RUN_NAME="your_run_name"

# Launch training
torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed configs/zero3.json \
    --output_dir runs/${EXP_NAME}/batch_size_${BATCH_SIZE} \
    --model_name_or_path path/to/your/model \
    --dataset_name data_config/rec.yaml \
    --image_root path/to/image/data \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_iterations 1 \
    --max_steps 500 \
    --run_name ${RUN_NAME}_batch_${BATCH_SIZE} \
    --save_steps 500 \
    --save_only_model true \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_dtype bfloat16 \
    --num_generations 7
```

## Important Parameters

### VLLM-specific Parameters
- `use_vllm`: Enable VLLM for inference (default: false)
- `vllm_gpu_memory_utilization`: GPU memory utilization for VLLM (0.0-1.0)
- `vllm_dtype`: Data type for VLLM inference (float16, bfloat16)

### Training Parameters
- `num_generations`: Number of generations per prompt (G in GRPO paper)
- `num_iterations`: Number of iterations per batch (μ in GRPO paper)
- `max_prompt_length`: Maximum length of input prompts
- `max_completion_length`: Maximum length of generated completions

### Hardware Configuration
- `nproc_per_node`: Number of GPUs to use (N-1 for training, 1 for VLLM)
- `gradient_accumulation_steps`: Number of steps to accumulate gradients

## Important Notes

1. When using VLLM, reserve one GPU exclusively for VLLM inference:
   - Use `nproc_per_node = total_gpus - 1`
   - VLLM will automatically use the last available GPU

2. Generation Length Requirements:
   - VLLM speedup is only noticeable when generation length is >= 200 tokens
   - For shorter generations, the tensor broadcasting overhead dominates and traditional generation may be faster
   - Consider disabling VLLM (`use_vllm=false`) if your use case involves short generations

3. Model Compatibility:
   - For older transformers versions, you may need to update the preprocessor_config.json and tokenizer_config.json files with the latest versions from the Qwen2.5-VL release.

4. Memory Management:
   - Adjust `vllm_gpu_memory_utilization` based on your GPU memory
   - Use gradient checkpointing if needed for larger models

5. Distributed Training:
   - VLLM runs on the main process only
   - Training is distributed across remaining GPUs

## Troubleshooting

1. If you encounter CUDA out of memory errors:
   - Reduce batch size
   - Lower `vllm_gpu_memory_utilization`
   - Enable gradient checkpointing

2. If you see tokenizer/preprocessor errors:
   - Update config files from the latest model release
   - Ensure image preprocessing settings match model requirements

3. For VLLM-specific issues:
   - Check GPU allocation
   - Verify CUDA compatibility
   - Monitor GPU memory usage
