# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import PIL

from datasets import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser
)

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging

from open_r1.trainer import VLMSFTTrainer

logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

@dataclass
class SFTScriptArguments:
    """
    Script arguments for the SFT training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype to use for training"},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    freeze_vision_modules: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the vision modules"}
    )
    peft_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to PEFT config file"}
    )

def main():
    parser = HfArgumentParser((SFTScriptArguments, TrainingArguments, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Load the JSONL datasets
    import json
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'image' in item:
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(image_folder, item['image'])
                    del item['image']  # remove the image column so that it can be loaded later
                
                # Remove immediate image loading
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                # Handle solution that could be a float or string
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value
                else:
                    # If it's a float or other non-string type, keep it as is
                    item['solution'] = str(solution_value)
                

                del item['conversations']
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        # Check if required keys exist
        if 'problem' not in example:
            print(f"Warning: 'problem' key missing in example with keys: {list(example.keys())}")
            # Add a default problem if missing
            example['problem'] = "No problem provided"
        
        if 'image_path' not in example:
            print(f"Warning: 'image_path' key missing in example with keys: {list(example.keys())}")
            # Add a default image_path if missing
            example['image_path'] = None

        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                'image_path': example['image_path'],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': f"{example['solution']}",
                'prompt': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'text': None},
                            {'type': 'text', 'text': example['problem']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': example['solution']}
                        ]
                    }
                ],
                'input_ids': None,  # Will be filled during training
                'attention_mask': None,  # Will be filled during training
                'labels': None,  # Will be filled during training
            }
        else:
            return {
                'problem': example['problem'],
                'solution': f"{example['solution']}",
                'prompt': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': example['problem']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': example['solution']}
                        ]
                    }
                ],
                'input_ids': None,  # Will be filled during training
                'attention_mask': None,  # Will be filled during training
                'labels': None,  # Will be filled during training
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Load model and processor
    model_name = model_args.model_name_or_path
    
    # Set up torch dtype
    torch_dtype = getattr(torch, script_args.torch_dtype) if script_args.torch_dtype != "auto" else "auto"
    
    # Load the appropriate model based on model name
    if "Qwen2-VL" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            attn_implementation=script_args.attn_implementation
        )
        processor = AutoProcessor.from_pretrained(model_name)
        processor.image_processor.max_pixels = script_args.max_pixels
        processor.image_processor.min_pixels = script_args.min_pixels
    elif "Qwen2.5-VL" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            attn_implementation=script_args.attn_implementation
        )
        processor = AutoProcessor.from_pretrained(model_name)
        processor.image_processor.max_pixels = script_args.max_pixels
        processor.image_processor.min_pixels = script_args.min_pixels
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Apply PEFT if specified
    peft_config = None
    if model_args.peft_config_path:
        from peft import get_peft_config
        peft_config = get_peft_config(model_args.peft_config_path)
    
    # Initialize the Trainer
    training_args.remove_unused_columns = False
    # We don't need to define a data_collator here as it's already defined in VLMSFTTrainer
    trainer = VLMSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation'),
        processing_class=processor,
        peft_config=peft_config,
        freeze_vision_modules=model_args.freeze_vision_modules,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        attn_implementation=script_args.attn_implementation,
        torch_dtype=script_args.torch_dtype,
    )

    # Train the model
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()