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
from typing import Any, Optional, Union

import torch
import PIL.Image
from datasets import Dataset, IterableDataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import is_peft_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

if is_peft_available():
    from peft import PeftConfig, get_peft_model

class VLMSFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) of Vision-Language Models.
    
    This trainer extends the Hugging Face Trainer to handle multimodal inputs,
    specifically for vision-language models like Qwen2-VL and Qwen2.5-VL.

    Args:
        model (`Union[str, PreTrainedModel]`):
            The model to train, fine-tune or evaluate.
        args (`TrainingArguments`):
            The arguments to use for training.
        train_dataset (`Optional[Union[Dataset, IterableDataset]]`):
            The dataset to use for training.
        eval_dataset (`Optional[Union[Dataset, IterableDataset]]`):
            The dataset to use for evaluation.
        processing_class (`Optional[PreTrainedTokenizerBase]`):
            The processor to use for tokenization and image processing.
        callbacks (`Optional[list[TrainerCallback]]`):
            A list of callbacks to customize the training loop.
        peft_config (`Optional[PeftConfig]`):
            The PEFT configuration to use for parameter-efficient fine-tuning.
        freeze_vision_modules (`bool`):
            Whether to freeze the vision modules during training.
        max_pixels (`int`):
            Maximum number of pixels for the image processor.
        min_pixels (`int`):
            Minimum number of pixels for the image processor.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: Any,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: bool = False,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
    ):
        # Load model if string is provided
        model_init_kwargs = {"attn_implementation": attn_implementation}
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        else:
            model_init_kwargs["torch_dtype"] = torch_dtype
            
        if isinstance(model, str):
            model_id = model
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                raise ValueError(f"Unsupported model: {model_id}")
        else:
            model_id = model.config._name_or_path

        # Initialize processor if not provided
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            if "Qwen" in model_id:
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels

        # Apply PEFT if config is provided
        self.vision_modules_keywords = ["visual"]
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in list(lora_module_names):  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            
            if not peft_config.target_modules:
                target_modules = find_all_linear_names(model, self.vision_modules_keywords)
                peft_config.target_modules = target_modules
            
            model = get_peft_model(model, peft_config)

        # Freeze vision modules if requested
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Data collator for SFT
        def data_collator(examples):
            device = self.accelerator.device if hasattr(self, "accelerator") else "cpu"
            
            prompts = [x["prompt"] for x in examples]
            prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in examples]
            
            # Handle both pre-loaded images and image paths
            images = []
            for x in examples:
                if "image" in x:
                    img = x["image"]
                elif "image_path" in x and x["image_path"] is not None:
                    img = PIL.Image.open(x["image_path"])
                
                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                        # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                        img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                
                    images.append(img)
                except:
                    pass
            
            # Process inputs based on whether images are present
            if len(images) > 0:
                batch = processing_class(
                    text=prompts_text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
            else:
                batch = processing_class(
                    text=prompts_text,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
            
            # Add labels for training
            # print(batch["input_ids"].shape)
            # print(batch["input_ids"])
            batch["labels"] = batch["input_ids"].clone()
            
            return batch

        # Initialize the Trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
            data_collator=data_collator,
            callbacks=callbacks,
        )

