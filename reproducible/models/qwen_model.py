"""
Qwen 2.5 VL model backend — HuggingFace transformers.

Reproduces the inference logic from ``vqa-qwen-2.ipynb``:
  - 8-bit quantisation via BitsAndBytesConfig
  - ``Qwen2_5_VLForConditionalGeneration`` + ``process_vision_info``
  - Chat-template based prompting
"""

import gc
import torch

from models.base_model import ModelBackend


class QwenBackend(ModelBackend):
    """Backend for ``Qwen/Qwen2.5-VL-3B-Instruct``."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, model_name=None, max_tokens=1024, cache_dir=None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir or "/data1/hf_cache/models"
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------
    def setup(self):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        print(f"Initializing Qwen model: {self.model_name} ...")

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map="auto",
            quantization_config=quantization_config,
        ).eval()

        print("Qwen model initialized successfully with 8-bit quantization.\n")

    # ------------------------------------------------------------------
    def infer(self, prompt, image_paths):
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": f"file://{path}"}
                        for path in image_paths
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs
        del generated_ids
        torch.cuda.empty_cache()
        gc.collect()

        return response.strip()

    # ------------------------------------------------------------------
    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        gc.collect()
        print("Qwen model cleaned up.")
