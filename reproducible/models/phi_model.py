"""
Phi-3.5 Vision model backend — HuggingFace transformers.

Reproduces the inference logic from ``vqa-phi.ipynb``:
  - FP16 precision
  - Phi-3 specific ``<|image_N|>`` / ``<|user|>`` / ``<|assistant|>`` tokens
  - ``torch.inference_mode`` for speed
"""

import gc
import torch
from PIL import Image

from models.base_model import ModelBackend


class PhiBackend(ModelBackend):
    """Backend for ``microsoft/Phi-3.5-vision-instruct``."""

    DEFAULT_MODEL = "microsoft/Phi-3.5-vision-instruct"

    def __init__(self, model_name=None, max_tokens=1024, cache_dir=None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir or "/data1/hf_cache/models"
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------
    def setup(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        print(f"Initializing Phi model: {self.model_name} ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
            device_map="auto",
            attn_implementation="eager",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        print("Phi model initialized successfully.\n")

    # ------------------------------------------------------------------
    def infer(self, prompt, image_paths):
        device = next(self.model.parameters()).device

        user_prompt = "<|user|>\n"
        assistant_prompt = "<|assistant|>\n"
        prompt_suffix = "<|end|>\n"

        # Build Phi-3 specific multi-image prompt
        if len(image_paths) == 1:
            full_prompt = (
                f"{user_prompt}<|image_1|>\n {prompt}"
                f"{prompt_suffix}{assistant_prompt}"
            )
            images = Image.open(image_paths[0])
        elif len(image_paths) == 2:
            full_prompt = (
                f"{user_prompt}<|image_1|>\n<|image_2|>\n {prompt}"
                f"{prompt_suffix}{assistant_prompt}"
            )
            images = [Image.open(p) for p in image_paths]
        else:
            image_tags = "".join(
                f"<|image_{i+1}|>\n" for i in range(len(image_paths))
            )
            full_prompt = (
                f"{user_prompt}{image_tags} {prompt}"
                f"{prompt_suffix}{assistant_prompt}"
            )
            images = [Image.open(p) for p in image_paths]

        with torch.inference_mode():
            inputs = self.processor(
                full_prompt, images, return_tensors="pt"
            ).to(device)

            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        del inputs
        del generate_ids
        torch.cuda.empty_cache()

        return response

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
        print("Phi model cleaned up.")
