"""
Baseline experiments — image-only and image + OCR.

The OCR branch is activated automatically when ``config["ocr_enabled"]``
is ``True``.
"""

import os
from base_evaluator import BaseVQAEvaluator


class BaselineExperiment(BaseVQAEvaluator):
    """Baseline VQA: direct question on the document image,
    optionally enriched with sorted OCR text."""

    # -- data extraction ----------------------------------------------------

    def _prepare_item_data(self, item, pages, image_paths):
        ocr_text_dict = None
        if self.config.get("ocr_enabled", False):
            ocr_text_dict = {}
            for page_id in pages:
                page_layout = pages[page_id]["layout_analysis"]
                page_ocr = self.get_sorted_ocr_text(page_layout)
                if page_ocr:
                    image_filename = os.path.basename(page_id)
                    image_path = os.path.join(
                        self.config["images_base_path"], image_filename
                    )
                    ocr_text_dict[image_path] = page_ocr
        return {"ocr_text_dict": ocr_text_dict}

    # -- prompt building ----------------------------------------------------

    def _build_prompt_for_window(
        self, question, item_data, window_paths, window_idx, total_images
    ):
        ocr_text_dict = item_data.get("ocr_text_dict")
        window_ocr = None
        if ocr_text_dict:
            texts = [ocr_text_dict.get(path, "") for path in window_paths]
            joined = "\n\n".join(filter(None, texts))
            window_ocr = joined or None
        return self._create_prompt(question, window_ocr)

    def _create_prompt(self, question, ocr_text=None):
        unable_to_respond_line = (
            "- If uncertain, return 'Unable to determine'\n"
            "- If you can't find the answer, return 'Unable to determine'"
            if self.unable_to_respond_aware
            else ""
        )

        if ocr_text:
            return (
                "You are an AI assistant specialized in analyzing document "
                "images and text. "
                "Your task is to answer questions about the document image "
                "content precisely.\n\n"
                f"For this question, you have the following OCR text:\n"
                f"{ocr_text}\n\n"
                "Guidelines:\n"
                "- Provide concise, focused answers (single word or short "
                "phrase)\n"
                "- Base your answer on both the image and the provided OCR "
                "text\n"
                f"{unable_to_respond_line}\n"
                f"Question: {question}\n"
            )

        return (
            "You are an AI assistant specialized in analyzing document "
            "images. "
            "Your task is to answer questions about the document image "
            "content precisely.\n\n"
            "Guidelines:\n"
            "- Provide concise, focused answers (single word or short "
            "phrase)\n"
            "- Base your answer exclusively on what you see in the image\n"
            f"{unable_to_respond_line}\n"
            f"Question: {question}"
        )

    # -- extra result fields ------------------------------------------------

    def _extra_vqa_fields(self, item_data):
        return {"ocr_enabled": bool(item_data.get("ocr_text_dict"))}
