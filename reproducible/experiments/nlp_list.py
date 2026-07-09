"""
NLP Entity List experiments — extracted entity lists for question and document.

Variants:
  - NLP List             : entity lists only
  - NLP List + CoT       : adds chain-of-thought
  - NLP List + OCR       : adds raw OCR text + contradiction check
  - NLP List + OCR + CoT : full pipeline with reasoning steps
"""

import os
from base_evaluator import BaseVQAEvaluator


# ---------------------------------------------------------------------------
# Family base — shared data extraction
# ---------------------------------------------------------------------------

class _NLPListBase(BaseVQAEvaluator):
    """Shared logic for NLP entity list experiments."""

    def _prepare_item_data(self, item, pages, image_paths):
        question_entities, doc_entities_dict = self.extract_entities_from_item(
            item, self.config["images_base_path"]
        )

        # Build OCR text dict (for variants that use it)
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

        return {
            "question_entities": question_entities,
            "doc_entities_dict": doc_entities_dict,
            "ocr_text_dict": ocr_text_dict,
        }

    def _build_prompt_for_window(
        self, question, item_data, window_paths, window_idx, total_images
    ):
        # Slice entities for this window
        doc_entities_dict = item_data.get("doc_entities_dict", {})
        window_doc_entities = []
        for path in window_paths:
            window_doc_entities.extend(doc_entities_dict.get(path, []))

        # Slice OCR for this window
        ocr_text_dict = item_data.get("ocr_text_dict")
        window_ocr = None
        if ocr_text_dict:
            texts = [ocr_text_dict.get(path, "") for path in window_paths]
            joined = "\n\n".join(filter(None, texts))
            window_ocr = joined or None

        return self._create_prompt(
            question,
            item_data["question_entities"],
            window_doc_entities,
            window_ocr,
        )

    def _extra_vqa_fields(self, item_data):
        return {
            "ocr_enabled": bool(item_data.get("ocr_text_dict")),
            "question_entities": item_data.get("question_entities", []),
            "doc_entities": item_data.get("doc_entities_dict", {}),
        }


# ---------------------------------------------------------------------------
# Concrete experiment classes
# ---------------------------------------------------------------------------

class NLPListExperiment(_NLPListBase):
    """NLP Entity List — match question entities against document entities."""

    def _create_prompt(
        self, question, question_entities, doc_entities, ocr_text=None
    ):
        unable_to_respond_line = (
            "- If uncertain, return 'Unable to determine'."
        )
        prompt = (
            "You are a precise AI assistant for document analysis.\n\n"
            "We provide you with two sets of annotated entities:\n"
            f"- Entities in the question: {question_entities}\n"
            f"- Entities in the document: {doc_entities}\n\n"
            "Guidelines:\n"
            "- Provide a concise answer (a single word or short phrase).\n"
            "- Match question entities against document entities.\n"
            "- If context does not align or entities are missing, answer "
            "'Unable to determine'.\n\n"
            f"{unable_to_respond_line}\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class NLPListCotExperiment(_NLPListBase):
    """NLP Entity List + CoT — adds chain-of-thought reasoning."""

    def _create_prompt(
        self, question, question_entities, doc_entities, ocr_text=None
    ):
        unable_to_respond_line = (
            "- If uncertain or context does not align, respond with "
            "'Unable to determine'."
        )
        prompt = (
            "You are a highly precise AI assistant specialized in "
            "document analysis.\n"
            "Your task is to analyze the question and the document "
            "entities to determine the correct answer.\n\n"
            "We provide you with two sets of annotated entities:\n"
            f"- Entities in the question: {question_entities}\n"
            f"- Entities in the document: {doc_entities}\n\n"
            "Follow these reasoning steps carefully before giving your "
            "final answer:\n"
            "1. **Identify Key Entities:** Examine the question and list "
            "all entities explicitly mentioned or implied.\n"
            "2. **Locate Matches:** Search for corresponding entities "
            "within the document entities that share the same type or "
            "meaning.\n"
            "3. **Check Context:** Evaluate whether the matching entities "
            "are found in a consistent context \u2014 same page, same "
            "semantic meaning, or same type of information (e.g., dates, "
            "organizations, people, locations).\n"
            "4. **Resolve Ambiguities:** If multiple possible matches "
            "exist, choose the most contextually relevant one.\n"
            "5. **Determine Validity:** If no valid match exists, or if "
            "the context is contradictory, the answer is 'Unable to "
            "determine'.\n"
            "6. **Provide the Final Answer:** Summarize your conclusion "
            "in a concise manner (a single word or short phrase).\n\n"
            "Guidelines:\n"
            "- Be concise and factual.\n"
            "- Base your reasoning only on the provided entities.\n"
            "- Do not invent or assume missing information.\n"
            f"{unable_to_respond_line}\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class NLPListOCRExperiment(_NLPListBase):
    """NLP Entity List + OCR — adds raw OCR text and contradiction check."""

    def _create_prompt(
        self, question, question_entities, doc_entities, ocr_text=None
    ):
        unable_to_respond_line = (
            "- If uncertain, return 'Unable to determine'."
        )
        prompt = (
            "You are a precise AI assistant for document analysis.\n\n"
            "We provide you with two sets of annotated entities and the "
            "document OCR:\n"
            f"- Entities in the question: {question_entities}\n"
            f"- Entities in the document: {doc_entities}\n"
            f"- Document OCR: {ocr_text}\n\n"
            "Guidelines:\n"
            "- Provide a concise answer (a single word or short phrase).\n"
            "- Match question entities against document entities.\n"
            "- If entities are missing, contexts are mismatched, or any "
            "contradiction is detected, respond 'Unable to determine'.\n\n"
            f"{unable_to_respond_line}\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class NLPListOCRCotExperiment(_NLPListBase):
    """NLP Entity List + OCR + CoT — full pipeline with chain-of-thought
    reasoning and contradiction detection."""

    def _create_prompt(
        self, question, question_entities, doc_entities, ocr_text=None
    ):
        prompt = (
            "You are a highly precise AI assistant specialized in "
            "document analysis.\n"
            "Your task is to analyze the question and the document "
            "entities to determine the correct answer.\n\n"
            "We provide you with two sets of annotated entities and the "
            "document OCR:\n"
            f"- Entities in the question: {question_entities}\n"
            f"- Entities in the document: {doc_entities}\n"
            f"- Document OCR: {ocr_text}\n\n"
            "Follow these reasoning steps carefully before giving your "
            "final answer:\n"
            "1. **Identify Key Entities:** Examine the question and list "
            "all entities explicitly mentioned or implied.\n"
            "2. **Locate Matches:** Search for corresponding entities "
            "within the document entities that share the same type or "
            "meaning.\n"
            "3. **Check Context:** Evaluate whether the matching entities "
            "are found in a consistent context \u2014 same page, same "
            "semantic meaning, or same type of information (e.g., dates, "
            "organizations, people, locations).\n"
            "4. **Resolve Ambiguities:** If multiple possible matches "
            "exist, choose the most contextually relevant one.\n"
            "5. **Check for Contradictions:** Determine if any document "
            "entity conflicts with or disproves information stated or "
            "implied in the question. "
            "If a contradiction is found, consider the document "
            "unreliable for answering.\n"
            "6. **Determine Validity:** If no valid match exists, or if "
            "the context or logic is contradictory, the answer must be "
            "'Unable to determine'.\n"
            "7. **Provide the Final Answer:** Summarize your conclusion "
            "in a concise manner (a single word or short phrase).\n\n"
            "Guidelines:\n"
            "- Be concise and factual.\n"
            "- Base your reasoning only on the provided entities.\n"
            "- Do not invent or assume missing information.\n"
            "- If uncertain, contradictory, or context does not align, "
            "respond with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt
