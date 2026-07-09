"""
NLP Tag experiments — OCR text with inline NLP entity tags.

Variants:
  - NLP Tag       : tagged OCR with entity-aware prompt
  - NLP Tag + CoT : adds chain-of-thought reasoning steps
"""

import os
from base_evaluator import BaseVQAEvaluator


# ---------------------------------------------------------------------------
# Family base — shared data extraction
# ---------------------------------------------------------------------------

class _NLPTagBase(BaseVQAEvaluator):
    """Shared logic for NLP-tagged OCR experiments."""

    def _prepare_item_data(self, item, pages, image_paths):
        entities = item.get("corrupted_entities", [])
        entity_types = item.get("entity_type", [])
        patch_entities = item.get("patch_entities", {})

        # Build NLP-tagged OCR text (page by page)
        nlp_tag_text_parts = []
        for page_id, page_data in pages.items():
            layout = page_data.get("layout_analysis", {})
            patches = patch_entities.get(page_id, {})
            annotated_ocr = self.get_nlp_tagged_ocr_text(layout, patches)
            nlp_tag_text_parts.append(
                f"--- Page: {os.path.basename(page_id)} ---\n"
                f"{annotated_ocr}"
            )

        return {
            "nlp_tag_text": "\n".join(nlp_tag_text_parts),
            "entities": entities,
            "entity_types": entity_types,
        }

    def _build_prompt_for_window(
        self, question, item_data, window_paths, window_idx, total_images
    ):
        return self._create_prompt(
            question,
            item_data["nlp_tag_text"],
            item_data.get("entities"),
            item_data.get("entity_types"),
        )

    def _extra_vqa_fields(self, item_data):
        return {"ocr_enabled": bool(item_data.get("nlp_tag_text"))}


# ---------------------------------------------------------------------------
# Concrete experiment classes
# ---------------------------------------------------------------------------

class NLPTagExperiment(_NLPTagBase):
    """NLP Tagged OCR — entity-annotated OCR with context verification."""

    def _create_prompt(
        self, question, NLP_tag_text, entities=None, entity_types=None
    ):
        annotated_question = self.annotate_question_with_entities(
            question, entities or [], entity_types or []
        )

        unable_to_respond_line = (
            "- If you are not certain or the context does not match, "
            "return 'Unable to determine'."
        )
        prompt = (
            "You are a highly precise AI assistant for document analysis."
            "Your task is to answer questions by meticulously checking "
            "the provided document content.\n\n"
            "The document text is structured with OCR text where entities "
            "are annotated using NLP tags.\n\n"
            "--- DOCUMENT CONTENT ---\n"
            f"{NLP_tag_text}\n"
            "--- END OF DOCUMENT CONTENT ---\n\n"
            "Guidelines:\n"
            "- Provide a concise answer (a single word or short phrase).\n"
            "- Your answer MUST be based on the information provided "
            "above.\n"
            "- **Crucially, verify the context. An entity (e.g., a year, "
            "a name) found in a different section or page than the one "
            "implied by the question makes the question unanswerable "
            "from that context.**\n"
            "- For example, if the question asks about a main fact, an "
            "answer found only in an [Endnote] or on a different page "
            "is NOT valid.\n"
            f"{unable_to_respond_line}\n\n"
            f"Question: {annotated_question}"
        )
        return prompt


class NLPTagCotExperiment(_NLPTagBase):
    """NLP Tagged OCR + CoT — adds step-by-step reasoning over tagged
    entities."""

    def _create_prompt(
        self, question, NLP_tag_text, entities=None, entity_types=None
    ):
        annotated_question = self.annotate_question_with_entities(
            question, entities or [], entity_types or []
        )

        prompt = (
            "You are a highly precise AI assistant for document "
            "analysis.\n"
            "Your task is to answer questions by carefully reasoning on "
            "the provided OCR text.\n\n"
            "The OCR text has been enriched with NLP entity tags (e.g., "
            "<year_numerical_value>, <time_information>, <event>), "
            "so that important elements are explicitly marked.\n\n"
            "--- DOCUMENT CONTENT WITH TAGGED ENTITIES ---\n"
            f"{NLP_tag_text}\n"
            "--- END OF DOCUMENT CONTENT ---\n\n"
            "Analyze the user's question by following these steps:\n"
            f"1. **Understand the Question:** Break down the question: "
            f"'{annotated_question}' into its key elements.\n"
            "2. **Identify Tagged Entities:** Which tagged entities "
            "(<...>...</...>) from the OCR are relevant to this "
            "question?\n"
            "3. **Locate Evidence in OCR:** Find where those entities "
            "appear in the document text provided above.\n"
            "4. **Verify Context:** Check if the context around the "
            "entities (the sentence, section, or page) is consistent "
            "with what the question is asking.\n"
            "5. **Resolve Ambiguities:** If multiple matches exist, "
            "choose the one that best fits the context implied by the "
            "question.\n"
            "6. **Formulate Final Answer:**\n"
            "   - If you find a clear, contextually correct match, "
            "provide the concise answer (a single word or short "
            "phrase).\n"
            "   - If the entities appear but only in mismatched context, "
            "respond with 'Unable to determine'.\n"
            "   - If the entities do not appear at all, respond with "
            "'Unable to determine'.\n\n"
            "Final Answer:"
        )
        return prompt
