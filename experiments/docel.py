"""
Document Element (DocEl) experiments — structured OCR with layout labels.

Variants:
  - DocEl                 : structured text only
  - DocEl + CoT v1–v4     : chain-of-thought reasoning steps
  - DocEl + CoT + NumVRE  : adds document structure summary
"""

import os
from base_evaluator import BaseVQAEvaluator


# ---------------------------------------------------------------------------
# Family base — shared data extraction for all DocEl variants
# ---------------------------------------------------------------------------

class _DocElBase(BaseVQAEvaluator):
    """Shared logic for all DocEl experiments."""

    def _prepare_item_data(self, item, pages, image_paths):
        ocr_text_dict = {}
        full_text_list = []

        for page_id in pages:
            image_filename = os.path.basename(page_id)
            image_path = os.path.join(
                self.config["images_base_path"], image_filename
            )
            page_ocr = self.get_structured_ocr_text(
                pages[page_id], image_filename
            )
            ocr_text_dict[image_path] = page_ocr
            full_text_list.append(page_ocr)

        return {
            "ocr_text_dict": ocr_text_dict,
            "full_structured_text": "\n\n".join(full_text_list),
            "layout_summary": self.get_document_layout_summary(pages),
        }

    def _build_prompt_for_window(
        self, question, item_data, window_paths, window_idx, total_images
    ):
        ocr_text_dict = item_data.get("ocr_text_dict", {})
        texts = [ocr_text_dict.get(path, "") for path in window_paths]
        window_structured_text = "\n\n".join(filter(None, texts))
        layout_summary = item_data.get("layout_summary", "")
        return self._create_prompt(
            question, window_structured_text, layout_summary
        )

    def _extra_vqa_fields(self, item_data):
        return {
            "ocr_enabled": True,
            "structured_ocr": item_data.get("full_structured_text", ""),
            "layout_summary": item_data.get("layout_summary", ""),
        }


# ---------------------------------------------------------------------------
# Concrete experiment classes
# ---------------------------------------------------------------------------

class DocElExperiment(_DocElBase):
    """DocEl — structured text with layout labels, no chain-of-thought."""

    def _create_prompt(self, question, structured_text, layout_summary=None):
        unable_to_respond_line = (
            "- If you are not certain or the context does not match, "
            "return 'Unable to determine'."
        )
        prompt = (
            "You are a highly precise AI assistant for document analysis."
            "Your task is to answer questions by meticulously checking "
            "the provided document content.\n\n"
            "The document text is structured with layout labels "
            "(e.g., [Title], [Plain Text], [Endnote]) on a per-page "
            "basis.\n\n"
            "--- DOCUMENT CONTENT ---\n"
            f"{structured_text}\n"
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
            f"Question: {question}"
        )
        return prompt


class DocElCotV1Experiment(_DocElBase):
    """DocEl + CoT v1 — chain-of-thought with document element analysis."""

    def _create_prompt(self, question, structured_text, layout_summary=None):
        prompt = (
            "You are a highly precise AI assistant for document analysis.\n"
            "Your task is to answer questions about the document image "
            "content precisely.\n\n"
            "### DOCUMENT CONTENT ###\n"
            f"{structured_text}\n\n"
            "### GUIDELINES ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "concepts from the question:\n"
            f"  Question: '{question}'\n"
            "   Example entities: dates, names, numbers, organizations, "
            "or key terms.\n\n"
            "2. **Identify and Categorize Document Elements:**\n"
            "    - Examine the document content to identify the distinct "
            "elements present (e.g., [Title], [Subtitle], [Plain Text], "
            "[Table], [Figure], [Endnote], [Header], [Footer]).\n"
            "    - For each page, note the structure and hierarchy of "
            "these elements to understand where key information is likely "
            "located.\n"
            "    - Determine which elements are primary (e.g., [Title], "
            "[Plain Text]) and which are secondary or contextual (e.g., "
            "[Endnote], [Reference]).\n"
            "    - This structural understanding will guide the subsequent "
            "entity matching process.\n"
            "3. **Check for Key Entities in Each Document Element:**\n"
            "   - Search the provided content for matches of these "
            "entities within document elements identified in the point 2\n"
            "   such as [Title], [Plain Text], [Table], [Figure], "
            "[Endnote], etc.\n"
            "   - Note where they appear and in which page or section.\n\n"
            "4. **Check for Consistency Between Document Elements and "
            "Question Context:**\n"
            "   - Evaluate whether the element containing the match "
            "(e.g., [Title], [Plain Text], [Endnote]) aligns with the "
            "question's intent.\n"
            "   - Example: factual answers should come from [Plain Text] "
            "or [Table], not [Endnote] or [Reference].\n"
            "   - If information appears only in secondary elements or "
            "mismatched contexts, treat as invalid.\n\n"
            "5. **Formulate the Answer:**\n"
            "   - If a valid and consistent match exists, provide a "
            "concise factual answer (single word or short phrase).\n"
            "   - If entities are missing or context mismatches, respond "
            "with 'Unable to determine'.\n\n"
            "--- RESPONSE FORMAT ---\n"
            "Return only the final answer.\n"
            "If uncertain, respond exactly with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class DocElCotV2Experiment(_DocElBase):
    """DocEl + CoT v2 — slightly different wording from v1."""

    def _create_prompt(self, question, structured_text, layout_summary=None):
        prompt = (
            "You are a highly precise AI assistant for document analysis.\n"
            "Your task is to answer the question by reasoning carefully "
            "over the structured document content.\n\n"
            "### DOCUMENT CONTENT ###\n"
            f"{structured_text}\n\n"
            "### GUIDELINES ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "concepts from the question "
            "(e.g., dates, names, organizations, key terms).\n"
            f"   Question: '{question}'\n\n"
            "2. **Identify and Categorize Document Elements:**\n"
            "   - Recognize document elements such as [Title], [Subtitle],"
            " [Plain Text], [Table], [Figure], [Endnote], [Header], "
            "[Footer].\n"
            "   - Determine their structure and hierarchy to locate where "
            "key information is likely found.\n"
            "   - Distinguish between primary elements ([Title], "
            "[Plain Text]) and secondary ones ([Endnote], [Reference])."
            "\n\n"
            "3. **Match Entities Within Elements:**\n"
            "   - Search for occurrences of the key entities inside the "
            "identified elements.\n"
            "   - Note where they appear and on which page or section.\n\n"
            "4. **Check Contextual Consistency:**\n"
            "   - Verify that the source element matches the question's "
            "intent.\n"
            "   - Example: factual answers should come from [Plain Text] "
            "or [Table], not from [Endnote] or [Reference].\n"
            "   - Treat mismatched or secondary contexts as invalid.\n\n"
            "5. **Formulate the Answer:**\n"
            "   - If a consistent and valid match exists, provide a "
            "concise answer (single word or short phrase).\n"
            "   - If no valid match is found, respond 'Unable to "
            "determine'.\n\n"
            "--- RESPONSE FORMAT ---\n"
            "Return only the final answer.\n"
            "If uncertain, respond exactly with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class DocElCotV3Experiment(_DocElBase):
    """DocEl + CoT v3 — adds contradiction check between question and
    document content."""

    def _create_prompt(self, question, structured_text, layout_summary=None):
        prompt = (
            "You are a highly precise AI assistant for document analysis.\n"
            "Your task is to answer questions about the document image "
            "content precisely.\n\n"
            "### DOCUMENT CONTENT ###\n"
            f"{structured_text}\n\n"
            "### GUIDELINES ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "concepts from the question:\n"
            f"  Question: '{question}'\n"
            "   Example entities: dates, names, numbers, organizations, "
            "or key terms.\n\n"
            "2. **Identify and Categorize Document Elements:**\n"
            "    - Examine the document content to identify the distinct "
            "elements present (e.g., [Title], [Subtitle], [Plain Text], "
            "[Table], [Figure], [Endnote], [Header], [Footer]).\n"
            "    - For each page, note the structure and hierarchy of "
            "these elements to understand where key information is likely "
            "located.\n"
            "    - Determine which elements are primary (e.g., [Title], "
            "[Plain Text]) and which are secondary or contextual (e.g., "
            "[Endnote], [Reference]).\n"
            "3. **Check for Key Question Entities in Each Document "
            "Element:**\n"
            "   - Search the provided content for matches of these "
            "question entities within document elements identified in "
            "the point 2\n"
            "   - Note where they appear and in which page or section.\n\n"
            "4. **Check for Consistency Between Document Elements and "
            "Question Context:**\n"
            "   - Evaluate whether the element containing the match "
            "(e.g., [Title], [Plain Text], [Endnote]) aligns with the "
            "question's intent.\n"
            "   - Example: factual answers should come from [Plain Text] "
            "or [Table], not [Endnote] or [Reference].\n"
            "   - If information appears only in secondary elements or "
            "mismatched contexts, treat as invalid.\n\n"
            "5. **Check for Contradictions Between Document Elements and "
            "the Question:**\n"
            "   - Examine whether any document element contains "
            "information that directly contradicts the facts or "
            "assumptions stated in the question.\n"
            "   - Examples: if the question implies 'the report was "
            "published in 2011' but the document states 'Report published "
            "in 2010', this is a contradiction.\n"
            "   - If any element presents a clear contradiction or "
            "conflicting information, consider the document unreliable "
            "for answering the question.\n"
            "   - In such cases, respond 'Unable to determine'.\n\n"
            "6. **Formulate the Answer:**\n"
            "   - If a valid and consistent match exists (no "
            "contradictions, correct context), provide a concise factual "
            "answer (single word or short phrase).\n"
            "   - If entities are missing, contexts are mismatched, or "
            "any contradiction is detected, respond 'Unable to "
            "determine'.\n\n"
            "### RESPONSE FORMAT ###\n"
            "Return only the final answer.\n"
            "If uncertain, respond exactly with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class DocElCotV4Experiment(_DocElBase):
    """DocEl + CoT v4 — shorter, more concise version of v3."""

    def _create_prompt(self, question, structured_text, layout_summary=None):
        prompt = (
            "You are a highly precise AI assistant for document analysis.\n"
            "Your task is to answer questions about the document image "
            "content precisely.\n\n"
            "DOCUMENT CONTENT\n"
            f"{structured_text}\n\n"
            "GUIDELINES\n"
            "1. Identify Key Entities: Extract the main entities or "
            "concepts from the question.\n"
            "2. Identify and Categorize Document Elements: Examine the "
            "document content to identify the distinct elements present "
            "(e.g., [Title], [Plain Text], [Table], [Figure], ecc).\n"
            "3. Check for Key Question Entities in Each Document Element: "
            "Look for matches between identified question entities inside "
            "document elements.\n"
            "4. Check for Consistency Between Document Elements and "
            "Question Context: Evaluate whether the element containing "
            "the match (e.g., [Title], [Plain Text], [Endnote]) aligns "
            "with the question's intent. If information appears only in "
            "secondary elements or mismatched contexts, treat as "
            "invalid.\n"
            "5. Check for Contradictions Between Document Elements and "
            "the Question: Examine whether any document element contains "
            "information that directly contradicts the facts or "
            "assumptions stated in the question. If any element presents "
            "a clear contradiction or conflicting information, respond "
            "'Unable to determine'.\n"
            "6. Formulate the Answer: If a valid and consistent match "
            "exists, provide a concise factual answer (single word or "
            "short phrase). If entities are missing, contexts are "
            "mismatched, or any contradiction is detected, respond "
            "'Unable to determine'.\n\n"
            "QUESTION\n"
            f"{question}\n"
            "Final Answer:"
        )
        return prompt


class DocElCotNumVREExperiment(_DocElBase):
    """DocEl + CoT + NumVRE — adds document structure summary with element
    type distribution percentages."""

    def _create_prompt(self, question, structured_text, layout_summary=None):
        prompt = (
            "You are a highly precise AI assistant for document analysis.\n"
            "Your task is to answer questions about the document image "
            "content precisely.\n\n"
            "### DOCUMENT CONTENT ###\n"
            f"{structured_text}\n\n"
            f"{layout_summary}\n\n"
            "### GUIDELINES ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "concepts from the question:\n"
            f"  Question: '{question}'\n"
            "   Example entities: dates, names, numbers, organizations, "
            "or key terms.\n\n"
            "2. **Identify and Categorize Document Elements:**\n"
            "    - Examine the document content to identify the distinct "
            "elements present (e.g., [Title], [Subtitle], [Plain Text], "
            "[Table], [Figure], [Endnote], [Header], [Footer]).\n"
            "    - For each page, note the structure and hierarchy of "
            "these elements to understand where key information is likely "
            "located.\n"
            "    - Determine which elements are primary (e.g., [Title], "
            "[Plain Text]) and which are secondary or contextual (e.g., "
            "[Endnote], [Reference]).\n\n"
            "3. **Evaluate Document Composition:**\n"
            "    - Use the document structure summary above to assess the "
            "overall composition of the document.\n"
            "    - Elements with higher frequency (e.g., many [Plain Text]"
            " or [Table]) indicate the main source of factual content.\n"
            "    - If a page is dominated by secondary elements (e.g., "
            "[Footer], [Endnote]), treat information from it as less "
            "reliable.\n"
            "    - Consider the distribution percentages before deciding "
            "where the most relevant evidence likely resides.\n\n"
            "4. **Check for Key Question Entities in Each Document "
            "Element:**\n"
            "   - Search the provided content for matches of these "
            "question entities within the identified elements.\n"
            "   - Note where they appear and in which page or section.\n\n"
            "5. **Check for Consistency Between Document Elements and "
            "Question Context:**\n"
            "   - Evaluate whether the element containing the match "
            "(e.g., [Title], [Plain Text], [Endnote]) aligns with the "
            "question\u2019s intent.\n"
            "   - Example: factual answers should come from [Plain Text] "
            "or [Table], not [Endnote] or [Reference].\n"
            "   - If information appears only in secondary elements or "
            "mismatched contexts, treat as invalid.\n\n"
            "6. **Check for Contradictions Between Document Elements and "
            "the Question:**\n"
            "   - Examine whether any document element contains "
            "information that directly contradicts the facts or "
            "assumptions stated in the question.\n"
            "   - Examples: if the question implies 'the report was "
            "published in 2011' but the document states 'Report published "
            "in 2010', this is a contradiction.\n"
            "   - If any element presents a clear contradiction or "
            "conflicting information, consider the document unreliable "
            "for answering the question.\n"
            "   - In such cases, respond 'Unable to determine'.\n\n"
            "7. **Formulate the Answer:**\n"
            "   - If a valid and consistent match exists (no "
            "contradictions, correct context), provide a concise factual "
            "answer (single word or short phrase).\n"
            "   - If entities are missing, contexts are mismatched, "
            "structure is ambiguous, or any contradiction is detected, "
            "respond 'Unable to determine'.\n\n"
            "### RESPONSE FORMAT ###\n"
            "Return only the final answer.\n"
            "If uncertain, respond exactly with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt
