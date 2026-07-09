"""
Layout experiments — spatial/positional reasoning on document pages.

Variants:
  - Layout v1 : basic quadrant analysis
  - Layout v2 : adds layout heuristics
  - Layout v3 : adds contradiction check + document length awareness
  - Layout v4 : explicit page_info (``page X/Y``) in prompt
"""

from base_evaluator import BaseVQAEvaluator


# ---------------------------------------------------------------------------
# Family base — shared data extraction
# ---------------------------------------------------------------------------

class _LayoutBase(BaseVQAEvaluator):
    """Shared logic for all Layout experiments.
    Layout experiments focus on visual/spatial reasoning — no OCR text is
    passed to the prompt (only images)."""

    def _prepare_item_data(self, item, pages, image_paths):
        # Layout experiments don't extract textual features
        return {}

    def _build_prompt_for_window(
        self, question, item_data, window_paths, window_idx, total_images
    ):
        # Compute page_info for the current window
        # (used by v3/v4; ignored by v1/v2)
        current_page = window_idx + 1
        page_info = f"page {current_page}/{total_images}"
        return self._create_prompt(question, page_info)

    def _extra_vqa_fields(self, item_data):
        return {"ocr_enabled": False}


# ---------------------------------------------------------------------------
# Concrete experiment classes
# ---------------------------------------------------------------------------

class LayoutV1Experiment(_LayoutBase):
    """Layout v1 — basic quadrant analysis with cross-page verification."""

    def _create_prompt(self, question, page_info=None):
        prompt = (
            "You are a highly precise AI assistant for document layout "
            "analysis.\n"
            "Your task is to answer questions about the visual and "
            "structural content of document images.\n\n"
            "### LAYOUT VERIFICATION STEPS ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "keywords from the question.\n"
            f"    Question: '{question}'\n\n"
            "2. **Analyze Spatial Positioning (In-Page and Cross-Page):**\n"
            "   - Divide each page into four quadrants:\n"
            "     * Top-left quarter\n"
            "     * Top-right quarter\n"
            "     * Bottom-left quarter\n"
            "     * Bottom-right quarter\n"
            "   - For each page, determine where the relevant entity or "
            "layout feature is likely located.\n"
            "   - In-page analysis: check whether the entity appears "
            "consistently in the same area within the page "
            "(e.g., always near the top or bottom).\n\n"
            "3. **Cross-Page Verification (only if multiple pages are "
            "provided):**\n"
            "   - If more than one page is available, compare the spatial "
            "distribution of the same entities across pages.\n"
            "   - Confirm that the positions are coherent across pages "
            "(e.g., all appear in top-left areas).\n"
            "   - If positions vary significantly between pages, or "
            "expected layout consistency is missing, treat the cross-page "
            "evidence as unreliable.\n"
            "   - If only one page is provided, skip cross-page "
            "verification entirely.\n\n"
            "4. **Formulate Spatially-Consistent Answer:**\n"
            "   - If the spatial pattern is consistent and aligns with "
            "the question type, provide a concise factual answer.\n"
            "   - If the pattern is inconsistent, ambiguous, or not "
            "aligned with the expected regions, respond 'Unable to "
            "determine'.\n\n"
            "### RESPONSE FORMAT ###\n"
            "Return only the final verified answer.\n"
            "If uncertain, inconsistent, or unsupported by layout "
            "reasoning, respond exactly with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class LayoutV2Experiment(_LayoutBase):
    """Layout v2 — more detailed spatial analysis with layout heuristics."""

    def _create_prompt(self, question, page_info=None):
        prompt = (
            "You are a highly precise AI assistant for document layout "
            "analysis.\n"
            "Your task is to reason spatially about the structure and "
            "layout of document pages to answer the question.\n"
            "Do not infer from text meaning \u2014 focus only on visual "
            "and positional reasoning.\n\n"
            "### LAYOUT REASONING STEPS ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "visual cues implied by the question.\n"
            f"   Question: '{question}'\n\n"
            "2. **In-Page Spatial Analysis:**\n"
            "   - Divide each page into four regions:\n"
            "     [Q1] Top-Left | [Q2] Top-Right | [Q3] Bottom-Left | "
            "[Q4] Bottom-Right\n"
            "   - Locate where each relevant entity or element appears "
            "within its page.\n"
            "   - Assess spatial coherence: does the entity consistently "
            "appear in a specific region (top, bottom, etc.)?\n\n"
            "3. **Cross-Page Verification:**\n"
            "   - If multiple pages exist, compare spatial zones across "
            "them.\n"
            "   - Confirm consistent positioning of similar entities "
            "across pages (e.g., always near top-left).\n"
            "   - Treat inconsistent spatial locations as unreliable "
            "evidence.\n\n"
            "4. **Layout Heuristics (for reasoning support):**\n"
            "   - Titles and headers usually appear in the upper "
            "regions.\n"
            "   - Tables and core facts often appear mid or bottom-left.\n"
            "   - References and footnotes are typically bottom-right.\n\n"
            "5. **Formulate Spatially-Consistent Answer:**\n"
            "   - If entities appear consistently in the expected regions,"
            " provide a concise factual answer.\n"
            "   - If layout evidence is ambiguous, inconsistent, or "
            "unsupported, respond 'Unable to determine'.\n\n"
            "### RESPONSE FORMAT ###\n"
            "Return only the final verified answer.\n"
            "If uncertain, inconsistent, or unsupported by layout "
            "reasoning, respond exactly with 'Unable to determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class LayoutV3Experiment(_LayoutBase):
    """Layout v3 — adds contradiction/inconsistency check and document
    length awareness."""

    def _create_prompt(self, question, page_info=None):
        prompt = (
            "You are a highly precise AI assistant for document layout "
            "analysis.\n"
            "Your task is to reason spatially about the structure and "
            "layout of document pages to answer the question.\n"
            "Do not infer from text meaning \u2014 focus only on visual "
            "and positional reasoning.\n\n"
            "### LAYOUT REASONING STEPS ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "visual cues implied by the question.\n"
            f"   Question: '{question}'\n\n"
            "2. **In-Page Spatial Analysis:**\n"
            "   - Divide each page into four regions:\n"
            "     [Q1] Top-Left | [Q2] Top-Right | [Q3] Bottom-Left | "
            "[Q4] Bottom-Right\n"
            "   - Locate where each relevant entity or element appears "
            "within its page.\n"
            "   - Assess spatial coherence: does the entity consistently "
            "appear in a specific region (top, bottom, etc.)?\n\n"
            "3. **Cross-Page Verification:**\n"
            "   - If multiple pages exist, compare spatial zones across "
            "them.\n"
            "   - **Consider document length:** the more pages a document "
            "has, the higher your confidence threshold must be.\n"
            "     Only provide an answer after checking most or all "
            "pages.\n\n"
            "4. **Layout Heuristics (for reasoning support):**\n"
            "   - Titles and headers usually appear in the upper "
            "regions.\n"
            "   - Tables and core facts often appear mid or bottom-left.\n"
            "   - References and footnotes are typically bottom-right.\n\n"
            "5. **Contradiction and Inconsistency Check:**\n"
            "   - If any spatial evidence or layout pattern directly "
            "contradicts or disproves what is implied by the question, "
            "treat the document as unreliable for answering.\n"
            "   - In such cases, respond with 'Unable to determine'.\n\n"
            "6. **Formulate Spatially-Consistent Answer:**\n"
            "   - If entities appear consistently in the expected regions,"
            " provide a concise factual answer.\n"
            "   - For multi-page documents, answer only when spatial "
            "evidence is strong and repeated across pages.\n"
            "   - If layout evidence is ambiguous, inconsistent, or "
            "contradictory (especially in long documents), respond "
            "'Unable to determine'.\n\n"
            "### RESPONSE FORMAT ###\n"
            "Return only the final verified answer.\n"
            "If uncertain, inconsistent, contradictory, or unsupported "
            "by layout reasoning, respond exactly with 'Unable to "
            "determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt


class LayoutV4Experiment(_LayoutBase):
    """Layout v4 — explicit page position info (``page X/Y``) in prompt."""

    def _create_prompt(self, question, page_info=None):
        prompt = (
            "You are a highly precise AI assistant for document layout "
            "analysis.\n"
            "Your task is to reason spatially about the structure and "
            "layout of document pages to answer the question.\n"
            "Do not infer from text meaning \u2014 focus only on visual "
            "and positional reasoning.\n\n"
            "### LAYOUT REASONING STEPS ###\n"
            "1. **Identify Key Entities:** Extract the main entities or "
            "visual cues implied by the question.\n"
            f"   Question: '{question}'\n\n"
            "2. **In-Page Spatial Analysis:**\n"
            "   - Divide each page into four regions:\n"
            "     [Q1] Top-Left | [Q2] Top-Right | [Q3] Bottom-Left | "
            "[Q4] Bottom-Right\n"
            "   - Locate where each relevant entity or element appears "
            "within its page.\n"
            "   - Assess spatial coherence: does the entity consistently "
            "appear in a specific region (top, bottom, etc.)?\n\n"
            "3. **Cross-Page Verification:**\n"
            "   - If multiple pages exist, compare spatial zones across "
            "them.\n"
            f"   - **Consider document length: {page_info}** Only provide "
            "an answer after checking all pages.\n\n"
            "4. **Layout Heuristics (for reasoning support):**\n"
            "   - Titles and headers usually appear in the upper "
            "regions.\n"
            "   - Tables and core facts often appear mid or bottom-left.\n"
            "   - References and footnotes are typically bottom-right.\n\n"
            "5. **Contradiction and Inconsistency Check:**\n"
            "   - If any spatial evidence or layout pattern directly "
            "contradicts or disproves what is implied by the question, "
            "treat the document as unreliable for answering.\n"
            "   - In such cases, respond with 'Unable to determine'.\n\n"
            "6. **Formulate Spatially-Consistent Answer:**\n"
            "   - If entities appear consistently in the expected regions,"
            " provide a concise factual answer.\n"
            "   - For multi-page documents, answer only when spatial "
            "evidence is strong and repeated across pages.\n"
            "   - If layout evidence is ambiguous, inconsistent, or "
            "contradictory (especially in long documents), respond "
            "'Unable to determine'.\n\n"
            "### RESPONSE FORMAT ###\n"
            "Return only the final verified answer.\n"
            "If uncertain, inconsistent, contradictory, or unsupported "
            "by layout reasoning, respond exactly with 'Unable to "
            "determine'.\n\n"
            f"Question: {question}\n"
            "Final Answer:"
        )
        return prompt
