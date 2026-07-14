"""
Base VQA Evaluator — shared infrastructure for all experiments.

Subclasses must override:
  - _create_prompt()
  - _prepare_item_data()
  - _build_prompt_for_window()

Optionally override:
  - _extra_vqa_fields()

The model backend (Ollama, Phi, Qwen, …) is injected via the
``model_backend`` parameter — see ``models/`` for implementations.
"""

import json
import os
import re
import random
import datetime
import traceback
from tqdm.auto import tqdm
from collections import Counter


# ---------------------------------------------------------------------------
# Base evaluator class
# ---------------------------------------------------------------------------

class BaseVQAEvaluator:
    """Base class for all VQA experiment evaluators."""

    def __init__(self, config_path, model_backend):
        """
        Parameters
        ----------
        config_path : str
            Path to the JSON configuration file.
        model_backend : models.base_model.ModelBackend
            An initialised backend that implements ``infer(prompt, images)``.
        """
        with open(config_path) as f:
            self.config = json.load(f)

        self.model_backend = model_backend

        # model_config is still used for windowing logic (batch_size / stride)
        self.model_config = {
            "name": getattr(model_backend, "model_name", "unknown"),
            "batch_size": 1,
            "stride": 1,
        }
        self.target_model = self.model_config["name"]

        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get(
            "unable_to_respond_aware", True
        )

        print(f"Initialized evaluator for model: {self.target_model}")
        print(f"Experiment class: {self.__class__.__name__}")

    # -----------------------------------------------------------------------
    # Utility methods — available to all subclasses
    # -----------------------------------------------------------------------

    def get_sorted_ocr_text(self, layout_analysis):
        """Basic OCR extraction sorted by spatial position (top-to-bottom,
        left-to-right).  Returns plain text."""
        ocr_items = []
        for obj in layout_analysis.values():
            if isinstance(obj, dict) and "OCR" in obj and "BBOX" in obj:
                bbox = obj["BBOX"]
                ocr_items.append((bbox[1], bbox[0], obj["OCR"]))
        ocr_items.sort()
        return "\n".join(item[2] for item in ocr_items)

    def get_structured_ocr_text(self, single_page_data, image_filename):
        """Extract OCR with layout-element labels ([Title], [Plain Text], …)
        for a single page."""
        full_structured_text = [f"--- Page: {image_filename} ---"]
        layout_objects = single_page_data.get("layout_analysis", {})

        sorted_objects = sorted(
            layout_objects.values(),
            key=lambda obj: (
                obj.get("BBOX", [0, 0])[1],
                obj.get("BBOX", [0, 0])[0],
            ),
        )

        for obj in sorted_objects:
            obj_type = obj.get("ObjectType", "unknown").replace("_", " ").title()
            ocr_text = obj.get("OCR", "").strip()
            if ocr_text:
                full_structured_text.append(f"[{obj_type}]: {ocr_text}")

        return "\n".join(full_structured_text)

    def get_document_layout_summary(self, pages):
        """Generate a per-page summary of layout-element type counts and
        percentages (used by DocEl + NumVRE experiments)."""
        if not pages:
            return "No layout information available."

        summary_lines = []
        for page_id, page_data in pages.items():
            layout_objs = page_data.get("layout_analysis", {})
            counter = Counter()
            for obj in layout_objs.values():
                obj_type = obj.get("ObjectType", "Unknown")
                counter[obj_type] += 1

            total_objects = sum(counter.values()) or 1
            formatted_items = [
                f"{obj_type}: {count} ({(count / total_objects) * 100:.1f}%)"
                for obj_type, count in sorted(counter.items())
            ]
            summary_lines.append(
                f"- {os.path.basename(page_id)} \u2192 {', '.join(formatted_items)}"
            )

        return (
            "### DOCUMENT STRUCTURE SUMMARY ###\n"
            "Below is the distribution of visual Document element types "
            "detected in the document pages:\n"
            + "\n".join(summary_lines)
        )

    def get_nlp_tagged_ocr_text(self, layout_analysis, patch_entities=None):
        """OCR sorted by position with NLP entity tags inserted inline
        (e.g. ``<year_numerical_value>2011</year_numerical_value>``)."""
        ocr_items = []
        for obj_id, obj in layout_analysis.items():
            if isinstance(obj, dict) and "OCR" in obj and "BBOX" in obj:
                bbox = obj["BBOX"]
                ocr_items.append((bbox[1], bbox[0], obj_id, obj["OCR"]))
        ocr_items.sort()

        annotated_lines = []
        for _, _, obj_id, ocr_text in ocr_items:
            text = ocr_text or ""
            entities = []
            if patch_entities:
                entities = (
                    patch_entities.get(obj_id, {}).get("entities", []) or []
                )

            if entities:
                for ent in sorted(
                    entities,
                    key=lambda e: e.get("start", 0),
                    reverse=True,
                ):
                    ent_text = ent.get("text", "")
                    start = ent.get("start")
                    end = ent.get("end")
                    tag = ent.get("label", "entity").lower()

                    applied = False
                    if (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start < end <= len(text)
                    ):
                        substr = text[start:end]
                        if (
                            substr.strip().lower() == ent_text.strip().lower()
                            or ent_text.strip().lower()
                            in substr.strip().lower()
                        ):
                            text = (
                                text[:start]
                                + f"<{tag}>{text[start:end]}</{tag}>"
                                + text[end:]
                            )
                            applied = True

                    if not applied and ent_text:
                        idx = text.lower().find(ent_text.strip().lower())
                        if idx != -1:
                            length = len(ent_text.strip())
                            text = (
                                text[:idx]
                                + f"<{tag}>{text[idx:idx+length]}</{tag}>"
                                + text[idx + length :]
                            )

            annotated_lines.append(text)

        return "\n".join(annotated_lines)

    def extract_entities_from_item(self, item, images_base_path):
        """Extract NLP entities for question and document pages.

        Returns
        -------
        question_entities : list[str]
            Flat list of tagged question entities.
        doc_entities_dict : dict[str, list[str]]
            Mapping ``image_path -> list of tagged document entities``.
        """
        question_entities = [
            f"<{etype}>{ent.get('text', '')}</{etype}>"
            for ent, etype in zip(
                item.get("corrupted_entities", []),
                item.get("entity_type", []),
            )
            if ent.get("text", "")
        ]

        doc_entities_dict = {}
        for page_id, page_entities in item.get("patch_entities", {}).items():
            image_filename = os.path.basename(page_id)
            image_path = os.path.join(images_base_path, image_filename)

            page_entity_list = []
            for obj_id, obj_data in page_entities.items():
                for ent in obj_data.get("entities", []):
                    label = ent.get("label", "entity").lower()
                    text = ent.get("text", "")
                    if text:
                        page_entity_list.append(f"<{label}>{text}</{label}>")

            doc_entities_dict[image_path] = page_entity_list

        return question_entities, doc_entities_dict

    def annotate_question_with_entities(self, question, entities, entity_types):
        """Insert NLP-entity tags into the question text around matching
        substrings (case-insensitive, longest-match-first)."""
        if not question:
            return question

        pairs = []
        if entity_types and entities:
            for ent, etype in zip(entities, entity_types):
                text = (
                    ent.get("text", "") if isinstance(ent, dict) else str(ent)
                )
                if text:
                    pairs.append((text, etype))
        else:
            for ent in entities or []:
                if isinstance(ent, dict):
                    text = ent.get("text", "")
                    tag = ent.get("label", "entity")
                else:
                    text = str(ent)
                    tag = "entity"
                if text:
                    pairs.append((text, tag))

        pairs = sorted(pairs, key=lambda x: len(x[0]), reverse=True)

        out = question
        for text, tag in pairs:
            if not text:
                continue
            pattern = re.compile(re.escape(text), flags=re.IGNORECASE)

            def _repl(m, _tag=tag):
                return f"<{_tag.lower()}>{m.group(0)}</{_tag.lower()}>"

            out, _ = pattern.subn(_repl, out, count=1)
        return out

    # -----------------------------------------------------------------------
    # Abstract / hook methods — override in subclasses
    # -----------------------------------------------------------------------

    def _create_prompt(self, question, **kwargs):
        """Return the prompt string for a single inference call.
        Must be overridden by every experiment subclass."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _create_prompt()"
        )

    def _prepare_item_data(self, item, pages, image_paths):
        """Extract experiment-specific features from a dataset item.

        Returns a dict that will be forwarded to
        ``_build_prompt_for_window`` and ``_extra_vqa_fields``.
        """
        return {}

    def _build_prompt_for_window(
        self, question, item_data, window_paths, window_idx, total_images
    ):
        """Build the prompt string for one inference window.

        Typically slices the ``item_data`` to the current window and
        delegates to ``_create_prompt``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement "
            "_build_prompt_for_window()"
        )

    def _extra_vqa_fields(self, item_data):
        """Return a dict of extra fields to merge into ``vqa_result``.
        Override to persist experiment-specific metadata in the output JSON."""
        return {}

    # -----------------------------------------------------------------------
    # Core inference logic (not overridden)
    # -----------------------------------------------------------------------

    def _generate_answer(self, question, image_paths, item_data):
        """Windowed inference loop — delegates each call to the
        injected ``model_backend``."""
        try:
            window_size = self.model_config.get("batch_size", 1)
            stride = (
                self.model_config.get("stride", window_size // 2)
                if window_size > 1
                else 1
            )

            total_images = len(image_paths)
            total_windows = max(
                1, (total_images - window_size + stride) // stride
            )
            all_responses = []

            for window_idx in range(total_windows):
                start_idx = window_idx * stride
                end_idx = min(start_idx + window_size, total_images)
                window_paths = image_paths[start_idx:end_idx]

                if (
                    window_idx == total_windows - 1
                    and end_idx < total_images
                ):
                    window_paths = image_paths[-window_size:]

                prompt = self._build_prompt_for_window(
                    question,
                    item_data,
                    window_paths,
                    window_idx,
                    total_images,
                )

                # Delegate inference to the pluggable backend
                response = self.model_backend.infer(prompt, window_paths)

                all_responses.append(
                    {"pages": window_paths, "answer": response}
                )

            return {
                "answer": all_responses,
                "query": question,
                "image_paths": image_paths,
                "analysis_type": f"window_size_{window_size}",
            }

        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            return {
                "answer": "Unable to determine: error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    CHECKPOINT_INTERVAL = 50  # Save checkpoint every N questions

    def _get_output_file(self):
        """Compute the final output filename (used by both checkpoint and save)."""
        output_file = (
            self.target_model.replace(":", "_")
            + "_"
            + self.config["output_file"]
        )

        if self.config.get("ocr_enabled") and not self.config.get(
            "unable_to_respond_aware"
        ):
            output_file = output_file.replace(".json", "_OCR_UNABLE.json")
        elif self.config.get("ocr_enabled"):
            output_file = output_file.replace(".json", "_OCR.json")
        elif not self.config.get("unable_to_respond_aware"):
            output_file = output_file.replace(".json", "_UNABLE.json")

        return output_file

    def _get_checkpoint_path(self):
        """Return the checkpoint file path for this experiment run."""
        output_file = self._get_output_file()
        experiment_name = self.__class__.__name__
        return output_file.replace(".json", f"_{experiment_name}.checkpoint.json")

    def _save_checkpoint(self, data, processed_index):
        """Save a checkpoint with current progress."""
        checkpoint = {
            "_checkpoint_meta": {
                "processed_index": processed_index,
                "experiment": self.__class__.__name__,
                "model": self.target_model,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "data": data,
        }
        checkpoint_path = self._get_checkpoint_path()
        try:
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load a checkpoint if one exists.

        Returns
        -------
        tuple (data, start_index) or (None, 0)
        """
        checkpoint_path = self._get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return None, 0

        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)

            meta = checkpoint.get("_checkpoint_meta", {})
            start_index = meta.get("processed_index", 0)
            data = checkpoint.get("data")

            if data and "corrupted_questions" in data:
                print(f"\n🔄 Checkpoint found! Resuming from question {start_index}/{len(data['corrupted_questions'])}")
                print(f"   (saved at {meta.get('timestamp', 'unknown')})\n")
                return data, start_index

        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Starting fresh.")

        return None, 0

    def _delete_checkpoint(self):
        """Remove the checkpoint file after successful completion."""
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"Checkpoint file removed: {checkpoint_path}")
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Main evaluation loop (not overridden)
    # -----------------------------------------------------------------------

    def evaluate(self):
        """Run the full evaluation pipeline with checkpoint support."""
        print(f"\nStarting evaluation with {self.target_model}...")

        # Try to resume from checkpoint
        data, start_index = self._load_checkpoint()

        if data is None:
            # Fresh start — load data from input file
            with open(self.config["input_file"]) as f:
                data = json.load(f)

            total_questions = len(data["corrupted_questions"])
            num_samples = int(
                total_questions * (self.sampling_percentage / 100)
            )

            if self.sampling_percentage < 100:
                data["corrupted_questions"] = random.sample(
                    data["corrupted_questions"], num_samples
                )
                print(
                    f"Sampled {num_samples} questions "
                    f"({self.sampling_percentage}%)"
                )
            start_index = 0

        questions = data["corrupted_questions"]
        total = len(questions)
        processed_count = 0
        success_count = 0
        error_count = 0

        for idx in tqdm(range(start_index, total), initial=start_index, total=total):
            item = questions[idx]
            try:
                processed_count += 1

                if "verification_result" not in item:
                    item["verification_result"] = {}
                if "vqa_results" not in item["verification_result"]:
                    item["verification_result"]["vqa_results"] = []

                question = item["corrupted_question"]
                pages = item["layout_analysis"]["pages"]

                # Build image paths
                image_paths = []
                for page_id in pages:
                    image_filename = os.path.basename(page_id)
                    image_paths.append(
                        os.path.join(
                            self.config["images_base_path"], image_filename
                        )
                    )

                # Extract experiment-specific features
                item_data = self._prepare_item_data(item, pages, image_paths)

                # Run inference
                result = self._generate_answer(
                    question, image_paths, item_data
                )

                # Assemble result record
                vqa_result = {
                    "model_type": self.target_model,
                    "model_config": {
                        "batch_size": self.model_config.get("batch_size", 1),
                    },
                    "question": question,
                    "answer": result.get("answer", "Unable to determine"),
                    "image_paths": image_paths,
                    "analysis_type": result.get("analysis_type", ""),
                    "timestamp": datetime.datetime.now().isoformat(),
                }

                # Merge experiment-specific fields
                vqa_result.update(self._extra_vqa_fields(item_data))

                if "error" in result:
                    vqa_result["error"] = result["error"]
                    vqa_result["traceback"] = result.get("traceback", "")
                    error_count += 1
                else:
                    success_count += 1

                item["verification_result"]["vqa_results"].append(vqa_result)

            except Exception as e:
                print(f"Error processing question {idx}: {str(e)}")
                error_count += 1

            # Save checkpoint periodically
            if (idx + 1) % self.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(data, idx + 1)
                print(f"  💾 Checkpoint saved at {idx + 1}/{total}")

        rate = (success_count / max(1, processed_count)) * 100
        print(f"\nProcessing completed. Success rate: {rate:.2f}%")
        self._save_results(data)
        self._delete_checkpoint()

    # -----------------------------------------------------------------------
    # Results persistence
    # -----------------------------------------------------------------------

    def _save_results(self, data):
        output_file = self._get_output_file()

        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

