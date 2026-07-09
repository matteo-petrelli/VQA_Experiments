# Reproducible VQA Agentic Pipeline Experiments

This folder contains a fully reproducible, object-oriented refactoring of the VQA pipeline experiments. 
It replaces the isolated Kaggle Jupyter notebooks and the `Experiments_Prompt.txt` file with a single cohesive Python codebase.

## How to use

Run the `run_experiment.py` script from the command line. You can choose which experiment to run by providing the `--experiment` flag.

```bash
# Basic usage (defaults to gemma3:4b and the standard kaggle config)
python run_experiment.py --experiment docel_cot_v1

# Specifying a different model and config file
python run_experiment.py --experiment baseline_ocr --model qwen2.5:3b --config path/to/config.json
```

## Available Experiments

You can list all available experiments by running:
```bash
python run_experiment.py --help
```

### Baseline Family
- `baseline`: Image-only reasoning.
- `baseline_ocr`: Image + standard OCR text.

### DocEl Family (Structured Document Elements)
- `docel`: Structured OCR with layout labels (e.g. `[Title]`).
- `docel_cot_v1` through `docel_cot_v4`: Structured OCR with various Chain-of-Thought reasoning steps.
- `docel_cot_numvre`: Adds document layout summary (percentages of layout element types).

### NLP Tag Family
- `nlp_tag`: OCR annotated inline with NLP tags (e.g. `<year>2011</year>`).
- `nlp_tag_cot`: Tagged OCR + Chain-of-Thought reasoning.

### NLP List Family
- `nlp_list`: Entity matching between question entities and document entities.
- `nlp_list_cot`: Entity matching + Chain-of-Thought reasoning.
- `nlp_list_ocr`: Entity matching + standard OCR text.
- `nlp_list_ocr_cot`: Full entity pipeline with CoT and OCR text.

### Layout Family
- `layout_v1` through `layout_v4`: Spatial and positional reasoning on the image without text, using quadrant analysis.

## Architecture

- `base_evaluator.py`: Contains `BaseVQAEvaluator`, which handles Ollama setup, standard OCR extraction utilities, the batched window inference loop, and evaluation statistics.
- `experiments/`: Contains Python files for each family. Subclasses inherit from the base evaluator and override:
  - `_prepare_item_data()`: To extract specific features from the dataset JSON (e.g., layout tags, entities).
  - `_create_prompt()`: To provide the exact experiment prompt.
  - `_build_prompt_for_window()`: To assemble the prompt for a specific sliding window of pages.
