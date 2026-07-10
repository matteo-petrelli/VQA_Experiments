# Reproducible VQA Agentic Pipeline Experiments

This folder contains a fully reproducible, object-oriented refactoring of the VQA pipeline experiments. 
It replaces the isolated Kaggle Jupyter notebooks and the `Experiments_Prompt.txt` file with a single cohesive Python codebase.

## How to use

Run the `run_experiment.py` script from the command line. You choose **which experiment** (prompt variant) and **which model backend** to use.

```bash
# Ollama backend (default) — Gemma 3 4B
python run_experiment.py --experiment docel_cot_v1 --backend ollama --model gemma3:4b

# Phi-3.5 Vision (HuggingFace)
python run_experiment.py --experiment nlp_tag_cot --backend phi

# Qwen 2.5 VL (HuggingFace, 8-bit quantisation)
python run_experiment.py --experiment baseline_ocr --backend qwen

# Qwen3-VL 8B via Ollama
python run_experiment.py --experiment docel_cot_v1 --backend ollama --model qwen3-vl:8b

# Custom config file
python run_experiment.py --experiment layout_v4 --backend ollama --config path/to/config.json
```

You can list all available experiments and backends by running:
```bash
python run_experiment.py --help
```

## Available Experiments

### Baseline Family
- `baseline`: Image-only reasoning.
- `baseline_ocr`: Image + standard OCR text.

### DocEl Family (Structured Document Elements)
- `docel`: Structured OCR with layout labels (e.g. `[Title]`).
- `docel_cot_v1` through `docel_cot_v4`: Structured OCR with various Chain-of-Thought reasoning steps.
- `docel_cot_numvre`: Adds document layout summary (percentages of layout element types).

### NLP Tag Family
- `nlp_tag`: OCR annotated inline with NLP entity tags.
- `nlp_tag_cot`: Tagged OCR + Chain-of-Thought reasoning.

### NLP List Family
- `nlp_list`: Entity matching between question and document entities.
- `nlp_list_cot`: Entity matching + Chain-of-Thought reasoning.
- `nlp_list_ocr`: Entity matching + standard OCR text.
- `nlp_list_ocr_cot`: Full entity pipeline with CoT and OCR text.

### Layout Family
- `layout_v1` through `layout_v4`: Spatial and positional reasoning on the image without text, using quadrant analysis.

## Available Model Backends

| Backend  | Model                              | Description                                        |
|----------|------------------------------------|----------------------------------------------------||
| `ollama` | `gemma3:4b` (default)              | Local Ollama server. Any Ollama vision model works. |
| `ollama` | `qwen3-vl:8b`                     | Qwen3-VL 8B via Ollama. Pass `--model qwen3-vl:8b`.|
| `phi`    | `Phi-3.5-vision-instruct` (default)| HuggingFace, FP16 precision.                       |
| `qwen`   | `Qwen2.5-VL-3B-Instruct` (default)| HuggingFace, 8-bit quantisation.                   |

## Architecture

```
reproducible/
├── base_evaluator.py           # BaseVQAEvaluator — shared loop, utilities
├── experiment_registry.py      # Maps experiment names → classes
├── run_experiment.py           # CLI entry point
├── experiments/                # One file per experiment family
│   ├── baseline.py
│   ├── docel.py
│   ├── nlp_tag.py
│   ├── nlp_list.py
│   └── layout.py
└── models/                     # One file per model backend
    ├── base_model.py           # Abstract ModelBackend interface
    ├── ollama_model.py         # Ollama (Gemma, Qwen3-VL, etc.)
    ├── phi_model.py            # Phi-3.5 Vision (HuggingFace)
    └── qwen_model.py           # Qwen 2.5 VL (HuggingFace)
```

- **`base_evaluator.py`**: Contains `BaseVQAEvaluator` with the evaluation loop, windowed inference, and all shared OCR/NLP utility methods. It receives a `ModelBackend` instance via dependency injection.
- **`experiments/`**: Each file contains subclasses that override `_create_prompt()`, `_prepare_item_data()`, and `_build_prompt_for_window()`.
- **`models/`**: Each file implements `ModelBackend.infer(prompt, image_paths)` for a specific model/framework. The same 17 experiment variants work with **any** backend.
