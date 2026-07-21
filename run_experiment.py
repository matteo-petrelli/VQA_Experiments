"""
CLI entry point — run any VQA experiment with any model backend.

Usage examples:
    python run_experiment.py --experiment docel_cot_v1 --backend ollama --model gemma3:4b
    python run_experiment.py --experiment docel_cot_v1 --backend ollama --model qwen3-vl:8b
    python run_experiment.py --experiment baseline_ocr  --backend phi
    python run_experiment.py --experiment nlp_tag_cot   --backend qwen
"""

import argparse
import json
import os
import sys

from experiment_registry import EXPERIMENTS, get_experiment_class


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

BACKENDS = {
    "ollama": "models.ollama_model.OllamaBackend",
    "phi":    "models.phi_model.PhiBackend",
    "qwen":  "models.qwen_model.QwenBackend",
}


def _create_backend(backend_name, model_override=None, ollama_host=None):
    """Instantiate and return the selected ModelBackend."""
    if backend_name == "ollama":
        from models.ollama_model import OllamaBackend
        model_name = model_override or "gemma3:4b"
        return OllamaBackend(model_name=model_name, host=ollama_host)

    elif backend_name == "phi":
        from models.phi_model import PhiBackend
        model_name = model_override or PhiBackend.DEFAULT_MODEL
        return PhiBackend(model_name=model_name)

    elif backend_name == "qwen":
        from models.qwen_model import QwenBackend
        model_name = model_override or QwenBackend.DEFAULT_MODEL
        return QwenBackend(model_name=model_name)

    else:
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Available: {', '.join(BACKENDS.keys())}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a VQA Pipeline Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available experiments:\n  "
            + "\n  ".join(sorted(EXPERIMENTS.keys()))
            + "\n\nAvailable backends:\n  "
            + "\n  ".join(sorted(BACKENDS.keys()))
        ),
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment (e.g. 'docel_cot_v1')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=sorted(BACKENDS.keys()),
        help="Model backend to use (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default model name for the chosen backend",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/kaggle/input/datasets/matteopetrelli/config-phi/config_kaggle.json",
        help="Path to the config JSON file",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama server URL (e.g. http://127.0.0.1:11435 for multi-GPU)",
    )

    args = parser.parse_args()

    # -- Validate experiment name --
    try:
        ExperimentClass = get_experiment_class(args.experiment)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # -- Validate config path --
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    # -- Handle baseline / baseline_ocr config overrides --
    config_path_to_use = args.config
    temp_config_path = None

    if args.experiment in ("baseline", "baseline_ocr"):
        with open(args.config, "r") as f:
            config_data = json.load(f)

        config_data["ocr_enabled"] = (args.experiment == "baseline_ocr")

        temp_config_path = os.path.join(os.getcwd(), "_temp_config.json")
        with open(temp_config_path, "w") as f:
            json.dump(config_data, f)
        config_path_to_use = temp_config_path

        note = "OCR enabled" if config_data["ocr_enabled"] else "OCR disabled"
        print(f"Note: Experiment '{args.experiment}' — {note}.")

    # -- Create and set up the backend --
    backend = _create_backend(args.backend, args.model, ollama_host=getattr(args, 'ollama_host', None))

    try:
        backend.setup()
    except Exception as e:
        print(f"Warning: Backend setup issue: {e}")
        print("Continuing — the backend may already be running.\n")

    # -- Instantiate experiment and run --
    print(f"\nInitializing {ExperimentClass.__name__}...")
    evaluator = ExperimentClass(
        config_path=config_path_to_use,
        model_backend=backend,
        experiment_name=args.experiment,
    )

    try:
        evaluator.evaluate()
    finally:
        backend.cleanup()
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()
