import argparse
import sys
import json
import os

from base_evaluator import setup_ollama
from experiment_registry import EXPERIMENTS, get_experiment_class

def main():
    parser = argparse.ArgumentParser(
        description="Run a VQA Pipeline Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available experiments:\n  " + "\n  ".join(EXPERIMENTS.keys())
    )
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True,
        help="Name of the experiment to run (e.g., 'baseline', 'docel_cot_v1')"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemma3:4b",
        help="Name of the Ollama model to use (default: gemma3:4b)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="/kaggle/input/datasets/matteopetrelli/config-phi/config_kaggle.json",
        help="Path to the config JSON file"
    )

    args = parser.parse_args()

    # Retrieve experiment class
    try:
        ExperimentClass = get_experiment_class(args.experiment)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Validate config path
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    # Set up Ollama server (downloads model if necessary)
    try:
        setup_ollama(model_name=args.model)
    except Exception as e:
        print(f"Warning: Failed to start Ollama automatically: {e}")
        print("Assuming Ollama is already running or you are not using Ollama.")

    # Check for baseline_ocr special case
    # If the user requested baseline_ocr, we need to ensure the config has "ocr_enabled" = True
    config_overrides = {}
    if args.experiment == "baseline_ocr":
        print("Note: Experiment is 'baseline_ocr'. Ensuring 'ocr_enabled' is True.")
        with open(args.config, "r") as f:
            config_data = json.load(f)
        config_data["ocr_enabled"] = True
        
        # Save temporary config
        temp_config_path = args.config + ".temp"
        with open(temp_config_path, "w") as f:
            json.dump(config_data, f)
        config_path_to_use = temp_config_path
    elif args.experiment == "baseline":
        print("Note: Experiment is 'baseline' (no OCR). Ensuring 'ocr_enabled' is False.")
        with open(args.config, "r") as f:
            config_data = json.load(f)
        config_data["ocr_enabled"] = False
        
        # Save temporary config
        temp_config_path = args.config + ".temp"
        with open(temp_config_path, "w") as f:
            json.dump(config_data, f)
        config_path_to_use = temp_config_path
    else:
        config_path_to_use = args.config

    # Initialize and run evaluator
    print(f"\nInitializing {ExperimentClass.__name__}...")
    evaluator = ExperimentClass(
        config_path=config_path_to_use, 
        model_name=args.model
    )
    
    try:
        evaluator.evaluate()
    finally:
        # Cleanup temp config if we created one
        if config_path_to_use.endswith(".temp") and os.path.exists(config_path_to_use):
            os.remove(config_path_to_use)

if __name__ == "__main__":
    main()
