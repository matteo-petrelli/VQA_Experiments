"""
Ollama model backend — uses the local Ollama server.

Supports any Ollama-compatible vision model, e.g.:
  - gemma3:4b
  - qwen3-vl:8b
"""

import os
import shutil
import subprocess
import time

from models.base_model import ModelBackend


class OllamaBackend(ModelBackend):
    """Backend that delegates inference to a local Ollama server."""

    def __init__(self, model_name="gemma3:4b"):
        self.model_name = model_name

    # ------------------------------------------------------------------
    def _find_ollama(self):
        """Locate the ollama binary (cross-platform)."""
        # Try PATH first (works on both Windows and Linux)
        path = shutil.which("ollama")
        if path:
            return path

        # Fallback: common Linux locations (e.g. Kaggle)
        for candidate in ("/usr/local/bin/ollama", "/usr/bin/ollama"):
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            "Could not find the 'ollama' binary. "
            "Make sure Ollama is installed and in your PATH."
        )

    # ------------------------------------------------------------------
    def setup(self):
        """Start the Ollama server and pull the model."""
        ollama_path = self._find_ollama()

        print("Starting Ollama background server...")
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)

        print(f"Pulling model '{self.model_name}'. This may take a minute...")
        subprocess.run([ollama_path, "pull", self.model_name])
        print("Ollama is ready!\n")

    # ------------------------------------------------------------------
    def infer(self, prompt, image_paths):
        import ollama

        output = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_paths,
                }
            ],
        )
        return output["message"]["content"].strip()
