"""
Ollama model backend — uses the local Ollama server (e.g. Gemma 3 4B).
"""

import os
import subprocess
import time

from models.base_model import ModelBackend


class OllamaBackend(ModelBackend):
    """Backend that delegates inference to a local Ollama server."""

    def __init__(self, model_name="gemma3:4b"):
        self.model_name = model_name

    # ------------------------------------------------------------------
    def setup(self):
        """Start the Ollama server and pull the model."""
        ollama_path = (
            "/usr/local/bin/ollama"
            if os.path.exists("/usr/local/bin/ollama")
            else "/usr/bin/ollama"
        )

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
