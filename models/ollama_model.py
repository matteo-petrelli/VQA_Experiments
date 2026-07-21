"""
Ollama model backend — uses the local Ollama server.

Supports any Ollama-compatible vision model, e.g.:
  - gemma3:4b
  - qwen3-vl:8b

Multi-GPU usage:
  Pass ``host`` to connect to a specific Ollama server instance
  (e.g. ``http://127.0.0.1:11435`` for a second server on GPU 1).
"""

import os
import shutil
import subprocess
import time

from models.base_model import ModelBackend


class OllamaBackend(ModelBackend):
    """Backend that delegates inference to a local Ollama server.

    Parameters
    ----------
    model_name : str
        Ollama model tag (e.g. ``qwen3-vl:8b``).
    host : str or None
        If provided, connect to this Ollama server URL instead of the
        default ``http://127.0.0.1:11434``.  Useful for multi-GPU
        setups where each GPU runs its own server on a different port.
    """

    def __init__(self, model_name="gemma3:4b", host=None):
        self.model_name = model_name
        self.host = host  # e.g. "http://127.0.0.1:11435"
        self._client = None

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
    def _get_client(self):
        """Lazily create an ollama.Client bound to the configured host."""
        if self._client is None:
            import ollama
            if self.host:
                self._client = ollama.Client(host=self.host)
            else:
                self._client = ollama.Client()
        return self._client

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
        client = self._get_client()

        output = client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_paths,
                }
            ],
            think=False,  # Disable thinking mode (e.g. Qwen3)
        )
        return output["message"]["content"].strip()

