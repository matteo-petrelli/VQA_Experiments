"""
Abstract base class for model backends.

Every backend must implement ``setup()`` and ``infer()``.
"""

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """Interface that every model backend must satisfy."""

    @abstractmethod
    def setup(self):
        """Perform any one-time initialisation (download weights, start
        server, load model onto GPU, …).  Called once before the
        evaluation loop starts."""

    @abstractmethod
    def infer(self, prompt, image_paths):
        """Run a single inference call.

        Parameters
        ----------
        prompt : str
            The fully-assembled prompt text.
        image_paths : list[str]
            Absolute paths to the page images for the current window.

        Returns
        -------
        str
            The model's response text (stripped).
        """

    def cleanup(self):
        """Optional: free GPU memory / stop server after evaluation."""
