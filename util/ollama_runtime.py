import ollama
import torch

class OllamaRuntime:
    def __init__(self, model="llama3.1", use_kv=True):
        self.model = model
        self.use_kv = use_kv

    def run_with_optional_kv(self, prompt, kv_cache=None):
        """
        Run inference with optional KV cache.
        For Ollama, KV caching is internal, so we just pass the prompt.
        Returns: (response_content, kv_tensor)
        """
        # Note: In a real environment with Ollama running, this would make a request.
        # For development/testing where Ollama might not be running, we might need to mock this
        # or wrap in try/except.
        try:
            response = ollama.chat(
                model=self.model,
                options={"num_ctx": 4096},
                messages=[{"role": "user", "content": prompt}],
                keep_alive=True  # enables reuse of kv
            )
            return response["message"]["content"], None
        except Exception as e:
            # Fallback for testing if Ollama is not reachable
            print(f"Ollama runtime warning: {e}")
            return "Mock Ollama Response", None
