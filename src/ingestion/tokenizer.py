# src/ingestion/tokenizer.py
import logging

try:
    import tiktoken
    _TK_AVAILABLE = True
except Exception:
    tiktoken = None
    _TK_AVAILABLE = False
    logging.getLogger(__name__).warning("tiktoken not available; falling back to whitespace tokenizer")

class Tokenizer:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        if _TK_AVAILABLE:
            try:
                # prefer cl100k_base if available
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                try:
                    self.enc = tiktoken.encoding_for_model(model_name)
                except Exception:
                    self.enc = None

    def count_tokens(self, text: str) -> int:
        """Return estimated token count for `text`."""
        if _TK_AVAILABLE and getattr(self, "enc", None) is not None:
            return len(self.enc.encode(text))
        # fallback: approximate by whitespace tokens
        return max(1, len(text.split()))