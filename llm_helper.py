"""Utility for interacting with a language model to propose axioms for Greek."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GreekLLMAssistant:
    """Wrapper around a causal language model for Greek grammar exploration."""

    model_name: str = "gpt2"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text from the language model based on a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the language model for Greek axioms.")
    parser.add_argument("prompt", help="Prompt given to the language model")
    parser.add_argument("--max-new-tokens", type=int, default=50, dest="max_new_tokens",
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()

    assistant = GreekLLMAssistant()
    print(assistant.generate(args.prompt, max_new_tokens=args.max_new_tokens))
