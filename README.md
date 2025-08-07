# Axiomatic Greek Language

This repository explores the Greek language through an axiomatic approach. Rules of grammar are expressed as axioms, from which theorems about the language can be derived.

The project will leverage large language models (LLMs) to propose candidate axioms and assist in proving theorems that formalize linguistic behavior.

## LLM Integration

The `llm_helper.py` script wraps a pretrained language model to experiment with axiom generation.
Install dependencies first:

```bash
pip install transformers torch
```

Then run the assistant with a prompt:

```bash
python llm_helper.py "Give a Greek grammar axiom."
```
