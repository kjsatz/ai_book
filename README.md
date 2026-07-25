# The Welch Labs Illustrated Guide to AI

This repo contains supporting code for the [Welch Labs Illustrated Guide to AI](https://www.welchlabs.com/ai-book). 

## Getting started

```bash
pip install -r requirements.txt
python check_setup.py
jupyter lab
```

The notebooks pick a PyTorch backend automatically — CUDA, Apple Silicon MPS, or
CPU — so they run as-is on most machines. A few chapters need Hugging Face models
that require accepting a license first; `check_setup.py` tells you which, and
[SETUP.md](SETUP.md) walks through the rest.
