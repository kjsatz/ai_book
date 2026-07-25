# Setup

## Quick start

```bash
pip install -r requirements.txt
python check_setup.py
jupyter lab
```

`check_setup.py` reports which PyTorch backend you'll get, whether any packages
are missing, and which of the Hugging Face models the notebooks use you can
currently download. Everything it flags is explained below.

Python 3.10–3.12. On 3.13+, `numba` (via `umap-learn`) has no usable wheel yet.

## Choosing a device

You don't have to. `book_setup` picks the best available backend on import and
every notebook uses it:

```python
from book_setup import DEVICE
model = model.to(DEVICE)
```

The order is CUDA → MPS → XPU → CPU, so the same notebook runs unmodified on an
NVIDIA box, an Apple Silicon Mac, or a laptop with no GPU. ROCm builds of PyTorch
report through the CUDA API, so AMD cards are covered by the first branch.

To force a specific backend — usually to check whether a bug is backend-specific:

```bash
AI_BOOK_DEVICE=cpu jupyter lab
```

On Apple Silicon, `book_setup` also sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so the
handful of operators with no Metal kernel run on the CPU instead of raising.
PyTorch only reads that variable while `import torch` runs, which is why
`book_setup` is imported **first** in each notebook's import cell. If you import
it after torch you'll get a warning saying the fallback didn't take effect.

## Hugging Face models

Several notebooks download model weights on first run. Most are open and just
work. Four are **gated**: the model card is public, but the weight files are
access-controlled because the publisher requires you to accept a license first.

Getting access is a three-step thing, and all three steps are required — having
an account isn't enough, and neither is accepting the license if your machine
can't prove who you are.

**1. Create a Hugging Face account** at <https://huggingface.co/join>. Free.

**2. Accept each model's license.** Visit the model page while logged in. Gated
repos show a box at the top of the page — something like *"You need to agree to
share your contact information to access this model"* — with a form or a button.

| Model | Needed by | Typical wait |
|---|---|---|
| [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | ch. 2 and 3 | form; minutes to a few hours |
| [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) | ch. 7 | click-through; instant |
| [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | ch. 7 | click-through; instant |
| [`stabilityai/stable-diffusion-2-1`](https://huggingface.co/stabilityai/stable-diffusion-2-1) | ch. 9 (`9_stable_diffusion_*`) | click-through; instant |

Approval is per-account, not per-machine — do it once.

**3. Log in on this machine** so downloads are authenticated:

```bash
hf auth login
```

It asks for an access token, which you create at
<https://huggingface.co/settings/tokens>. A **read** token is sufficient. The
token is saved to `~/.cache/huggingface/token` and picked up automatically from
then on. In CI or a container, set `HF_TOKEN` in the environment instead.

Re-run `python check_setup.py` to confirm; each gated model should flip to `ok`.

### Reading the error messages

If you skip a step, the failure looks different depending on the library:

- **transformers** is clear: `You are trying to access a gated repo` plus a 401.
- **diffusers** is misleading: `stabilityai/stable-diffusion-2-1 is not a local
  folder and is not a valid model identifier listed on 'https://huggingface.co/models'`.
  The repo exists and the id is correct — the real cause is a 401, and the fix is
  the three steps above.

### Avoiding the gate

Only Stable Diffusion has a drop-in open substitute. In the `9_stable_diffusion_*`
notebooks, set:

```python
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
```

That checkpoint is ungated. The chapter's argument doesn't depend on which of
the two you use, though the images differ.

## Data the notebooks can't download

Two things aren't fetchable and have to be supplied by hand:

- **Animal Faces (AFHQ)** for `1_perceptron_cats_vs_dogs`. Download
  [`andrewmvd/animal-faces`](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
  from Kaggle, then point the notebook at the extracted `train` folder:

  ```bash
  AI_BOOK_AFHQ_DIR=~/Downloads/afhq/train jupyter lab
  ```

- **Precomputed loss landscapes** for the wormhole figures in
  `2_gradient_descent`, which loads `.npy` files from `wormhole_merged/` and
  `apr_29_2/` under the export directory. These come from long runs of
  `2_wormhole.ipynb`; the rest of that notebook works without them.

## Where output goes

Notebooks that save figures or CSVs write under `exports/` in the repo, via
`book_setup.export_path()`, which creates parent directories as needed.
`AI_BOOK_EXPORT_DIR` overrides the location.

## Environment variables

| Variable | Effect |
|---|---|
| `AI_BOOK_DEVICE` | Force a PyTorch device (`cpu`, `cuda`, `cuda:1`, `mps`, …) |
| `AI_BOOK_EXPORT_DIR` | Where notebooks write figures and CSVs |
| `AI_BOOK_AFHQ_DIR` | AFHQ `train` folder for `1_perceptron_cats_vs_dogs` |
| `HF_TOKEN` | Hugging Face token, as an alternative to `hf auth login` |

## Troubleshooting

**`MPS framework doesn't support float64`** — MPS is float32-only. Cast the
tensor, or run that notebook with `AI_BOOK_DEVICE=cpu`.

**`Placeholder storage has not been allocated on MPS device!`** — a tensor and a
module are on different devices. Usually something was constructed without being
moved to `DEVICE`.

**A notebook looks like it already ran.** Outputs are committed, so cells show
the author's results before you execute anything. Kernel → Restart and Run All.

**Downloads are slow on first run.** Stable Diffusion is several GB. Everything
lands in `~/.cache/huggingface` and is reused afterwards.
