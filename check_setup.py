"""Check that this machine can actually run the notebooks.

    python check_setup.py

Reports the PyTorch backend that will be used, any missing packages, and which
of the Hugging Face models the notebooks need are downloadable with your current
credentials. See SETUP.md for what to do about anything it flags.
"""

import importlib.util
import sys

# Package name -> import name, where they differ.
PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "plotly": "plotly",
    "tqdm": "tqdm",
    "einops": "einops",
    "jaxtyping": "jaxtyping",
    "transformers": "transformers",
    "datasets": "datasets",
    "diffusers": "diffusers",
    "transformer-lens": "transformer_lens",
    "sae-lens": "sae_lens",
    "smalldiffusion": "smalldiffusion",
    "umap-learn": "umap",
    "huggingface-hub": "huggingface_hub",
}

# repo id -> notebooks that need it
MODELS = [
    ("openai/clip-vit-base-patch32",
     "9_diffusion_clip"),
    ("gpt2",
     "8_attention, exercises/8_attention_exercises"),
    ("meta-llama/Llama-3.2-1B",
     "2_gradient_descent, 2_wormhole, 3_backpropagation, exercises/2_exercises_gradient_descent"),
    ("google/gemma-2-2b",
     "7_mech_interp"),
    ("google/gemma-2-2b-it",
     "7_mech_interp"),
    ("stabilityai/stable-diffusion-2-1",
     "9_stable_diffusion_ddim/_ddpm/_guidance"),
]

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def main():
    problems = []

    print("== Python ==")
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) < (3, 13):
        print(f"[{OK}] {sys.version.split()[0]}")
    else:
        print(f"[{WARN}] {sys.version.split()[0]} -- this stack is only known good on 3.10-3.12")
        problems.append("Python version")

    print("\n== Packages ==")
    # find_spec rather than import, so torch is not loaded before book_setup gets
    # a chance to set PYTORCH_ENABLE_MPS_FALLBACK.
    missing = []
    for dist, mod in PACKAGES.items():
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(dist)
        except (ImportError, ValueError):
            missing.append(dist)
    if missing:
        print(f"[{BAD}] missing: {', '.join(missing)}")
        print("         pip install -r requirements.txt")
        problems.append("missing packages")
        return report(problems)
    print(f"[{OK}] all {len(PACKAGES)} present")

    print("\n== PyTorch backend ==")
    import book_setup

    print(f"[{OK}] {book_setup.describe()}")
    if book_setup.DEVICE == "cpu":
        print("         No GPU backend found. Everything still runs, but the")
        print("         training and diffusion notebooks will be slow.")
    print(f"         exports -> {book_setup.EXPORT_DIR}")

    print("\n== Hugging Face model access ==")
    from huggingface_hub import auth_check, get_token
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if get_token():
        print(f"[{OK}] logged in")
    else:
        print(f"[{WARN}] no token found -- run `hf auth login` (needed for the gated models below)")

    blocked = []
    for repo, used_by in MODELS:
        try:
            auth_check(repo)
            print(f"[{OK}] {repo}")
        except (GatedRepoError, RepositoryNotFoundError):
            print(f"[{WARN}] {repo} -- no access")
            print(f"         needed by: {used_by}")
            print(f"         accept the license at https://huggingface.co/{repo}, then `hf auth login`")
            blocked.append(repo)
        except Exception as exc:  # offline, rate limited, DNS, ...
            print(f"[{WARN}] {repo} -- could not check ({type(exc).__name__})")
    if blocked:
        problems.append(f"{len(blocked)} model(s) not accessible")

    return report(problems)


def report(problems):
    print()
    if problems:
        print("Not ready: " + "; ".join(problems) + ". See SETUP.md.")
        return 1
    print("Ready. Launch with `jupyter lab` and pick any notebook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
