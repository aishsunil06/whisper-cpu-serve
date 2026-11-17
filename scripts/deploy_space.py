"""Deploy this service to a Hugging Face Space (Docker SDK).

    python scripts/deploy_space.py

Requires a write-scoped token: `huggingface-cli login` or HF_TOKEN in the
environment. The namespace is read from the token rather than hardcoded.

Uploads only what the container needs: the Dockerfile, serving requirements,
source, README (its YAML frontmatter is the Space config), and the INT8 model.
The fp32 graphs, tests, docs and benchmark output are deliberately left out --
same reasoning as .dockerignore, since the deployed artifact is the optimized
one. The .onnx files exceed Hugging Face's inline limit, so upload_folder puts
them through LFS automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "whisper-cpu-serve"

# Mirrors .dockerignore: everything the image needs, nothing it does not.
ALLOW_PATTERNS = [
    "Dockerfile",
    "requirements.txt",
    "README.md",
    "src/**",
    "models/onnx-int8/**",
]


def main() -> None:
    api = HfApi()

    try:
        username = api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(
            "Not authenticated to Hugging Face. Run `huggingface-cli login` "
            f"(needs a write-scoped token) and try again.\n  {type(exc).__name__}: {exc}"
        ) from exc

    repo_id = f"{username}/{SPACE_NAME}"

    model_dir = REPO_ROOT / "models" / "onnx-int8"
    if not (model_dir / "encoder_model.onnx").exists():
        raise SystemExit(
            f"INT8 model missing at {model_dir}. Run scripts/export_onnx.py first."
        )

    print(f"Creating Space {repo_id} (docker sdk)...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    size_mb = sum(p.stat().st_size for p in model_dir.glob("*.onnx")) / 1e6
    print(f"Uploading source and the INT8 model ({size_mb:.0f} MB via LFS)...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(REPO_ROOT),
        allow_patterns=ALLOW_PATTERNS,
        commit_message="Deploy quantized Whisper CPU inference service",
    )

    print(f"\nDone: https://huggingface.co/spaces/{repo_id}")
    print("The Space builds the image on push; first boot takes several minutes.")
    print(f"Check: curl https://{username}-{SPACE_NAME}.hf.space/health")


if __name__ == "__main__":
    sys.exit(main())
