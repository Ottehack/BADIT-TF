# Data, models, and frozen artifacts

## SuperNI

Download Super-NaturalInstructions from its official distribution and point
`DATA_ROOT` at the dataset root. This artifact does not redistribute dataset
contents. Use `experiments/prepare/freeze_splits.py` and the P2/P3 preparation
scripts to create immutable sample-ID manifests. Retain the generated SHA256.

## Models

Place supported Hugging Face or ModelScope snapshots under `MODEL_ROOT`, or set
`model_path` in a private overlay. Formal runs used Qwen, Llama, and Gemma
families; exact revisions and access terms must be frozen by the reproducer.
The runners use `local_files_only=True`, so model downloads are intentionally a
separate, auditable step.

## Checkpoints and assignments

The repository does not include large training checkpoints. Put initial-bank
and trained checkpoints under `ARTIFACT_ROOT/checkpoints`, assignments under
`ARTIFACT_ROOT/assignments`, profiles under `ARTIFACT_ROOT/profiles`, and frozen
manifests under `ARTIFACT_ROOT/splits`. Replace placeholder names with your
actual immutable artifacts in a private YAML overlay.

## Secrets

No experiment requires a credential after assets have been downloaded. Never
place access tokens, OSS credentials, internal hosts, or `.env` in Git.

