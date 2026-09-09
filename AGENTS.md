# BADIT-TF Agent Instructions

Your goal is to reproduce or extend BADIT-TF without changing the experimental
protocol silently.

1. Work from this repository root. Never insert cluster IPs or user-specific
   absolute paths into tracked files.
2. Read `README.md`, `docs/EXPERIMENTS.md`, and `docs/DATA_AND_MODELS.md`.
3. Copy `.env.example` to `.env`; set model, data, artifact, output, and GPU
   roots there. Do not commit `.env`.
4. Install with `python -m pip install -e '.[train,test]'`.
5. Run `badit-tf audit`, `pytest`, and `badit-tf doctor` before a GPU job.
6. Use `badit-tf list` and `badit-tf run ... --dry-run` to inspect the exact
   command. Launch only after all referenced frozen inputs exist.
7. Preserve failed runs. A correction receives a new run ID and records its
   deviation; it never overwrites a failed artifact.
8. Do not select hyperparameters on final-test or fidelity results. Keep
   assignment, Fisher, validation, fidelity, confirmation, and final-test
   sample IDs disjoint as required by the protocol.
9. Store resolved config, hashes, stdout/stderr, environment fingerprint,
   checkpoint identity, and raw per-task metrics beside each run.
10. Multi-task and continual learning are both primary experiments. Do not
    substitute one for the other or report a smoke/proxy run as a paper result.

If an input is missing, stop with a precise missing-artifact report. Do not
fabricate a path, checkpoint, result, or paper number.

