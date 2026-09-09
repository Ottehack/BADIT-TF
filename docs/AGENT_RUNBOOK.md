# Agent runbook

## Acceptance sequence

1. `badit-tf audit` must return `PASS`.
2. `pytest` must pass without a GPU or external dataset.
3. `badit-tf doctor --config configs/base.yaml --config <recipe overlay>` must
   resolve configuration and find the configured model/data roots.
4. `badit-tf run <recipe> --phase <phase> ... --dry-run` must produce the
   intended launcher, GPU count, runner, resolved config, and forwarded flags.
5. Execute the command and preserve `.runs/<recipe>_<phase>_<timestamp>.yaml`.

## Minimal local layout

```text
local/
  models/Qwen3-4B/
  data/SuperNI/
  artifacts/
    assignments/
    checkpoints/
    profiles/
    splits/
outputs/
```

The directories may live elsewhere; `.env` is the only place that needs to
change. Environment values may be relative to the repository or absolute on the
operator's machine. Generated/resolved configs are excluded from Git.

## Launch conventions

- `python`: CPU aggregation/preparation or a single-process diagnostic.
- `torchrun`: one process per local GPU. Override with `--world-size N`.
- Runner-specific options go after `--`.
- Phases whose legacy aggregator has no `--config` argument should use
  `--no-config` and pass its required flags after `--`.

## Evidence required for a formal result

- immutable run ID and source commit;
- resolved YAML and SHA256;
- model/checkpoint SHA256 or immutable upstream revision;
- sample-ID manifest and SHA256 for every split;
- raw per-task/per-layer/per-probe outputs;
- seed, GPU/CUDA/PyTorch/Transformers/DeepSpeed fingerprint;
- aggregation command and summary artifact hash;
- explicit `SUCCESS`, `FAILED`, `PARTIAL`, or `INVALID` state.

Submission to a scheduler and the existence of a log file are not completion.

