# BADIT-TF

This repository is the portable experiment artifact for BADIT-TF. It contains
the method implementation, multi-task and continual-learning runners, objective
and fidelity diagnostics, ablations, transfer/stability/efficiency analyses,
aggregation code, and an Agent-oriented runbook.

The artifact never assumes an internal mount, cluster address, PM service, or
OSS bucket. Models, SuperNI data, frozen manifests, checkpoints, and outputs are
configured through environment variables and YAML overlays.

## Start here (human or Agent)

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
badit-tf audit
pytest
badit-tf list
```

The CPU unit suite validates the Task-Fisher core without downloading a model.
For GPU experiments, install the training dependencies and edit only `.env`:

```bash
python -m pip install -e '.[train,test]'
badit-tf doctor --config configs/base.yaml --config configs/recipes/p1.yaml
badit-tf run p1 --phase collect \
  --config configs/base.yaml --config configs/recipes/p1.yaml --dry-run
```

Remove `--dry-run` after `doctor` passes and the referenced manifests and
checkpoints exist. Arguments following `--` are forwarded to the underlying
runner. For example:

```bash
badit-tf run p2 --phase train \
  --config configs/base.yaml --config configs/recipes/multitask.yaml \
  -- --variant tf --run-id qwen3_4b_mixed_tf_seed1
```

Continual learning is restart-delimited. Run one task stage at a time and pass
the preceding stage checkpoint explicitly:

```bash
badit-tf run p3 --phase train-stage \
  --config configs/base.yaml --config configs/recipes/continual.yaml \
  -- --run-id qwen3_4b_continual_tf_seed1 --task-index 0
```

See [docs/AGENT_RUNBOOK.md](docs/AGENT_RUNBOOK.md) for the exact Agent contract,
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for experiment-to-code coverage, and
[docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md) for external assets.

## Repository layout

```text
src/badit_tf/          Method, assignment, calibration, training and reporting
experiments/runners/   GPU and analysis entry points
experiments/prepare/   Split/config/assignment preparation
experiments/aggregate/ Metric aggregation and table construction
configs/               Portable overlays and recipe catalog
tools/                 Result exporters
tests/                 CPU-verifiable core tests
```

## Reproducibility boundary

This package includes code and portable configuration templates, not model
weights, SuperNI contents, private checkpoints, internal scheduler scripts, or
raw cluster logs. A formal run is valid only when its resolved config, sample-ID
manifest, code commit, environment fingerprint, and output hashes are retained.

## Publication status

The technical artifact is staged and audited, but it is not legally ready for a
public release until the authors add a project license. Read
[LICENSE_SELECTION_REQUIRED.md](LICENSE_SELECTION_REQUIRED.md) before publishing.

