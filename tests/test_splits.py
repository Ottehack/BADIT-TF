import json
from pathlib import Path

from badit_tf.splits import SplitCounts, build_split_manifest, load_role_instances


def _task(path: Path, count: int) -> None:
    payload = {
        "Definition": ["Do the task."],
        "Instances": [
            {"id": f"id-{index}", "input": str(index), "output": [str(index)]}
            for index in range(count)
        ],
    }
    for split in ("train", "dev", "test"):
        with (path / f"{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def test_manifest_is_deterministic_and_disjoint(tmp_path):
    for task_index in range(15):
        task = tmp_path / f"task{task_index:03d}"
        task.mkdir()
        _task(task, 40)
    counts = SplitCounts(
        assignment=2, fisher=2, damping_validation=1, fidelity=2
    )
    first = build_split_manifest(tmp_path, seed=5, counts=counts)
    second = build_split_manifest(tmp_path, seed=5, counts=counts)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    sets = {
        role: {row["sample_id"] for row in rows}
        for role, rows in first["roles"].items()
    }
    for role, ids in sets.items():
        assert len(ids) == len(first["roles"][role])
        assert not any(ids & other for name, other in sets.items() if name != role)
    resolved = load_role_instances(first, "assignment")
    assert len(resolved) == 30

