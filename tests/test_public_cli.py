from pathlib import Path

from badit_tf.cli import ROOT, _catalog, resolve_configs


def test_catalog_scripts_exist() -> None:
    for recipe in _catalog().values():
        for phase in recipe["phases"].values():
            assert (ROOT / phase["script"]).is_file()


def test_config_overlay_and_environment(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ROOT", "local/models")
    monkeypatch.setenv("DATA_ROOT", "local/data/SuperNI")
    monkeypatch.setenv("ARTIFACT_ROOT", "local/artifacts")
    monkeypatch.setenv("OUTPUT_ROOT", "outputs")
    config = resolve_configs(
        [ROOT / "configs/base.yaml", ROOT / "configs/recipes/p1.yaml"]
    )
    assert config["model_path"] == "local/models/Qwen3-4B"
    assert config["output_dir"] == "outputs/p1"
    assert config["world_size"] == 8

