from __future__ import annotations

import subprocess
from pathlib import Path

from engine.model_store import ModelArtifact
from engine.rvc_engine import RVCEngine, RVCSettings
from engine.schemas import RVCParams


def test_official_rvc_runs_cli_as_a_module(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "rvc"
    (root / "infer").mkdir(parents=True)
    (root / "infer" / "cli.py").write_text("", encoding="utf-8")
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    source = input_directory / "cue.wav"
    source.write_bytes(b"input")
    checkpoint = tmp_path / "voice.pth"
    checkpoint.write_bytes(b"model")
    output_directory = tmp_path / "output"
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / source.name).write_bytes(b"converted")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = RVCEngine(RVCSettings("official-rvc", root, "rvc-python", 30))

    outputs = engine.convert_batch(
        [source],
        output_directory,
        ModelArtifact(checkpoint, None, "sha256"),
        RVCParams(),
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == ["rvc-python", "-m", "infer.cli"]
    assert captured["cwd"] == root
    assert outputs[source].read_bytes() == b"converted"


def test_official_rvc_aliases_community_trained_index(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "rvc"
    (root / "infer").mkdir(parents=True)
    (root / "infer" / "cli.py").write_text("", encoding="utf-8")
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    source = input_directory / "cue.wav"
    source.write_bytes(b"input")
    checkpoint = tmp_path / "voice.pth"
    checkpoint.write_bytes(b"model")
    trained_index = tmp_path / "trained_IVF.index"
    trained_index.write_bytes(b"index")
    output_directory = tmp_path / "output"
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        index = Path(command[command.index("--index") + 1])
        captured["index"] = index
        captured["index_bytes"] = index.read_bytes()
        output = Path(command[command.index("--output") + 1])
        (output / source.name).write_bytes(b"converted")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = RVCEngine(RVCSettings("official-rvc", root, "rvc-python", 30))

    outputs = engine.convert_batch(
        [source],
        output_directory,
        ModelArtifact(checkpoint, trained_index, "sha256"),
        RVCParams(index_rate=0.75),
    )

    index = captured["index"]
    assert isinstance(index, Path)
    assert "trained" not in index.name
    assert captured["index_bytes"] == b"index"
    assert not index.exists()
    assert outputs[source].read_bytes() == b"converted"
