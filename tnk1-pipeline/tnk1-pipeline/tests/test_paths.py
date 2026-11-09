"""Tests for the feature-store resolver and manifest verification."""
import hashlib
import json

import pytest

from tnk1.utils import paths
from tnk1.utils.verify_features import main as verify_main


def _write(d, name, content: bytes):
    p = d / name
    p.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


@pytest.fixture
def store(tmp_path):
    """A miniature feature store: one good file, one corrupt, one absent."""
    sha_a, size_a = _write(tmp_path, "good.parquet", b"alpha-contents")
    _write(tmp_path, "corrupt.parquet", b"tampered")
    manifest = {
        "good.parquet": {"size_bytes": size_a, "sha256": sha_a},
        "corrupt.parquet": {"size_bytes": 8, "sha256": "0" * 64},
        "gone.parquet": {"size_bytes": 10, "sha256": "1" * 64},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_repo_root_contains_the_checkout():
    root = paths.repo_root()
    assert (root / "tnk1-pipeline").is_dir()
    assert (root / "ml_research").is_dir()


def test_features_dir_resolves_inside_the_repo(monkeypatch):
    monkeypatch.delenv("TNK1_FEATURES_DIR", raising=False)
    d = paths.features_dir()
    assert d.name == "features"
    assert (d / "manifest.json").exists()


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("TNK1_FEATURES_DIR", str(tmp_path))
    assert paths.features_dir() == tmp_path


def test_env_override_pointing_nowhere_raises_when_required(monkeypatch, tmp_path):
    monkeypatch.setenv("TNK1_FEATURES_DIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="TNK1_FEATURES_DIR"):
        paths.features_dir(required=True)


def test_feature_file_required_names_the_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TNK1_FEATURES_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="unique_molecules_788k.parquet"):
        paths.feature_file("unique_molecules_788k.parquet", required=True)


def test_verify_features_classifies_each_file(store):
    st = paths.verify_features(directory=store)
    assert st["good.parquet"] == "ok"
    assert st["corrupt.parquet"].startswith("sha256-mismatch")
    assert st["gone.parquet"] == "absent"


def test_verify_features_detects_size_mismatch(store):
    (store / "good.parquet").write_bytes(b"different-length-contents")
    assert paths.verify_features(["good.parquet"], directory=store)["good.parquet"].startswith(
        "size-mismatch"
    )


def test_verify_features_strict_raises_on_corruption(store):
    with pytest.raises(RuntimeError, match="verification failed"):
        paths.verify_features(["corrupt.parquet"], directory=store, strict=True)
    # a verified file passes strict mode
    assert paths.verify_features(["good.parquet"], directory=store, strict=True) == {
        "good.parquet": "ok"
    }


def test_unknown_name_is_flagged(store):
    assert paths.verify_features(["mystery.parquet"], directory=store) == {
        "mystery.parquet": "not-in-manifest"
    }


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        paths.load_manifest(tmp_path)


def test_cli_exit_codes(store, capsys):
    assert verify_main(["good.parquet", "--dir", str(store)]) == 0
    assert verify_main(["corrupt.parquet", "--dir", str(store)]) == 1
    assert verify_main(["gone.parquet", "--dir", str(store)]) == 1
    out = capsys.readouterr().out
    assert "CORRUPT" in out or "missing" in out
