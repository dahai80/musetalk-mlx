import json
from pathlib import Path

import pytest

from musetalk_mlx.utils.weights_verify import (
    compute_manifest,
    verify_weights,
    write_manifest,
)


def _make_dist(d: Path, body: str = "x") -> None:
    for name in ("config.json", "unet.safetensors", "vae.safetensors", "whisper_encoder.safetensors"):
        (d / name).write_text(body * 10)


def test_manifest_roundtrip(tmp_path):
    _make_dist(tmp_path)
    m = write_manifest(tmp_path)
    data = json.loads(m.read_text())
    assert data["version"] == 1
    assert set(data["files"]) == {
        "config.json",
        "unet.safetensors",
        "vae.safetensors",
        "whisper_encoder.safetensors",
    }
    ok, reason = verify_weights(tmp_path)
    assert ok, reason


def test_verify_detects_tamper(tmp_path):
    _make_dist(tmp_path)
    write_manifest(tmp_path)
    Path(tmp_path / "unet.safetensors").write_text("TAMPERED")
    ok, reason = verify_weights(tmp_path)
    assert not ok
    assert "unet.safetensors" in reason


def test_verify_detects_missing_file(tmp_path):
    _make_dist(tmp_path)
    write_manifest(tmp_path)
    (tmp_path / "vae.safetensors").unlink()
    ok, reason = verify_weights(tmp_path)
    assert not ok
    assert "vae.safetensors" in reason


def test_verify_missing_manifest(tmp_path):
    _make_dist(tmp_path)
    ok, reason = verify_weights(tmp_path)
    assert not ok
    assert "no manifest" in reason


def test_sign_and_verify_authenticity(tmp_path):
    pytest.importorskip("cryptography")
    from musetalk_mlx.tools.sign_weights import _gen_keypair, sign

    _make_dist(tmp_path)
    priv, pub = tmp_path / "k.pem", tmp_path / "k.pub.pem"
    _gen_keypair(priv, pub)
    sign(tmp_path, priv)
    assert (tmp_path / "manifest.sig").exists()
    pubkey = pub.read_bytes()
    ok, reason = verify_weights(tmp_path, pubkey_pem=pubkey)
    assert ok, reason
    assert "authenticity" in reason


def test_sign_rejects_tamper_after_sign(tmp_path):
    pytest.importorskip("cryptography")
    from musetalk_mlx.tools.sign_weights import _gen_keypair, sign

    _make_dist(tmp_path)
    priv, pub = tmp_path / "k.pem", tmp_path / "k.pub.pem"
    _gen_keypair(priv, pub)
    sign(tmp_path, priv)
    # Tamper after signing: sha256 mismatch caught at integrity layer.
    Path(tmp_path / "unet.safetensors").write_text("EVIL")
    ok, reason = verify_weights(tmp_path, pubkey_pem=pub.read_bytes())
    assert not ok
    assert "unet.safetensors" in reason


def test_compute_manifest_ignores_unknown_files(tmp_path):
    _make_dist(tmp_path)
    (tmp_path / "README.md").write_text("hi")
    (tmp_path / "unet.safetensors.part").write_text("partial")
    m = compute_manifest(tmp_path)
    assert "README.md" not in m["files"]
    assert "unet.safetensors.part" not in m["files"]
