import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SIG_NAME = "manifest.sig"
# Files that make up a distributable MLX weights dir. Anything else in the
# dir (temp downloads, README) is ignored by the integrity check.
WEIGHT_FILES = ("unet.safetensors", "vae.safetensors", "whisper_encoder.safetensors", "config.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_manifest(dist_dir: Path) -> dict:
    # Walk the known weight files and hash each. Unknown files are ignored so a
    # stray .tmp or README does not fail verification (audit P0-4).
    files = {}
    for name in WEIGHT_FILES:
        p = dist_dir / name
        if p.exists():
            files[name] = {"sha256": _sha256(p), "size": p.stat().st_size}
    return {"version": 1, "files": files}


def write_manifest(dist_dir: Path) -> Path:
    manifest = compute_manifest(dist_dir)
    out = dist_dir / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2))
    log.info("manifest written -> %s (%d files)", out, len(manifest["files"]))
    return out


def verify_weights(dist_dir: Path, pubkey_pem: bytes | None = None) -> tuple[bool, str]:
    # Two layers (release-audit P0-4):
    #   1. Integrity (always): manifest.json lists each weight file's sha256.
    #      Verifies no file was corrupted or swapped. Stdlib only.
    #   2. Authenticity (optional): if manifest.sig + a pubkey are present,
    #      verify an Ed25519 signature over the manifest bytes. Requires the
    #      `cryptography` package (optional `sign` extra); without it, layer 2
    #      is skipped with a warning and only integrity is enforced.
    #
    # trust-on-first-load is NOT acceptable for commercial distribution: a
    # tampered weights bundle directly poisons the lip-sync output. Call this
    # before MuseTalkPipeline.from_pretrained_mlx.
    dist_dir = Path(dist_dir)
    manifest_path = dist_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False, f"no {MANIFEST_NAME} in {dist_dir}; weights are trust-on-first-load"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        return False, f"manifest unparseable: {e}"
    files = manifest.get("files", {})
    if not files:
        return False, "manifest lists no files"
    for name, meta in files.items():
        p = dist_dir / name
        if not p.exists():
            return False, f"missing weight file: {name}"
        if p.stat().st_size != meta.get("size"):
            return False, f"size mismatch: {name}"
        if _sha256(p) != meta.get("sha256"):
            return False, f"sha256 mismatch: {name} (tampered or corrupted)"
    log.info("weight integrity OK: %d files verified", len(files))
    # Layer 2: Ed25519 authenticity.
    sig_path = dist_dir / SIG_NAME
    if pubkey_pem is None or not sig_path.exists():
        if pubkey_pem is not None and not sig_path.exists():
            log.warning("pubkey provided but no %s; authenticity not verified", SIG_NAME)
        elif pubkey_pem is None and sig_path.exists():
            log.warning("%s present but no pubkey; authenticity not verified (integrity only)", SIG_NAME)
        return True, "integrity OK (authenticity skipped)"
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        log.warning("cryptography not installed; manifest.sig present but authenticity not verified")
        return True, "integrity OK (cryptography missing, authenticity skipped)"
    try:
        pub = Ed25519PublicKey.from_public_bytes(
            serialization.load_pem_public_key(pubkey_pem).public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
        pub.verify(sig_path.read_bytes(), manifest_path.read_bytes())
        log.info("weight authenticity OK (Ed25519 signature verified)")
        return True, "integrity + authenticity OK"
    except Exception as e:
        return False, f"signature verification FAILED: {e}"
