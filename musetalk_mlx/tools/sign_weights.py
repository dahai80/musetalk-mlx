import argparse
import logging
from pathlib import Path

from ..utils.weights_verify import write_manifest

log = logging.getLogger(__name__)


def _gen_keypair(priv_path: Path, pub_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_path.write_bytes(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    priv_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    priv_path.chmod(0o600)
    log.info("Ed25519 keypair generated: %s / %s", priv_path, pub_path)


def sign(dist_dir: Path, priv_path: Path) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    manifest = write_manifest(dist_dir)
    priv = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise SystemExit(f"{priv_path} is not an Ed25519 private key")
    sig = priv.sign(manifest.read_bytes())
    sig_path = dist_dir / "manifest.sig"
    sig_path.write_bytes(sig)
    log.info("manifest signed -> %s (%d bytes)", sig_path, len(sig))
    return sig_path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Ed25519 keypair and sign the weights manifest")
    p.add_argument("--weights", required=True, help="MLX weights dir to sign")
    p.add_argument("--priv-key", required=True, help="private key PEM (created if absent)")
    p.add_argument("--pub-key", required=True, help="public key PEM (created if absent)")
    a = p.parse_args()
    priv, pub = Path(a.priv_key), Path(a.pub_key)
    if not priv.exists():
        _gen_keypair(priv, pub)
    sign(Path(a.weights), priv)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
