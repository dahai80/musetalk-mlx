import argparse
import json
import os
from pathlib import Path

import numpy as np

log = None


def main() -> int:
    # Layered-threshold parity fixtures (PRD 分层阈值): run ONCE in a torch env
    # (torch CPU + diffusers + transformers), NOT a runtime dep. Writes
    # tests/parity/fixtures/*.npz; tests/parity/test_parity.py compares them
    # against the MLX pipeline torch-free at test time.
    import torch
    from diffusers import AutoencoderKL
    from transformers import WhisperModel

    global log
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("gen_parity_fixtures")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default="/tmp/mtlk_weights",
        help="torch weights root (whisper-tiny, sd-vae-ft-mse, MuseTalk unet)",
    )
    p.add_argument(
        "--musetalk-src",
        default=os.environ.get("MT_MUSETALK", str(Path(__file__).resolve().parents[3] / "MuseTalk")),
        help="MuseTalk repo clone root (for the torch UNet source)",
    )
    p.add_argument("--out", default="tests/parity/fixtures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # ---- Whisper-tiny encoder (FR-MLX-002, threshold >=0.98) ----
    mel = rng.standard_normal((1, 80, 3000)).astype(np.float32) * 0.5
    wm = WhisperModel.from_pretrained(f"{a.weights}/whisper-tiny")
    wm.eval()
    with torch.no_grad():
        # MLX encoder stacks ALL hidden states (B,1500,n_layers+1,384);
        # hidden_states tuple length = n_layers+1 matches that stacking.
        o = wm.encoder(input_features=torch.from_numpy(mel), output_hidden_states=True)
        ref = np.stack([h.float().numpy() for h in o.hidden_states], axis=2)
    np.savez_compressed(out / "whisper.npz", mel=mel, ref=ref)
    log.info("whisper ref %s", ref.shape)

    # ---- SD-VAE encode + decode (FR-MLX-004, PSNR>=38 / SSIM>=0.95) ----
    # preprocess_img-format tensor: (1,3,256,256) normalized to [-1,1].
    img = rng.standard_normal((1, 3, 256, 256)).astype(np.float32)
    vae = AutoencoderKL.from_pretrained(f"{a.weights}/sd-vae-ft-mse")
    vae.eval()
    with torch.no_grad():
        lat = vae.encode(torch.from_numpy(img)).latent_dist.mean
        scaling = vae.config.scaling_factor
        np.savez_compressed(out / "vae.npz", img=img, scaling=scaling, lat=lat.float().numpy())
        # decode side: fixed latent -> image (matches decode_latents(z/scaling))
        z = rng.standard_normal((1, 4, 32, 32)).astype(np.float32)
        rec = vae.decode(torch.from_numpy(z / scaling)).sample
    np.savez_compressed(out / "vae_decode.npz", z=z, scaling=scaling, img=rec.float().numpy())
    log.info("vae ref lat %s rec %s scaling=%.4f", lat.shape, rec.shape, scaling)

    # ---- 8-ch UNet (FR-MLX-003, threshold >=0.98) ----
    # PE is applied to the whisper chunk BEFORE the unet (scripts/inference.py:200),
    # matching fusion-mlx generate_faces(apply_pe -> unet).
    import sys

    sys.path.insert(0, a.musetalk_src)
    from musetalk.models.unet import PositionalEncoding, UNet

    u = UNet(
        unet_config=f"{a.weights}/MuseTalk/musetalkV15/unet.json",
        model_path=f"{a.weights}/MuseTalk/musetalkV15/unet.pth",
        device=torch.device("cpu"),
    )
    u.model.eval()
    pe = PositionalEncoding(d_model=384)
    lat8 = rng.standard_normal((1, 8, 32, 32)).astype(np.float32)
    tstep = torch.tensor([0])
    audio = rng.standard_normal((1, 50, 384)).astype(np.float32)
    with torch.no_grad():
        pred = u.model(
            torch.from_numpy(lat8),
            tstep,
            encoder_hidden_states=pe(torch.from_numpy(audio)),
        ).sample
    np.savez_compressed(out / "unet.npz", lat8=lat8, audio=audio, ref=pred.float().numpy())
    log.info("unet ref %s", pred.shape)

    meta = {"torch": torch.__version__, "seed": 0}
    (out / "meta.json").write_text(json.dumps(meta))
    log.info("fixtures written to %s", out)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
