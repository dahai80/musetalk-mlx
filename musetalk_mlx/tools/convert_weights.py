import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def convert(weights_root: Path, dist_dir: Path, dtype: str = "float16") -> None:
    # PyTorch MuseTalk weights -> MLX-native safetensors dist dir (PRD section 8).
    # Input layout (matches MuseTalkPipeline.from_pretrained):
    #   weights_root/sd-vae-ft-mse/  weights_root/MuseTalk/musetalkV15/unet.pth  weights_root/whisper-tiny/
    # Output layout (matches MuseTalkPipeline.from_pretrained_mlx):
    #   dist_dir/config.json  vae.safetensors  unet.safetensors  whisper_encoder.safetensors
    # Dist files must keep native (unwrapped) param names: save_native walks
    # the module tree, and fusion-mlx's pipeline-level SmartConv2d wrapping
    # (default ON, env FUSION_MUSETALK_SMART_CONV) inserts an extra `.conv`
    # level that the strict load_native cannot match back. Opt out here.
    import os

    from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline
    from fusion_mlx.video.musetalk_mlx.utils.weights import save_native

    os.environ["FUSION_MUSETALK_SMART_CONV"] = "0"
    dist_dir.mkdir(parents=True, exist_ok=True)
    pipe = MuseTalkPipeline.from_pretrained(weights_root)
    meta = {"dtype": dtype, "scaling_factor": pipe.scaling_factor}
    (dist_dir / "config.json").write_text(json.dumps(meta, indent=2))
    save_native(pipe.vae, dist_dir / "vae.safetensors")
    save_native(pipe.unet, dist_dir / "unet.safetensors")
    save_native(pipe.whisper_encoder, dist_dir / "whisper_encoder.safetensors")
    log.info("weights converted: %s -> %s", weights_root, dist_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert MuseTalk PyTorch weights to MLX safetensors")
    p.add_argument("--weights", required=True, help="weights root (from_pretrained layout)")
    p.add_argument("--out", required=True, help="output dist dir")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    a = p.parse_args()
    convert(Path(a.weights), Path(a.out), a.dtype)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
