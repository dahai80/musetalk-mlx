__version__ = "0.1.0"


def __getattr__(name):
    # Lazy: MuseTalkSession pulls in fusion_mlx/MLX. The eval subpackage
    # (musetalk_mlx.eval) must stay importable in a torch-only env without
    # the MLX stack (offline eval runs in the openclaw conda env).
    if name == "MuseTalkSession":
        from .pipeline.session import MuseTalkSession

        return MuseTalkSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MuseTalkSession", "__version__"]
