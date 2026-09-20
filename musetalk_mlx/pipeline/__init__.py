# Lazy package init: MuseTalkSession pulls in fusion_mlx/MLX. The spawn-based
# paste child imports musetalk_mlx.pipeline.blending — a package-level session
# import here would drag MLX into the child (seconds of startup + GPU side
# effects). Keep heavy imports behind __getattr__.

__all__ = ["MuseTalkSession"]


def __getattr__(name):
    if name == "MuseTalkSession":
        from .session import MuseTalkSession

        return MuseTalkSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
