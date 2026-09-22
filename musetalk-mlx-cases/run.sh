#!/bin/zsh
# MuseTalk MLX demo one-shot runner. Relative paths only — works from a fresh
# clone after the parent musetalk-mlx venv is set up.
set -e
cd "$(dirname "$0")"
# venv + python live one level up (musetalk-mlx repo root).
exec ../.venv/bin/python inference.py "$@"
