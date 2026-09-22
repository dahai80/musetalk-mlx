#!/bin/zsh
# Fetch demo assets (video/audio) for musetalk-mlx-cases.
#
# Assets come from a MuseTalk repo clone (data/video, data/audio) — they are
# the official MuseTalk 1.0 demo materials. Not tracked in git (large binaries);
# this script symlinks them in so the cases config resolves.
#
# Usage:
#   bash scripts/fetch_assets.sh [MuseTalk_clone_path]
#   # default: ../../MuseTalk (sibling of the musetalk-mlx repo)
#
# If you don't have the MuseTalk clone, download from the HF mirror:
#   https://hf-mirror.com/TMElyralab/MuseTalk
# and place the files under data/video/ and data/audio/ manually.
set -e
cd "$(dirname "$0")/.."
MUSETALK="${1:-../../MuseTalk}"

if [ ! -d "$MUSETALK/data/video" ] || [ ! -d "$MUSETALK/data/audio" ]; then
  echo "ERROR: MuseTalk data not found at $MUSETALK/data/{video,audio}" >&2
  echo "Clone MuseTalk first:  git clone https://github.com/TMElyralab/MuseTalk $MUSETALK" >&2
  echo "Or download data from https://hf-mirror.com/TMElyralab/MuseTalk into ./data/" >&2
  exit 1
fi

mkdir -p data/video data/audio

# Audio (3 files used by configs)
for f in yongen.wav eng.wav sun.wav; do
  src="$MUSETALK/data/audio/$f"
  [ -e "$src" ] && ln -sf "$(cd "$MUSETALK/data/audio" && pwd)/$f" "data/audio/$f" && echo "audio: $f"
done

# Videos: the 8 README cases use these base videos. Some (yongen, sun) are
# directly in MuseTalk/data/video; others (musk, man, monalisa, sit, video1)
# are static images turned into 5s motion clips — see GUIDELINE.md "Static-
# image cases". This script links the directly-available ones; generate the
# rest per the guideline.
for f in yongen.mp4 sun.mp4; do
  src="$MUSETALK/data/video/$f"
  [ -e "$src" ] && ln -sf "$(cd "$MUSETALK/data/video" && pwd)/$f" "data/video/$f" && echo "video: $f"
done

# sun.mp4 -> sun_real.mp4 (config references sun for the sun1/sun2 cases)
[ -e "data/video/sun.mp4" ] && [ ! -e data/video/sun_real.mp4 ] && ln -sf sun.mp4 data/video/sun_real.mp4

echo ""
echo "Linked available assets. For static-image cases (musk/man/monalisa/sit/video1),"
echo "generate 5s Ken Burns clips per GUIDELINE.md, then run amplify_eyes.py for sun_amp.mp4."
echo ""
echo "Done. Verify:  ls -la data/video/ data/audio/"
