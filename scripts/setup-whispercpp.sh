#!/usr/bin/env bash
# Sets up the viewer's whisper.cpp dictation backend: installs whisper-cli
# (Homebrew's whisper-cpp) when it is missing and downloads a ggml model into
# the viewer cache. Handy's .gguf model does not load in whisper.cpp, so the
# backend uses its own ggml copy.
set -euo pipefail

CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
MODEL_DIR="$CACHE_ROOT/agent-log-viewer/whispercpp"
MODEL="${LLV_WHISPERCPP_MODEL_NAME:-ggml-medium-q8_0.bin}"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL"

if [ -n "${LLV_WHISPERCPP_BIN:-}" ]; then
  BIN="$LLV_WHISPERCPP_BIN"
elif command -v whisper-cli >/dev/null 2>&1; then
  BIN="$(command -v whisper-cli)"
elif command -v brew >/dev/null 2>&1; then
  brew install whisper-cpp
  BIN="$(brew --prefix)/bin/whisper-cli"
else
  echo "whisper-cli not found and Homebrew is unavailable: install whisper.cpp and set LLV_WHISPERCPP_BIN" >&2
  exit 1
fi
[ -x "$BIN" ] || { echo "not an executable: $BIN" >&2; exit 1; }

mkdir -p "$MODEL_DIR"
if [ ! -s "$MODEL_DIR/$MODEL" ]; then
  # Download beside the target and rename, so an interrupted run never leaves
  # a truncated file the viewer would pick up as a model.
  curl -fL --retry 3 -o "$MODEL_DIR/$MODEL.part" "$MODEL_URL"
  mv "$MODEL_DIR/$MODEL.part" "$MODEL_DIR/$MODEL"
fi

echo "whisper.cpp ready: $BIN with $MODEL_DIR/$MODEL"
echo "pick \"whisper.cpp\" in the mic's right-click menu (it is also the default when faster-whisper is not set up)"
