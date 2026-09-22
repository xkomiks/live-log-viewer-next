#!/usr/bin/env bash
# Sets up the viewer's whisper.cpp dictation backend: installs whisper-cli
# (Homebrew's whisper-cpp) when it is missing and downloads a ggml model into
# the viewer cache. Handy's .gguf model does not load in whisper.cpp, so the
# backend uses its own ggml copy. The Silero VAD model (~1 MB) lands beside
# it: without VAD, whisper-cli turns silence into words (" you").
set -euo pipefail

CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
MODEL_DIR="$CACHE_ROOT/agent-log-viewer/whispercpp"
MODEL="${LLV_WHISPERCPP_MODEL_NAME:-ggml-medium-q8_0.bin}"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL"
VAD_MODEL="ggml-silero-v5.1.2.bin"
VAD_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/main/$VAD_MODEL"

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

# Download beside the target and rename, so an interrupted run never leaves
# a truncated file the viewer would pick up as a model.
fetch() {
  local url="$1" target="$2"
  [ -s "$target" ] && return 0
  curl -fL --retry 3 -o "$target.part" "$url"
  mv "$target.part" "$target"
}

mkdir -p "$MODEL_DIR"
fetch "$MODEL_URL" "$MODEL_DIR/$MODEL"
fetch "$VAD_URL" "$MODEL_DIR/$VAD_MODEL"

echo "whisper.cpp ready: $BIN with $MODEL_DIR/$MODEL (VAD: $MODEL_DIR/$VAD_MODEL)"
echo "pick \"whisper.cpp\" in the mic's right-click menu (it is also the default when faster-whisper is not set up)"
