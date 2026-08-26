#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=== kokoro-voice-loop 環境準備 ==="

if ! command -v uv &>/dev/null; then
  echo "未偵測到 uv，使用系統 python3 -m venv 建立環境..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install kokoro-onnx soundfile "misaki[zh]"
else
  echo "使用 uv 建立虛擬環境並安裝套件..."
  uv venv .venv --seed
  uv pip install --python .venv/bin/python kokoro-onnx soundfile "misaki[zh]"
fi

mkdir -p models
if [[ ! -f models/kokoro-v1.0.onnx ]]; then
  echo "下載 Kokoro-82M ONNX 模型..."
  curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
fi

if [[ ! -f models/voices-v1.0.bin ]]; then
  echo "下載 Kokoro 音色檔..."
  curl -L -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
fi

echo "執行 selfcheck..."
.venv/bin/python voice_loop.py --selfcheck
echo "=== 設定完成！可直接執行 .venv/bin/python voice_loop.py ==="
