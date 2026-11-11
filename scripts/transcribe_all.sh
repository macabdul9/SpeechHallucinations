#!/usr/bin/env bash
# Transcribe every configured dataset with every model from the paper's runs.
# Extra arguments (for example --device cuda:1 or --num-samples 100) are
# forwarded to each invocation.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

models=(
  "whisper_en openai/whisper-tiny.en"
  "whisper openai/whisper-tiny"
  "whisper_en openai/whisper-small.en"
  "whisper openai/whisper-small"
  "whisper_en openai/whisper-medium.en"
  "whisper openai/whisper-medium"
  "whisper openai/whisper-large"
  "whisper openai/whisper-large-v2"
  "whisper openai/whisper-large-v3"
  "whisper openai/whisper-large-v3-turbo"
  "whisper_en distil-whisper/distil-small.en"
  "whisper_en distil-whisper/distil-medium.en"
  "whisper distil-whisper/distil-large-v2"
  "whisper distil-whisper/distil-large-v3"
  "seamless facebook/hf-seamless-m4t-medium"
  "seamless facebook/hf-seamless-m4t-large"
  "seamlessv2 facebook/seamless-m4t-v2-large"
  "hubert facebook/hubert-large-ls960-ft"
  "wav2vec jonatasgrosman/wav2vec2-large-xlsr-53-english"
  "qwen Qwen/Qwen2-Audio-7B"
  "speecht5 microsoft/speecht5_asr"
  "speechllm skit-ai/speechllm-1.5B"
)

for entry in "${models[@]}"; do
  read -r model_type model <<< "$entry"
  python -m speech_hallucinations.transcribe \
    --model-type "$model_type" \
    --model "$model" \
    "$@"
done
