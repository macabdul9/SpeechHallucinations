"""Model loaders and single-utterance transcription for the evaluated ASR models.

Audio is passed with ``sampling_rate=16000`` for every model, as in the original
runs, regardless of the dataset's native rate. See ``transcribe --resample-to``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAMPLING_RATE = 16000
QWEN_PROMPT = (
    "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech in this audio word-for-word, "
    "without any additional description or commentary:"
)
SPEECHLLM_INSTRUCTION = "Give me the following information about the audio [Transcript]"

MODEL_TYPES = (
    "whisper",
    "whisper_en",
    "seamless",
    "seamlessv2",
    "wav2vec",
    "hubert",
    "qwen",
    "speecht5",
    "speechllm",
)


@dataclass(slots=True)
class ASRModel:
    model_type: str
    model: Any
    processor: Any = None

    @property
    def device(self):
        return self.model.device


def load_model(model_type: str, model_name_or_path: str, device: str) -> ASRModel:
    import transformers as tf

    if model_type in ("whisper", "whisper_en"):
        processor = tf.WhisperProcessor.from_pretrained(model_name_or_path)
        model = tf.WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    elif model_type in ("seamless", "seamlessv2"):
        processor = tf.AutoProcessor.from_pretrained(model_name_or_path)
        model_class = tf.SeamlessM4Tv2Model if model_type == "seamlessv2" else tf.SeamlessM4TModel
        model = model_class.from_pretrained(model_name_or_path)
    elif model_type in ("wav2vec", "hubert"):
        processor = tf.Wav2Vec2Processor.from_pretrained(model_name_or_path)
        model_class = tf.HubertForCTC if model_type == "hubert" else tf.Wav2Vec2ForCTC
        model = model_class.from_pretrained(model_name_or_path)
    elif model_type == "qwen":
        processor = tf.AutoProcessor.from_pretrained(model_name_or_path)
        model = tf.Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    elif model_type == "speecht5":
        processor = tf.SpeechT5Processor.from_pretrained(model_name_or_path)
        model = tf.SpeechT5ForSpeechToText.from_pretrained(model_name_or_path)
    elif model_type == "speechllm":
        processor = None
        model = tf.AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {MODEL_TYPES}")
    return ASRModel(model_type, model.to(device).eval(), processor)


def transcribe(asr: ASRModel, audio) -> str:
    import torch

    model, processor = asr.model, asr.processor
    with torch.no_grad():
        if asr.model_type in ("whisper", "whisper_en"):
            inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLING_RATE)
            # English-only checkpoints reject a language argument.
            language = {"language": "en"} if asr.model_type == "whisper" else {}
            outputs = model.generate(**inputs.to(asr.device), **language)
            return processor.decode(outputs[0], skip_special_tokens=True)

        if asr.model_type in ("seamless", "seamlessv2"):
            inputs = processor(audios=audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            outputs = model.generate(**inputs.to(asr.device), tgt_lang="eng", generate_speech=False)
            return processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        if asr.model_type in ("wav2vec", "hubert"):
            inputs = processor(
                audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True
            ).to(asr.device)
            predicted_ids = torch.argmax(model(inputs.input_values).logits, dim=-1)
            return processor.batch_decode(predicted_ids)[0]

        if asr.model_type == "qwen":
            inputs = processor(
                text=QWEN_PROMPT, audios=audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
            ).to(asr.device)
            generated = model.generate(**inputs, max_length=512)
            generated = generated[:, inputs.input_ids.size(1) :]
            return processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        if asr.model_type == "speecht5":
            inputs = processor(audio=audio, return_tensors="pt", sampling_rate=SAMPLING_RATE)
            outputs = model.generate(**inputs.to(asr.device))
            return processor.decode(outputs[0], skip_special_tokens=True)

        if asr.model_type == "speechllm":
            response = model.generate_meta(
                audio_tensor=torch.tensor(audio).unsqueeze(0).to(asr.device),
                instruction=SPEECHLLM_INSTRUCTION,
                max_new_tokens=500,
            )
            # The model answers with a JSON-like string: {"Transcript": "..."}.
            return response.split('"')[3]

    raise ValueError(f"Unknown model type {asr.model_type!r}")
