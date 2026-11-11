# Third-party attribution

`src/speech_hallucinations/divergences.py` is a NumPy port of the PyTorch
divergence modules used in our experiments. Those modules were adapted from
domadapter (Kashyap et al.),
[Xlearn](https://github.com/thuml/Xlearn), and the
[Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)
(`dalib`). The central moment discrepancy follows the reference implementation
by Werner Zellinger ([wzell/mann](https://github.com/wzell/mann)) for
Zellinger et al., "Central Moment Discrepancy (CMD) for Domain-Invariant
Representation Learning" (ICLR 2017). This release keeps only CMD and CORAL and
does not depend on `dalib`. The license terms of each upstream project apply to
the adapted code.

`normalize` in `src/speech_hallucinations/wer.py` reproduces
`BasicTextNormalizer` from [openai/whisper](https://github.com/openai/whisper)
(MIT License), which the experiments imported from Hugging Face Transformers.

Model inference uses Hugging Face Transformers, PyTorch, librosa, and
[SONAR](https://github.com/facebookresearch/SONAR). Their licenses apply
independently, as do the terms of each ASR checkpoint and of the OpenAI, Azure
OpenAI, and Gemini APIs.

The evaluation datasets, including the adversarial LibriSpeech examples of
Olivier and Raj (2022), are distributed under their own terms. Consult each
dataset card before use or redistribution.
