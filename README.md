# Lost in Transcription, Found in Distribution Shift

Code for our Findings of ACL 2025 paper [**Lost in Transcription, Found in Distribution Shift: Demystifying Hallucination in Speech Foundation Models**](https://aclanthology.org/2025.findings-acl.1190/).

**Hanin Atwany\*, Abdul Waheed\*, Rita Singh, Monojit Choudhury, and Bhiksha Raj**  
Carnegie Mellon University, MBZUAI  
\*Equal contribution.

[![Paper](https://img.shields.io/badge/Paper-ACL%20Findings%202025-blue?logo=readthedocs&logoColor=white)](https://aclanthology.org/2025.findings-acl.1190/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.12414-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.12414)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/macabdul9/SpeechHallucinations)

We transcribe 13 English speech datasets with encoder-only, encoder-decoder, and
speech-LLM ASR models, then label every reference/hypothesis pair with an LLM judge (GPT-4o-mini) to obtain a
hallucination error rate (HER) next to WER. Distribution shift from LibriSpeech
is measured with the central moment discrepancy (CMD) between SONAR embeddings,
and we correlate it with WER and HER degradation.

## Installation

Use Python 3.10 or newer and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plots]'
python -m pip install -r requirements/asr.txt     # transcription, perturbation, heuristic baseline
```

The base install (NumPy, pandas, SciPy, jiwer, requests) covers WER, the LLM judge,
divergences, aggregation, and correlation. SONAR embedding extraction needs a
separate environment, because fairseq2 wheels are tied to particular torch and
CUDA builds:

```bash
python -m pip install -e .
python -m pip install -r requirements/sonar.txt
```

The pinned versions are from the experiment environment. GPU stages have not been
re-run from this release, so smoke-test them on your CUDA stack first.

## Datasets

`configs/datasets.json` lists the Hugging Face datasets with their config, split,
and text/audio columns:

| Dataset | Hub ID |
| --- | --- |
| LibriSpeech test-clean (source domain) | `kresnik/librispeech_asr_test` (`clean`) |
| LibriSpeech test-noise | `macabdul9/LibriSpeech_test_noise` |
| GLOBE | `MushanW/GLOBE` |
| SPGISpeech | `kensho/spgispeech` (`S`) |
| Supreme Court speech | `janaab/supreme-court-speech` |
| Primock57 | `macabdul9/Primock57` |
| AMI | `edinburghcstr/ami` (`ihm`) |
| ATCOsim | `Jzuluaga/atcosim_corpus` |
| SLUE-VoxCeleb | `asapp/slue` (`voxceleb`) |
| BERSt | `macabdul9/BERSt` |
| Adversarial | `HaninZ/adversarial` |
| SALT multi-speaker | `macabdul9/salt-multispeaker-eng` |
| SNIPS SLU | `MWilinski/snips_slu_v1.0` |

Some datasets are gated and require `huggingface-cli login`. Result files and
embedding files are keyed by the part of the Hub ID after the slash (for example
`slue`, `ami`). The config order matters for the perturbation pool (see below).

`HaninZ/adversarial` combines the `adv_0.04`, `adv_0.015`, and `adv_0.015_RIR`
splits of `RaphaelOlivier/librispeech_asr_adversarial`. To rebuild it locally:

```bash
python -m speech_hallucinations.adversarial --output-dir data/adversarial
```

This saves a `datasets` directory; push it to the Hub (or point the config at
your copy) to evaluate it.

## Transcribe

```bash
bash scripts/transcribe.sh \
    --model-type whisper \
    --model openai/whisper-large-v3 \
    --num-samples 1000 \
    --output-dir runs/asr
```

This writes `runs/asr/whisper-large-v3/<dataset>.csv` with `audio_id`, `reference`,
and `hypothesis`. Existing files are skipped and interrupted datasets resume
from a checkpoint. `--datasets slue ami` restricts the run to part of the config.
Clips that raise inside model code are skipped and logged, as in the original runs.

Model types: `whisper` (multilingual Whisper and distil-Whisper, decoded with
`language="en"`), `whisper_en` (`.en` checkpoints), `seamless`, `seamlessv2`,
`wav2vec`, `hubert`, `qwen` (Qwen2-Audio), `speecht5`, and `speechllm`.
`scripts/transcribe_all.sh` runs every checkpoint from our experiments in sequence
and forwards extra arguments to each run, for example `--device cuda:1`.

Two defaults reproduce the original runs and are worth checking for new work:

- `--sampling first` takes the first `--num-samples` rows of each split without
  shuffling. The paper describes 1,000 *random* samples per dataset;
  `--sampling shuffle --seed 42` gives a seeded random subset instead.
- Audio is passed to every processor with `sampling_rate=16000` without resampling.
  For datasets stored at other rates, `--resample-to 16000` resamples on load.

## WER and CER

```bash
bash scripts/compute_wer.sh runs/asr
```

This adds per-row `WER`, `CER` (percent), `nErrors`, and `RefLength` columns to every
CSV under the directory, after Whisper's `BasicTextNormalizer`. Dataset-level WER
in the paper and in this code is the mean of the per-row WER values, not a
corpus-level WER.

## Hallucination labels (LLM judge)

```bash
export OPENAI_API_KEY=...
bash scripts/judge.sh runs/asr --granularity coarse
bash scripts/judge.sh runs/asr --granularity fine
```

Each call adds a column named `<model>_<prompt>`, here `gpt-4o-mini_coarsegrained`
and `gpt-4o-mini_finegrained`, to every CSV in place. The prompts are
`prompts/coarsegrained.txt` (Hallucination / Non-Hallucination / No Error) and
`prompts/finegrained.txt` (Hallucination / Language / Oscillation / Phonetic /
No Error). Decoding uses temperature 0, and both texts go through the same
normalizer as WER before they are inserted into the prompt, as in our GPT-4o-mini runs.
Failed calls are retried and then stored as `APIError`. Re-running the command
fills empty and `APIError` rows only. Add `--dry-run` to print the first prompt
without calling an API.

Backends:

| `--backend` | Environment | Default model |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` | `gpt-4o-mini` |
| `azure` | `AZURE_OPENAI_ENDPOINT` (full chat-completions URL including deployment and `api-version`), `AZURE_OPENAI_API_KEY` | `gpt-4o-mini` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash-001` |

These calls are billed by the provider. Our experiments called GPT-4o-mini through an
Azure deployment.

## HER and WER tables

```bash
bash scripts/aggregate.sh --results-dir runs/asr --output-dir runs/tables
```

This writes `results.csv` (model, dataset, n, WER, CER, HER_coarse, HER_fine) and
wide `WER.csv`, `HER_coarse.csv`, and `HER_fine.csv` tables. HER is the percentage of
rows labelled Hallucination. The denominator includes every row, `APIError` and
unparseable answers included. Judge answers are matched to labels after stripping
whitespace, quotes, markdown bold, a leading `Output:`, and a trailing period. Our
analysis scripts counted exact `Hallucination Error` strings, which gives the same
result for answers that contain only the label.

## Distribution shift and alpha

Extract SONAR speech and text embeddings (1,000 clips per dataset, shuffled with
seed 42):

```bash
bash scripts/extract_embeddings.sh --output-dir runs/embeddings --device cuda:0
```

Compute CMD and CORAL from LibriSpeech test-clean to every dataset:

```bash
bash scripts/compute_shifts.sh \
    --embeddings-dir runs/embeddings \
    --embedding speech \
    --output runs/shifts/speech.json
```

Each domain gets a `mean` value (between the two mean embeddings) and an `all` value
(between equal-sized sets, with the larger set subsampled using `--seed`, default 42;
the original subsampling was unseeded). Our figures use `CMD` / `mean` on speech
embeddings. Because the central moments of a single vector are zero, this value
equals the L2 distance between the mean SONAR speech embeddings of the two domains.

Compute WERD, HERD, and alpha:

```bash
bash scripts/correlate.sh \
    --results-dir runs/asr \
    --shifts runs/shifts/speech.json \
    --output-dir runs/correlation
```

WERD is mean WER(target) minus mean WER(source), and HERD is the corresponding HER
difference in percentage points, using the coarse labels by default. alpha is
computed per model across the target datasets of Figure 2
(`LibriSpeech_test_noise GLOBE slue Primock57 ami atcosim_corpus`; change with
`--domains`). The paper defines alpha only as "the correlation coefficient". This
code uses the Pearson coefficient by default, and `--method spearman` is available.
Please confirm which statistic produced the reported 0.91.

## Perturbations

```bash
bash scripts/perturb.sh \
    --model-type whisper \
    --model openai/whisper-large-v3 \
    --output-dir runs/perturbation
bash scripts/judge.sh runs/perturbation --granularity fine
```

`perturbation` takes 1,000 clips per configured dataset (shuffle, seed 42), pools
them, and keeps 1,000 clips of the pool after a second seeded shuffle. Every clip is
transcribed clean (`none`), with Gaussian white noise (std 0.01), time-stretched by a
rate drawn from U(0.9, 1.1), and pitch-shifted by an integer number of semitones in
[-2, 2]. Before concatenation, every dataset is cast to the features of the first
dataset in `configs/datasets.json`, including its audio sampling rate. Stretch and
pitch draws reproduce the original seeded sequence. The original noise draws were
unseeded, and here they are seeded with `--seed`.

## Heuristic baseline

The baseline follows Frieske and Shi: flag a hypothesis when WER > 30, the Flan-T5
(`google/flan-t5-small`) perplexity of the hypothesis is below 200, and the cosine
similarity between mean-pooled Flan-T5 encoder states of reference and hypothesis is
below 0.2.

```bash
bash scripts/heuristic_baseline.sh runs/human_eval/samples.csv --output-dir runs/heuristic
```

This writes `heuristic_labels.csv` and `summary.json`. `--sweep` adds the original
grid (WER 20–80, cosine 0.1–0.5). `--random-hypotheses N` reproduces the benchmark
setup, in which N references are paired with shuffled hypotheses and every pair
counts as a hallucination.

**Preserved scale bug.** The experiment script compared jiwer's WER, a fraction,
against thresholds written as percentages (20–80). The WER condition therefore almost
never held and the heuristic flagged nothing, which is the 0.00 Human–Heuristic
agreement in Table 4. The default `--wer-scale fraction` keeps this behavior.
`--wer-scale percent` compares WER × 100 against the threshold, which is the
intended rule.

## Human-evaluation sample

```bash
bash scripts/sample_human_eval.sh --results-dir runs/asr --output runs/human_eval/samples.csv
```

The command pools all model outputs and keeps rows with WER > 60 where reference and
hypothesis have more than one word and the hypothesis has fewer than 100 words. It
samples 500 rows (seed 42), then permutes the hypotheses of 50 random rows among
themselves to plant known hallucinations (`flag = hallucination`). Result files are
read in sorted order. The original script used directory listing order, so the
drawn rows can differ from the annotated set. Judge the sample with
`scripts/judge.sh` (add `--backend gemini` for the Gemini comparison).

## Figures

```bash
python -m speech_hallucinations.tsne --embeddings-dir runs/embeddings --output-dir runs/tsne
python -m speech_hallucinations.plots shift --correlation-dir runs/correlation
python -m speech_hallucinations.plots model-size --correlation-dir runs/correlation
python -m speech_hallucinations.plots perturbation runs/perturbation/*/perturbation_output.csv
```

Figures are written as PDF under `runs/figures` (and `runs/tsne`). `model-size`
averages WERD and HERD over the Figure 2 datasets for the Whisper and distil-Whisper
checkpoints in `configs/whisper_model_sizes.json`.

## Reproducing the paper

| Result | Commands |
| --- | --- |
| Table 5, appendix WER/CER/HER tables | transcribe → `compute_wer` → `judge` (coarse and fine) → `aggregate` |
| Figure 1 | `extract_embeddings` → `tsne` |
| Figure 2, alpha | `extract_embeddings` → `compute_shifts` → `correlate` → `plots shift` |
| Figure 3 | `perturb` → `judge --granularity fine` → `plots perturbation` |
| Figure 4 | `correlate` → `plots model-size` |
| Table 4 inputs | `sample_human_eval` → `judge` (openai, gemini) and `heuristic_baseline` |

Results depend on API model versions and on the Hub dataset revisions, so expect
small differences from the published numbers.

## Layout

| Path | Contents |
| --- | --- |
| `src/speech_hallucinations/` | Transcription, WER, LLM judge, divergences, analysis, and plotting |
| `scripts/` | Bash launchers for the pipeline stages |
| `configs/` | Dataset list, display names, Whisper model sizes |
| `prompts/` | Coarse- and fine-grained judge prompts |
| `requirements/` | GPU and SONAR environments |
| `tests/` | CPU tests on synthetic inputs |

No transcriptions, judge outputs, embeddings, human annotations, or figures are
included. Commands write to `runs/` by default, which Git ignores.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for adapted code. A
repository-wide license has not been selected yet. Datasets and models remain under
their own licenses.

## Not included

- **Table 4 agreement scores.** The paper reports "raw agreement" between human
  annotators but does not specify how the model–human scores or the ± values were
  computed, so no agreement script is provided. The human annotations are not
  released.
- Verbalized confidence analysis (Tables 2–3), normalized-versus-orthographic label
  transitions (Tables 6–7), the additional perturbations in Table 8 (cross-lingual
  noise, distortion, echo, pub noise, reverberation), and the Canary and Parakeet
  (NeMo) results. The original code for these is not part of this release.
- CMD here is the implementation behind our figures: it sums L2 distances of the
  mean and of central moments 2–5, with no range scaling and no squaring. This
  differs from Equation 1 in the paper, which averages squared mean differences over
  layers. With the `mean` variant used in the figures, the value is the L2 distance
  between mean embeddings.
- Our Gemini runs sent unnormalized text, and one helper used temperature 0.1. Use
  `--no-normalize --temperature 0.1` to match those settings.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
ruff format --check src tests
pytest
```

Tests run on CPU without torch, transformers, or librosa. Tests for librosa-based
perturbations are skipped when librosa is missing.

## Acknowledgements

The divergence code is adapted from domadapter, Xlearn, and the Transfer Learning
Library (`dalib`), and the CMD implementation follows Zellinger et al. The WER
normalizer is Whisper's `BasicTextNormalizer`. We use SONAR for embeddings and the
adversarial LibriSpeech set of Olivier and Raj (2022).

## Citation

```bibtex
@inproceedings{atwany-etal-2025-lost,
    title = "Lost in Transcription, Found in Distribution Shift: Demystifying Hallucination in Speech Foundation Models",
    author = "Atwany, Hanin  and
      Waheed, Abdul  and
      Singh, Rita  and
      Choudhury, Monojit  and
      Raj, Bhiksha",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1190/",
    doi = "10.18653/v1/2025.findings-acl.1190",
    pages = "23181--23203",
    ISBN = "979-8-89176-256-5"
}
```
