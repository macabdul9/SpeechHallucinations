"""Label reference/hypothesis pairs with an LLM judge.

Adds one column, ``<judge-model>_<prompt name>`` (for example
``gpt-4o-mini_coarsegrained``), to each CSV in place. Re-running the same
command fills only empty rows and rows recorded as ``APIError``.

Credentials come from the environment:
  openai  OPENAI_API_KEY, optional OPENAI_BASE_URL
  azure   AZURE_OPENAI_ENDPOINT (full chat-completions URL with deployment and
          api-version) and AZURE_OPENAI_API_KEY
  gemini  GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from speech_hallucinations.data import REPO_ROOT
from speech_hallucinations.metrics import GRANULARITY_PROMPTS
from speech_hallucinations.wer import iter_csv_files, normalize

logger = logging.getLogger(__name__)

API_ERROR = "APIError"
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "azure": "gpt-4o-mini", "gemini": "gemini-2.0-flash-001"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def format_prompt(template: str, reference, hypothesis, normalize_text: bool = False) -> str:
    reference, hypothesis = str(reference), str(hypothesis)
    if normalize_text:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    return template.format(ground_truth=reference, output=hypothesis)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"Set {name} in the environment")
    return value


def build_request(backend: str, model: str, prompt: str, temperature: float) -> dict:
    """Return the URL, headers, and JSON body for one judge call."""
    messages = [{"role": "user", "content": prompt}]
    if backend == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        return {
            "url": f"{base_url}/chat/completions",
            "headers": {"Authorization": f"Bearer {_require_env('OPENAI_API_KEY')}"},
            "json": {"model": model, "messages": messages, "temperature": temperature},
        }
    if backend == "azure":
        return {
            "url": _require_env("AZURE_OPENAI_ENDPOINT"),
            "headers": {"api-key": _require_env("AZURE_OPENAI_API_KEY")},
            "json": {"model": model, "messages": messages, "temperature": temperature},
        }
    if backend == "gemini":
        return {
            "url": GEMINI_URL.format(model=model),
            "headers": {"x-goog-api-key": _require_env("GEMINI_API_KEY")},
            "json": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
        }
    raise ValueError(f"Unknown backend: {backend}")


def parse_response(backend: str, body: dict) -> str:
    if backend == "gemini":
        return body["candidates"][0]["content"]["parts"][0]["text"]
    return body["choices"][0]["message"]["content"]


def classify(
    backend: str, model: str, prompt: str, temperature: float, max_retries: int, timeout: float
) -> str:
    request = build_request(backend, model, prompt, temperature)
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(**request, timeout=timeout)
            response.raise_for_status()
            return parse_response(backend, response.json()).strip()
        except (requests.RequestException, KeyError, IndexError) as error:
            logger.warning("Judge call failed (attempt %d): %s", attempt + 1, error)
            time.sleep(2**attempt)
    return API_ERROR


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def judge_file(path: Path, column: str, template: str, args: argparse.Namespace) -> None:
    frame = pd.read_csv(path)
    if column not in frame:
        frame[column] = None
    frame[column] = frame[column].astype(object)
    pending = frame.index[frame[column].isna() | (frame[column] == API_ERROR)]
    for count, index in enumerate(tqdm(pending, desc=path.name), start=1):
        prompt = format_prompt(
            template,
            frame.at[index, "reference"],
            frame.at[index, "hypothesis"],
            normalize_text=not args.no_normalize,
        )
        frame.at[index, column] = classify(
            args.backend, args.model, prompt, args.temperature, args.max_retries, args.timeout
        )
        if count % args.checkpoint_every == 0:
            _write_atomic(frame, path)
        time.sleep(args.sleep)
    _write_atomic(frame, path)
    failed = int((frame[column] == API_ERROR).sum())
    logger.info("%s: %d judged, %d API errors", path, len(pending), failed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or result directories")
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default="openai")
    parser.add_argument("--model", help="Defaults to gpt-4o-mini, or gemini-2.0-flash-001")
    parser.add_argument("--granularity", choices=sorted(GRANULARITY_PROMPTS), default="coarse")
    parser.add_argument("--prompt", type=Path, help="Override the granularity's prompt file")
    parser.add_argument("--column", help="Override the output column name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Send raw text; by default both texts pass through Whisper's basic normalizer",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between calls")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.model = args.model or DEFAULT_MODELS[args.backend]
    prompt_path = (
        args.prompt or REPO_ROOT / "prompts" / f"{GRANULARITY_PROMPTS[args.granularity]}.txt"
    )
    template = prompt_path.read_text()
    column = args.column or f"{args.model}_{prompt_path.stem}"
    files = iter_csv_files(args.paths)

    if args.dry_run:
        row = pd.read_csv(files[0]).iloc[0]
        print(f"column: {column}\n")
        print(format_prompt(template, row["reference"], row["hypothesis"], not args.no_normalize))
        return
    for path in files:
        judge_file(path, column, template, args)


if __name__ == "__main__":
    main()
