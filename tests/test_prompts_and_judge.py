from pathlib import Path

import pytest

from speech_hallucinations import judge

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


@pytest.mark.parametrize(
    ("name", "labels"),
    [
        ("coarsegrained", ["Non-Hallucination Error", "Hallucination Error", "No Error"]),
        (
            "finegrained",
            ["Phonetic Error", "Oscillation Error", "Hallucination Error", "Language Error"],
        ),
    ],
)
def test_prompt_files_format_with_braces_in_inputs(name, labels):
    template = (PROMPTS / f"{name}.txt").read_text()
    prompt = judge.format_prompt(template, "ref {with} braces", "hyp {output}")
    assert 'Ground Truth: "ref {with} braces"' in prompt
    assert 'Generated Output: "hyp {output}"' in prompt
    assert "{insert your classification here}" in prompt
    for label in labels:
        assert label in prompt


def test_backends_read_credentials_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    request = judge.build_request("openai", "gpt-4o-mini", "p", 0.0)
    assert request["url"] == "https://api.openai.com/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-openai"
    assert request["json"]["temperature"] == 0.0

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/chat/completions")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure")
    request = judge.build_request("azure", "gpt-4o-mini", "p", 0.0)
    assert request["headers"] == {"api-key": "test-azure"}

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    request = judge.build_request("gemini", "gemini-2.0-flash-001", "p", 0.0)
    assert "gemini-2.0-flash-001:generateContent" in request["url"]
    assert request["json"]["contents"][0]["parts"][0]["text"] == "p"


def test_missing_credentials_fail_before_any_request(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(OSError, match="GEMINI_API_KEY"):
        judge.build_request("gemini", "gemini-2.0-flash-001", "p", 0.0)


def test_response_parsing_per_backend():
    openai_body = {"choices": [{"message": {"content": "No Error"}}]}
    gemini_body = {"candidates": [{"content": {"parts": [{"text": "Hallucination Error"}]}}]}
    assert judge.parse_response("openai", openai_body) == "No Error"
    assert judge.parse_response("gemini", gemini_body) == "Hallucination Error"


def test_failed_calls_are_recorded_as_api_errors(monkeypatch):
    import requests

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)

    def fail(**_):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(judge.requests, "post", fail)
    assert judge.classify("openai", "gpt-4o-mini", "p", 0.0, 1, 1.0) == judge.API_ERROR
