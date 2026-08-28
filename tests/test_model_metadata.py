"""
Tests for the model capability matching tool (app/services/model_metadata.py).

Design reference: docs/plans/模型能力元数据扩展-设计方案.md §2.1-§2.3 / §4.1 / §6.

Contract these tests pin down (M2):

    resolve_model_metadata(provider_id, model_id) -> dict with keys
        context_length        int | None
        max_output_tokens     int | None
        input_modalities      list[str]
        output_modalities     list[str]
        input_price           float | None   ($/M token)
        output_price          float | None   ($/M token)
        cached_input_price    float | None   ($/M token)

    _normalize_model_id(model_id) -> str
    _from_litellm(model_id) -> dict | None
    _from_models_dev(model_id) -> dict | None
    _from_openrouter(model_id) -> dict | None
    _from_provider_models(provider_id, model_id) -> dict | None   (modalities-only)
    _merge_price(values, model_id="", field="") -> float | None
    _merge_modalities(candidates) -> dict  {"input": [...], "output": [...]}
    clear_caches() -> None
    invalidate_model_cache(provider_id, model_id) -> None

Key frozen decisions under test:
- Three sources ALWAYS merge in parallel (models.dev ‖ OpenRouter ‖ LiteLLM). LiteLLM is
  local and free, so it is never demoted to a fallback-only source; without it the
  ">=3 candidates -> median" branch would be dead code.
- Price merge: >=3 -> median; ==2 -> HIGHER price (conservative: under-pricing costs the
  gateway money, and the cheaper number is usually stale) + WARN when they differ >20%;
  ==1 -> single source; 0 -> None.
- Modalities: whitelist union, normalized vocabulary (image<->vision, pdf/file), and
  merged BEFORE price so the embedding check can relax the two-price requirement.
- L2 only ever consumes L1; it must never issue its own network request.
"""
import json
import logging
import time

import pytest

import app.database as db_mod
from app.database import init_db, add_provider
from app.services import model_metadata as mm
from app.services.model_metadata import (
    resolve_model_metadata,
    _normalize_model_id,
    _from_litellm,
    _from_models_dev,
    _from_openrouter,
    _from_provider_models,
    _merge_price,
    _merge_modalities,
)


# -- Fixtures --

@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    db_mod._initialized = False
    init_db(str(tmp_path / "metadata.db"))
    try:
        yield
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized


@pytest.fixture(autouse=True)
def clear_metadata_caches():
    """L1/L2 are module-level: never let one test's cache leak into the next."""
    mm.clear_caches()
    yield
    mm.clear_caches()


@pytest.fixture
def openai_provider():
    add_provider({
        "id": "openai",
        "name": "OpenAI",
        "provider_type": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": [{"id": "gpt-4o", "name": "GPT-4o", "enabled": True}],
    })
    return "openai"


# -- Sample upstream payloads (field positions verified against the live APIs) --

MODELS_DEV_PAYLOAD = {
    "gpt-4o": {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "limit": {"context": 128000, "output": 16384},
        "modalities": {"input": ["text", "image", "audio"], "output": ["text"]},
        # models.dev already publishes $/M token: must NOT be scaled again.
        "cost": {"input": 2.5, "output": 10.0, "cache_read": 1.25},
    },
    "text-embedding-3-small": {
        "id": "text-embedding-3-small",
        "limit": {"context": 8191},
        "modalities": {"input": ["text"], "output": ["embedding"]},
        "cost": {"input": 0.02},
    },
}

# The live models.dev `/api.json` is keyed by PROVIDER, with models nested under
# each provider's `models` key. This fixture mirrors that real top-level shape so we
# can pin the flattening logic in `_fetch_models_dev_catalog_uncached` (which the
# previous flat `MODELS_DEV_PAYLOAD` masked — the bug that made the main source dead).
MODELS_DEV_RAW_PROVIDER_KEYED = {
    "openai": {
        "id": "openai",
        "models": {
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "limit": {"context": 128000, "output": 16384},
                "modalities": {"input": ["text", "image", "audio"], "output": ["text"]},
                "cost": {"input": 2.5, "output": 10.0, "cache_read": 1.25},
            },
        },
    },
    "openrouter": {
        "id": "openrouter",
        "models": {
            "deepseek/deepseek-v4-flash": {
                "id": "deepseek/deepseek-v4-flash",
                "name": "DeepSeek V4 Flash",
                "limit": {"context": 1048576, "output": 128000},
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.028},
            },
        },
    },
}

OPENROUTER_PAYLOAD = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "context_length": 128000,
            "top_provider": {"max_completion_tokens": 16384},
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            # OpenRouter publishes per-token price STRINGS: float() then x1e6.
            "pricing": {
                "prompt": "0.0000025",
                "completion": "0.00001",
                "input_cache_read": "0.00000125",
            },
        }
    ]
}

LITELLM_MODEL_COST = {
    "gpt-4o": {
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        # LiteLLM stores per-token floats: x1e6 to reach $/M.
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1.25e-06,
        "mode": "chat",
        "supports_vision": True,
    },
    "text-embedding-3-small": {
        "max_input_tokens": 8191,
        "input_cost_per_token": 2e-08,
        "mode": "embedding",
    },
}


@pytest.fixture
def all_sources(monkeypatch):
    """Wire all three sources to deterministic in-memory payloads.
    Patches the uncached fetchers so L1 caching works correctly.
    """
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog_uncached", lambda: MODELS_DEV_PAYLOAD)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog_uncached", lambda: OPENROUTER_PAYLOAD)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)

# -- model_id normalization (§2.3) --

@pytest.mark.parametrize("raw,expected", [
    # (1) strip provider prefixes
    ("vcp-gpt-4o", "gpt-4o"),
    ("openai/gpt-4o", "gpt-4o"),
    ("anthropic.claude-3-opus", "claude-3-opus"),
    # (2) strip date / version suffixes
    ("ark-code-latest", "ark-code"),
    ("claude-3-opus-20240229", "claude-3-opus"),
    ("gpt-3.5-turbo-0125", "gpt-3.5-turbo"),
    ("gpt-4-1106-preview", "gpt-4"),
    # (3) lowercase
    ("GPT-4o", "gpt-4o"),
    ("Claude-3-Opus-20240229", "claude-3-opus"),
    # combined
    ("openai/GPT-4-1106-preview", "gpt-4"),
])
def test_normalize_model_id(raw, expected):
    assert _normalize_model_id(raw) == expected


def test_normalize_model_id_is_idempotent():
    once = _normalize_model_id("openai/GPT-4o-2024-05-13")
    assert _normalize_model_id(once) == once


def test_normalize_model_id_handles_empty_and_none():
    assert _normalize_model_id("") == ""
    assert _normalize_model_id(None) == ""


def test_unmatched_model_id_logs_and_does_not_raise(all_sources, openai_provider, caplog):
    """Design §2.3: 无法匹配时记日志、不阻断."""
    with caplog.at_level(logging.INFO):
        result = resolve_model_metadata(openai_provider, "totally-unknown-model-xyz")
    assert result["input_price"] is None
    assert result["output_price"] is None
    assert result["context_length"] is None
    assert result["input_modalities"] == []
    assert result["output_modalities"] == []


# -- Per-source adapters: unit normalization (§6 用例 1) --

def test_from_models_dev_keeps_dollar_per_million_unscaled(monkeypatch):
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: MODELS_DEV_PAYLOAD)
    data = _from_models_dev("gpt-4o")
    assert data["input_price"] == pytest.approx(2.5)
    assert data["output_price"] == pytest.approx(10.0)
    assert data["cached_input_price"] == pytest.approx(1.25)


def test_from_models_dev_reads_limit_and_modalities(monkeypatch):
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: MODELS_DEV_PAYLOAD)
    data = _from_models_dev("gpt-4o")
    assert data["context_length"] == 128000
    assert data["max_output_tokens"] == 16384
    assert set(data["input_modalities"]) >= {"text", "audio"}
    assert data["output_modalities"] == ["text"]



def test_models_dev_fetcher_flattens_provider_keyed_api(monkeypatch):
    """CRITICAL regression: the live `/api.json` is keyed by provider; the
    fetcher must flatten nested `models` into a model-id-keyed dict.

    The old code used a wrong URL AND assumed a flat model-id-keyed dict, so the
    main source was dead in production while tests (which mocked the wrong flat
    shape) stayed green. This pins the real top-level structure."""
    class FakeResp:
        def __init__(self, data):
            self._data = data
        def raise_for_status(self):
            return None
        def json(self):
            return self._data
    monkeypatch.setattr(mm.httpx, "get", lambda *a, **k: FakeResp(MODELS_DEV_RAW_PROVIDER_KEYED))
    flat = mm._fetch_models_dev_catalog_uncached()
    assert "gpt-4o" in flat
    assert "deepseek/deepseek-v4-flash" in flat
    assert flat["gpt-4o"]["limit"]["context"] == 128000
    assert flat["deepseek/deepseek-v4-flash"]["cost"]["input"] == 0.14
    # The flattened catalog feeds _from_models_dev directly.
    assert _from_models_dev("gpt-4o")["context_length"] == 128000
    assert _from_models_dev("deepseek/deepseek-v4-flash")["input_price"] == pytest.approx(0.14)

    # Verify the exact API URL is used (not the broken hypothetical one).
    captured = []
    def spy(*a, **k):
        captured.append(a[0]); return FakeResp(MODELS_DEV_RAW_PROVIDER_KEYED)
    monkeypatch.setattr(mm.httpx, "get", spy)
    mm._fetch_models_dev_catalog_uncached()
    assert captured and captured[0] == "https://models.dev/api.json"

def test_from_openrouter_converts_per_token_strings(monkeypatch):
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: OPENROUTER_PAYLOAD)
    data = _from_openrouter("gpt-4o")
    assert data["input_price"] == pytest.approx(2.5)
    assert data["output_price"] == pytest.approx(10.0)
    assert data["cached_input_price"] == pytest.approx(1.25)


def test_from_openrouter_reads_architecture_modalities(monkeypatch):
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: OPENROUTER_PAYLOAD)
    data = _from_openrouter("gpt-4o")
    assert data["context_length"] == 128000
    assert "text" in data["input_modalities"]
    assert data["output_modalities"] == ["text"]


def test_from_litellm_converts_per_token_floats(monkeypatch):
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)
    data = _from_litellm("gpt-4o")
    assert data["input_price"] == pytest.approx(2.5)
    assert data["output_price"] == pytest.approx(10.0)
    assert data["cached_input_price"] == pytest.approx(1.25)
    assert data["context_length"] == 128000
    assert data["max_output_tokens"] == 16384


def test_all_three_sources_agree_after_unit_normalization(all_sources):
    """§6 用例 1: 2.5 $/M must come out identical from all three sources."""
    prices = [
        _from_models_dev("gpt-4o")["input_price"],
        _from_openrouter("gpt-4o")["input_price"],
        _from_litellm("gpt-4o")["input_price"],
    ]
    for price in prices:
        assert price == pytest.approx(2.5)


def test_from_litellm_matches_via_normalized_id(monkeypatch):
    """LiteLLM keys carry provider prefixes / aliases: lookup goes through normalization."""
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)
    assert _from_litellm("openai/gpt-4o")["input_price"] == pytest.approx(2.5)
    assert _from_litellm("GPT-4o")["input_price"] == pytest.approx(2.5)


def test_adapters_return_none_for_unknown_model(all_sources):
    assert _from_models_dev("no-such-model") is None
    assert _from_openrouter("no-such-model") is None
    assert _from_litellm("no-such-model") is None


# -- Price merge (§2.1 / §6 用例 3) --

def test_merge_price_three_candidates_takes_median():
    assert _merge_price([0.10, 0.20, 0.90]) == pytest.approx(0.20)


def test_merge_price_four_candidates_takes_median():
    assert _merge_price([0.10, 0.20, 0.30, 0.40]) == pytest.approx(0.25)


def test_merge_price_two_candidates_takes_higher():
    """==2 -> take the HIGHER price: conservative, never under-bill the gateway."""
    assert _merge_price([0.15, 0.30]) == pytest.approx(0.30)
    assert _merge_price([0.30, 0.15]) == pytest.approx(0.30)


def test_merge_price_two_candidates_conflict_over_20_percent_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        picked = _merge_price([0.15, 0.30], model_id="gpt-4o", field="input_price")
    assert picked == pytest.approx(0.30)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a >20% two-source conflict must emit a WARN"
    message = " ".join(warnings)
    assert "price conflict" in message
    assert "gpt-4o" in message          # WARN must name the model (四轮 #1)
    assert "0.15" in message and "0.3" in message  # ...and both candidate values


def test_merge_price_two_candidates_within_20_percent_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        picked = _merge_price([1.00, 1.10], model_id="quiet-model", field="input_price")
    assert picked == pytest.approx(1.10)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_merge_price_single_candidate_is_adopted():
    assert _merge_price([0.42]) == pytest.approx(0.42)


def test_merge_price_empty_returns_none():
    assert _merge_price([]) is None


def test_merge_price_ignores_none_candidates():
    """None means 'not published by this source', not 'free'."""
    assert _merge_price([None, 0.5, None]) == pytest.approx(0.5)
    assert _merge_price([None, None]) is None


def test_merge_price_keeps_genuine_zero():
    """A published 0.0 (free model) is real data and must survive."""
    assert _merge_price([0.0]) == pytest.approx(0.0)


def test_merge_price_over_10x_outlier_is_logged_but_still_merged(caplog):
    """§2.1: >10x 偏离仅记日志（接受降级），不丢弃候选、不抛错."""
    with caplog.at_level(logging.WARNING):
        picked = _merge_price([0.1, 0.2, 50.0], model_id="outlier-model", field="input_price")
    assert picked == pytest.approx(0.2)
    assert any("outlier-model" in r.getMessage() for r in caplog.records)


# -- Modalities merge (§2.2 / §6 用例 7) --

def test_merge_modalities_takes_union():
    merged = _merge_modalities([
        {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        {"input_modalities": ["text", "audio"], "output_modalities": ["text"]},
    ])
    assert set(merged["input"]) == {"text", "image", "audio"}
    assert merged["output"] == ["text"]


def test_merge_modalities_normalizes_vision_and_image():
    """image <-> vision are the same capability under different vocabularies."""
    merged = _merge_modalities([
        {"input_modalities": ["text", "vision"], "output_modalities": []},
        {"input_modalities": ["text", "image"], "output_modalities": []},
    ])
    assert len([m for m in merged["input"] if m in {"image", "vision"}]) == 1


def test_merge_modalities_normalizes_pdf_and_file():
    merged = _merge_modalities([
        {"input_modalities": ["text", "pdf"], "output_modalities": []},
        {"input_modalities": ["text", "file"], "output_modalities": []},
    ])
    assert len([m for m in merged["input"] if m in {"pdf", "file"}]) == 1


def test_merge_modalities_drops_values_outside_whitelist():
    """宁漏勿虚: an unrecognized modality must never reach the routing layer."""
    merged = _merge_modalities([
        {"input_modalities": ["text", "telepathy", "<script>"], "output_modalities": ["text"]},
    ])
    assert merged["input"] == ["text"]
    assert "telepathy" not in merged["input"]


def test_merge_modalities_empty_sources_returns_empty_lists():
    merged = _merge_modalities([])
    assert merged == {"input": [], "output": []}


def test_merge_modalities_output_is_deterministically_ordered():
    """Stable order keeps the stored JSON diff-friendly and the UI chips steady."""
    first = _merge_modalities([
        {"input_modalities": ["audio", "text", "image"], "output_modalities": []},
    ])
    second = _merge_modalities([
        {"input_modalities": ["image", "audio", "text"], "output_modalities": []},
    ])
    assert first["input"] == second["input"]


def test_mode_inference_is_display_only_not_routing(monkeypatch):
    """§2.2: mode='chat'/'embedding' 推断仅兜底展示，不参与路由决策.

    An inferred modality must be distinguishable from a declared one, so the
    routing layer can ignore inferences. Only LiteLLM `mode` is available here.
    """
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: {
        "mode-only-model": {"mode": "chat", "input_cost_per_token": 1e-06},
    })
    data = _from_litellm("mode-only-model")
    assert data.get("modalities_inferred") is True



def test_litellm_inferred_modalities_do_not_reach_stored_result(openai_provider, monkeypatch):
    """§2.3 定案 5: LiteLLM mode/supports_vision inference is display-only and
    must NOT pollute the stored (routing) modalities. When only LiteLLM infers
    vision and no source declares it, `image` must NOT appear in the result."""
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {})
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: {"data": []})
    monkeypatch.setattr(mm, "_from_provider_models", lambda pid, mid: None)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: {
        "vision-only-model": {
            "mode": "chat",
            "supports_vision": True,
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
        },
    })
    result = resolve_model_metadata(openai_provider, "vision-only-model")
    assert "image" not in result["input_modalities"]
    assert "text" not in result["input_modalities"]
    # LiteLLM still contributes prices.
    assert result["input_price"] == pytest.approx(1.0)

    # But when a source DECLARES vision, it is kept.
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {
        "vision-declared": {"id": "vision-declared",
            "modalities": {"input": ["text", "image"], "output": ["text"]},
            "cost": {"input": 1.0, "output": 2.0}},
    })
    result2 = resolve_model_metadata(openai_provider, "vision-declared")
    assert "image" in result2["input_modalities"]

# -- Merge order: modalities before price (§2.3 / §6 用例 7) --

def test_embedding_model_keeps_input_price_without_output_price(all_sources, openai_provider):
    """Embedding models publish only an input price; the two-price gate must not apply.

    This only works if modalities are merged BEFORE price, so that the embedding
    check has already seen output_modalities == ['embedding'].
    """
    result = resolve_model_metadata(openai_provider, "text-embedding-3-small")
    assert result["output_modalities"] == ["embedding"]
    assert result["input_price"] == pytest.approx(0.02)
    assert result["output_price"] is None


def test_non_embedding_model_gets_both_prices(all_sources, openai_provider):
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["input_price"] == pytest.approx(2.5)
    assert result["output_price"] == pytest.approx(10.0)


# -- Full resolve: three sources always in parallel (§2.1 定案 4/9) --

def test_resolve_uses_all_three_sources_including_litellm(all_sources, openai_provider, monkeypatch):
    """LiteLLM must be consulted even when both network sources succeed.

    This is the fix for the dead-code bug: if LiteLLM were fallback-only, the
    candidate count could never reach 3 and the median branch would be unreachable.
    """
    called = []
    original = mm._from_litellm
    monkeypatch.setattr(mm, "_from_litellm", lambda mid: (called.append(mid), original(mid))[1])

    resolve_model_metadata(openai_provider, "gpt-4o")
    assert called, "_from_litellm must always participate in the merge"


def test_resolve_reaches_three_candidates_and_uses_median(openai_provider, monkeypatch):
    """Three live sources -> median branch is genuinely reachable."""
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {
        "gpt-4o": {"cost": {"input": 1.0}, "modalities": {"input": ["text"], "output": ["text"]}},
    })
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: {
        "data": [{"id": "openai/gpt-4o", "pricing": {"prompt": "0.000002"},
                  "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}],
    })
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: {
        "gpt-4o": {"input_cost_per_token": 9e-06, "mode": "chat"},
    })
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    # candidates = [1.0, 2.0, 9.0] -> median 2.0 (not mean 4.0, not max 9.0)
    assert result["input_price"] == pytest.approx(2.0)


def test_resolve_context_length_is_a_single_scalar_window(all_sources, openai_provider):
    """§1.3 定案 12: context_length 是单标量窗口，不做 per-modal 分档.

    定案 12 原文说「取典型/最大窗口」，但那句话是相对 per-modal 分档而言的
    (Gemini 文本窗口 vs 含音视频窗口)，不是指跨源取 max() —— 见 P3-5。
    """
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["context_length"] == 128000
    assert isinstance(result["context_length"], int)


# -- P3-5: context fields use source priority, not max() --

def _ctx_sources(monkeypatch, models_dev=None, openrouter=None, litellm=None):
    """Patch the three sources with only the context fields under test.

    Note the models.dev payload is FLAT (keyed by model id): the patched
    `_fetch_models_dev_catalog_uncached` IS the flattener, so a provider-nested
    payload would never be flattened here.
    """
    monkeypatch.setattr(
        mm, "_fetch_models_dev_catalog_uncached",
        (lambda: {"gpt-4o": {"limit": models_dev}}) if models_dev else (lambda: {}),
    )
    monkeypatch.setattr(
        mm, "_fetch_openrouter_catalog_uncached",
        (lambda: {"data": [dict({"id": "openai/gpt-4o"}, **openrouter)]}) if openrouter else (lambda: {}),
    )
    monkeypatch.setattr(
        mm, "_litellm_model_cost",
        (lambda: {"gpt-4o": litellm}) if litellm else (lambda: {}),
    )


def test_context_length_prefers_models_dev_even_when_smaller(openai_provider, monkeypatch):
    """The fix itself: a SMALLER models.dev value must win over larger fallbacks.

    Under the old max() this returned 999000 (LiteLLM's max_input_tokens), which
    silently relabelled an input ceiling as a total window.
    """
    _ctx_sources(
        monkeypatch,
        models_dev={"context": 128000, "output": 16384},
        openrouter={"context_length": 500000, "top_provider": {"max_completion_tokens": 99999}},
        litellm={"max_input_tokens": 999000, "max_output_tokens": 777777},
    )
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["context_length"] == 128000
    assert result["max_output_tokens"] == 16384


def test_context_length_falls_back_to_openrouter(openai_provider, monkeypatch):
    _ctx_sources(
        monkeypatch,
        openrouter={"context_length": 200000, "top_provider": {"max_completion_tokens": 8192}},
        litellm={"max_input_tokens": 999000, "max_output_tokens": 777777},
    )
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["context_length"] == 200000
    assert result["max_output_tokens"] == 8192


def test_context_length_falls_back_to_litellm_last(openai_provider, monkeypatch):
    _ctx_sources(monkeypatch, litellm={"max_input_tokens": 32000, "max_output_tokens": 4096})
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["context_length"] == 32000
    assert result["max_output_tokens"] == 4096


def test_context_fields_fall_back_independently(openai_provider, monkeypatch):
    """Per-field fallback: models.dev may win one field and lose the other."""
    _ctx_sources(
        monkeypatch,
        models_dev={"context": 128000},          # no "output" key
        openrouter={"context_length": 500000, "top_provider": {"max_completion_tokens": 8192}},
    )
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["context_length"] == 128000    # models.dev wins
    assert result["max_output_tokens"] == 8192   # falls through to OpenRouter


def test_resolve_returns_lists_not_json_strings(all_sources, openai_provider):
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert isinstance(result["input_modalities"], list)
    assert isinstance(result["output_modalities"], list)


# -- Network failure / degradation (§6 用例 2) --

def test_network_sources_timeout_degrades_to_litellm(openai_provider, monkeypatch):
    """Both network sources down -> LiteLLM alone, no exception, no 500."""
    def boom():
        raise TimeoutError("models.dev unreachable")

    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", boom)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", boom)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)

    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["input_price"] == pytest.approx(2.5)   # single-source adoption
    assert result["context_length"] == 128000


def test_all_sources_failing_returns_nulls_not_exception(openai_provider, monkeypatch):
    def boom():
        raise TimeoutError("down")

    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", boom)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", boom)
    monkeypatch.setattr(mm, "_litellm_model_cost", boom)

    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["input_price"] is None
    assert result["output_price"] is None
    assert result["cached_input_price"] is None
    assert result["context_length"] is None
    assert result["max_output_tokens"] is None
    assert result["input_modalities"] == []
    assert result["output_modalities"] == []


def test_malformed_source_payload_is_skipped_not_fatal(openai_provider, monkeypatch):
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {"gpt-4o": "not-a-dict"})
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: {"data": "not-a-list"})
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)

    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["input_price"] == pytest.approx(2.5)


def test_source_with_missing_price_fields_still_contributes_modalities(openai_provider, monkeypatch):
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {
        "gpt-4o": {"modalities": {"input": ["text", "audio"], "output": ["text"]}},
    })
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: {"data": []})
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)

    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert "audio" in result["input_modalities"]
    assert result["input_price"] == pytest.approx(2.5)   # only LiteLLM had a price


# -- Provider /models as a modalities-only supplement (§2.2 定案 5 / §6 用例 6) --

def test_provider_models_supplies_modalities(openai_provider, monkeypatch):
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {})
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: {"data": []})
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: {})
    monkeypatch.setattr(
        mm, "_from_provider_models",
        lambda pid, mid: {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
    )
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert set(result["input_modalities"]) == {"text", "image"}


def test_provider_models_never_contributes_a_price(openai_provider, monkeypatch):
    """定案 5: 供应商 /models 不参与价格候选计数.

    Two real price sources + provider /models must still be treated as 2 candidates
    (higher price), never as 3 (median).
    """
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: {
        "gpt-4o": {"cost": {"input": 0.15}, "modalities": {"input": ["text"], "output": ["text"]}},
    })
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: {
        "data": [{"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000003"},
                  "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}],
    })
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: {})
    monkeypatch.setattr(
        mm, "_from_provider_models",
        # A price here must be ignored outright.
        lambda pid, mid: {"input_modalities": ["text"], "output_modalities": ["text"],
                          "input_price": 99.0},
    )
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["input_price"] == pytest.approx(0.30)  # max(0.15, 0.30), never 99.0


def test_provider_models_failure_does_not_affect_degradation(openai_provider, monkeypatch):
    """定案 5: 供应商 /models 不影响降级判定."""
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: MODELS_DEV_PAYLOAD)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: OPENROUTER_PAYLOAD)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)

    def boom(pid, mid):
        raise RuntimeError("provider /models unreachable")

    monkeypatch.setattr(mm, "_from_provider_models", boom)
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["input_price"] == pytest.approx(2.5)
    assert result["context_length"] == 128000


def test_provider_models_missing_fields_is_tolerated(openai_provider, monkeypatch):
    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: MODELS_DEV_PAYLOAD)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: OPENROUTER_PAYLOAD)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)
    monkeypatch.setattr(mm, "_from_provider_models", lambda pid, mid: {})
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    assert result["context_length"] == 128000


# -- Two-level cache (§4.1 / §6 用例 9) --

def test_l1_catalog_is_fetched_once_for_many_models(openai_provider, monkeypatch):
    """L1 全量 JSON 惰性加载 + 24h TTL: N models must not mean N network calls."""
    fetches = []

    def counting_models_dev():
        fetches.append("models.dev")
        return MODELS_DEV_PAYLOAD

    monkeypatch.setattr(mm, "_fetch_models_dev_catalog_uncached", counting_models_dev)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: OPENROUTER_PAYLOAD)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)

    resolve_model_metadata(openai_provider, "gpt-4o")
    resolve_model_metadata(openai_provider, "text-embedding-3-small")
    assert len(fetches) == 1, f"L1 fetched {len(fetches)} times, expected 1"


def test_l2_only_consumes_l1_and_never_hits_the_network(openai_provider, monkeypatch):
    """§4.1 层级约束: L2 只消费 L1，绝不自发发网络请求."""
    network_calls = []

    def forbidden(*args, **kwargs):
        network_calls.append(args)
        raise AssertionError("L2 must not issue a network request")

    monkeypatch.setattr(mm, "_fetch_models_dev_catalog", lambda: MODELS_DEV_PAYLOAD)
    monkeypatch.setattr(mm, "_fetch_openrouter_catalog", lambda: OPENROUTER_PAYLOAD)
    monkeypatch.setattr(mm, "_litellm_model_cost", lambda: LITELLM_MODEL_COST)
    resolve_model_metadata(openai_provider, "gpt-4o")   # warms L1 + L2

    monkeypatch.setattr(mm.httpx, "get", forbidden)
    second = resolve_model_metadata(openai_provider, "gpt-4o")
    assert second["input_price"] == pytest.approx(2.5)
    assert not network_calls


def test_l2_key_uses_normalized_model_id(all_sources, openai_provider, monkeypatch):
    """§6 用例 9: `gpt-4` and `openai/gpt-4` must hit the same L2 entry."""
    resolve_model_metadata(openai_provider, "gpt-4o")

    merges = []
    original = mm._merge_price
    monkeypatch.setattr(mm, "_merge_price", lambda *a, **kw: (merges.append(a), original(*a, **kw))[1])

    resolve_model_metadata(openai_provider, "openai/GPT-4o")
    assert not merges, "normalized alias should have been served from L2, not re-merged"


def test_invalidate_model_cache_forces_a_recompute(all_sources, openai_provider, monkeypatch):
    """写库成功后主动失效对应键（DB 变化是唯一失效源）."""
    resolve_model_metadata(openai_provider, "gpt-4o")
    mm.invalidate_model_cache(openai_provider, "gpt-4o")

    merges = []
    original = mm._merge_price
    monkeypatch.setattr(mm, "_merge_price", lambda *a, **kw: (merges.append(a), original(*a, **kw))[1])

    resolve_model_metadata(openai_provider, "gpt-4o")
    assert merges, "cache invalidation must force a fresh merge"


def test_invalidate_model_cache_accepts_unnormalized_id(all_sources, openai_provider, monkeypatch):
    resolve_model_metadata(openai_provider, "gpt-4o")
    mm.invalidate_model_cache(openai_provider, "openai/GPT-4o")

    merges = []
    original = mm._merge_price
    monkeypatch.setattr(mm, "_merge_price", lambda *a, **kw: (merges.append(a), original(*a, **kw))[1])

    resolve_model_metadata(openai_provider, "gpt-4o")
    assert merges, "invalidation must normalize the id before evicting"


def test_l2_is_keyed_per_provider(all_sources, monkeypatch):
    """Key is (provider_id, normalized model_id): two providers must not share a row."""
    for pid in ("prov-a", "prov-b"):
        add_provider({
            "id": pid,
            "name": pid,
            "provider_type": "openai",
            "api_base": "https://api.example/v1",
            "api_key": "sk-test",
            "enabled": True,
            "models": [{"id": "gpt-4o", "name": "GPT-4o", "enabled": True}],
        })
    resolve_model_metadata("prov-a", "gpt-4o")

    merges = []
    original = mm._merge_price
    monkeypatch.setattr(mm, "_merge_price", lambda *a, **kw: (merges.append(a), original(*a, **kw))[1])

    resolve_model_metadata("prov-b", "gpt-4o")
    assert merges, "a different provider_id must not reuse another provider's L2 entry"


def test_clear_caches_resets_both_levels(all_sources, openai_provider, monkeypatch):
    resolve_model_metadata(openai_provider, "gpt-4o")
    mm.clear_caches()

    fetches = []
    monkeypatch.setattr(
        mm, "_fetch_models_dev_catalog_uncached",
        lambda: (fetches.append(1), MODELS_DEV_PAYLOAD)[1],
    )
    resolve_model_metadata(openai_provider, "gpt-4o")
    assert fetches, "clear_caches must drop L1 as well as L2"


# -- Idempotency (§6 用例 5, tool level) --

def test_resolve_is_idempotent(all_sources, openai_provider):
    first = resolve_model_metadata(openai_provider, "gpt-4o")
    second = resolve_model_metadata(openai_provider, "gpt-4o")
    assert first == second


def test_resolve_result_is_json_serializable(all_sources, openai_provider):
    """The result goes straight into SQLite via json.dumps for the modalities columns."""
    result = resolve_model_metadata(openai_provider, "gpt-4o")
    json.dumps(result)

# -- L1 locking: no network I/O under the state lock (§ P2-1 thundering herd) --


def test_l1_cold_start_stampede_fetches_once():
    """N concurrent cold callers must trigger exactly ONE upstream fetch."""
    import threading

    cache = mm._L1Cache()
    calls = []
    release = threading.Event()

    def slow_fetcher():
        calls.append(1)
        release.wait(5)
        return {"gpt-4o": {"context_length": 128000}}

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.get_models_dev(slow_fetcher)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    # Let every thread pile up on the refresh lock before the fetch returns.
    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(5)

    assert len(calls) == 1, f"cold stampede fetched {len(calls)} times, expected 1"
    assert len(results) == 8
    assert all(r == {"gpt-4o": {"context_length": 128000}} for r in results)


def test_l1_readers_are_not_blocked_by_an_in_flight_refresh():
    """A stale-catalog refresh must not stall readers (the P2-1 regression)."""
    import threading

    cache = mm._L1Cache(ttl=-1)  # everything is instantly stale
    cache._models_dev = {"stale": True}
    cache._models_dev_ts = time.time()

    in_fetch = threading.Event()
    release = threading.Event()

    def blocking_fetcher():
        in_fetch.set()
        release.wait(5)
        return {"fresh": True}

    refresher = threading.Thread(target=lambda: cache.get_models_dev(blocking_fetcher))
    refresher.start()
    assert in_fetch.wait(5), "refresh never started"

    # While the fetch is parked, a reader must return immediately with stale data.
    started = time.time()
    value = cache.get_models_dev(blocking_fetcher)
    elapsed = time.time() - started

    assert value == {"stale": True}, "reader should see the stale catalog, not block"
    assert elapsed < 1.0, f"reader blocked {elapsed:.2f}s behind an in-flight fetch"

    release.set()
    refresher.join(5)
    assert cache._models_dev == {"fresh": True}, "the refresh result must land in the cache"


def test_l1_state_lock_is_never_held_across_the_fetcher():
    """Structural guard: fetcher() must not run while the state lock is held."""
    cache = mm._L1Cache()
    observed = {}

    def introspecting_fetcher():
        # acquire() must succeed -> the state lock is free during the fetch.
        observed["free"] = cache._lock.acquire(blocking=False)
        if observed["free"]:
            cache._lock.release()
        return {"ok": True}

    cache.get_models_dev(introspecting_fetcher)
    assert observed["free"] is True, "state lock was held across fetcher(); P2-1 regressed"


def test_l1_cold_fetch_failure_retries_on_next_call():
    """A failed cold fetch must not poison the cache with a permanent empty catalog."""
    cache = mm._L1Cache()
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("upstream down")
        return {"gpt-4o": {}}

    assert cache.get_models_dev(flaky) == {}
    assert cache.get_models_dev(flaky) == {"gpt-4o": {}}
    assert len(attempts) == 2, "second call must retry after a cold-start failure"


def test_l1_stale_refresh_failure_keeps_serving_the_old_catalog():
    """If a stale refresh fails, the previous catalog must survive."""
    cache = mm._L1Cache(ttl=-1)
    cache._models_dev = {"old": True}
    cache._models_dev_ts = time.time()

    def failing():
        raise RuntimeError("upstream down")

    assert cache.get_models_dev(failing) == {"old": True}
    assert cache.get_models_dev(failing) == {"old": True}


def test_l1_sources_refresh_independently():
    """models.dev and OpenRouter must not serialise against each other."""
    import threading

    cache = mm._L1Cache()
    release = threading.Event()

    def parked_models_dev():
        release.wait(5)
        return {"md": True}

    t = threading.Thread(target=lambda: cache.get_models_dev(parked_models_dev))
    t.start()
    time.sleep(0.2)

    started = time.time()
    assert cache.get_openrouter(lambda: {"or": True}) == {"or": True}
    elapsed = time.time() - started
    assert elapsed < 1.0, f"OpenRouter blocked {elapsed:.2f}s behind a models.dev fetch"

    release.set()
    t.join(5)

