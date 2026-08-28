"""
Discovery integration tests for model capability metadata (M3).

Design reference: docs/plans/模型能力元数据扩展-设计方案.md §2.4 / §4.2 / §6 用例 5 / 10.

Contract these tests pin down:
- refresh_provider_models fills capability fields for `source='auto'` rows only.
- `source='custom'` rows are user-owned: refresh must never touch their capability
  fields, even when the same model_id also appears upstream.
- The extended INSERT (discovery.py:174) must write metadata for newly discovered
  models, not rely on column DEFAULTs.
- Batch prefetch happens OUTSIDE the per-model loop (asyncio.gather + Semaphore(4)),
  so N models do not mean N serialized awaits on one connection.
- A single model's resolve failure is swallowed per-model (try/except): it must not
  roll back the whole refresh, and the field stays NULL for the next retry.
- per-provider asyncio.Lock serializes concurrent refresh of the same provider.
"""
import asyncio
import json

import pytest

import app.database as db_mod
from app.database import init_db, get_db, add_provider, get_provider
from app.services import discovery as disc
from app.services.discovery import refresh_provider_models, refresh_all_providers


GPT4O_METADATA = {
    "context_length": 128000,
    "max_output_tokens": 16384,
    "input_modalities": ["text", "image"],
    "output_modalities": ["text"],
    "input_price": 2.5,
    "output_price": 10.0,
    "cached_input_price": 1.25,
}

EMPTY_METADATA = {
    "context_length": None,
    "max_output_tokens": None,
    "input_modalities": [],
    "output_modalities": [],
    "input_price": None,
    "output_price": None,
    "cached_input_price": None,
}


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    db_mod._initialized = False
    init_db(str(tmp_path / "discovery_metadata.db"))
    try:
        yield
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized


def _set_source(pid, model_id, source):
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET source = ? WHERE provider_id = ? AND model_id = ?",
            (source, pid, model_id),
        )


def _row(pid, model_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ? AND model_id = ?",
            (pid, model_id),
        ).fetchone()


def _write_metadata(pid, model_id, **fields):
    """Directly seed capability columns, bypassing the service layer."""
    assignments = ", ".join(f"{k} = ?" for k in fields)
    values = [
        json.dumps(v) if k.endswith("_modalities") else v
        for k, v in fields.items()
    ]
    with get_db() as db:
        db.execute(
            f"UPDATE provider_models SET {assignments} WHERE provider_id = ? AND model_id = ?",
            (*values, pid, model_id),
        )


def _add_provider(pid, models=()):
    add_provider({
        "id": pid,
        "name": pid,
        "provider_type": "openai",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": list(models),
    })
    return pid


def _stub_discovery(monkeypatch, models):
    async def discover(_provider_id):
        return list(models)

    monkeypatch.setattr("app.services.discovery.discover_models", discover)


def _stub_resolver(monkeypatch, mapping, calls=None):
    """Patch the metadata resolver as discovery sees it."""
    def resolve(provider_id, model_id):
        if calls is not None:
            calls.append((provider_id, model_id))
        return mapping.get(model_id, dict(EMPTY_METADATA))

    monkeypatch.setattr("app.services.discovery.resolve_model_metadata", resolve)
    return resolve


# -- auto models get filled --

@pytest.mark.asyncio
async def test_refresh_fills_metadata_for_newly_discovered_auto_model(monkeypatch):
    """The extended INSERT must carry metadata, not rely on column DEFAULTs."""
    _add_provider("auto-new")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    result = await refresh_provider_models("auto-new")
    assert result["added"] == 1

    row = _row("auto-new", "gpt-4o")
    assert row["source"] == "auto"
    assert row["context_length"] == 128000
    assert row["max_output_tokens"] == 16384
    assert row["input_price"] == pytest.approx(2.5)
    assert row["output_price"] == pytest.approx(10.0)
    assert row["cached_input_price"] == pytest.approx(1.25)
    assert json.loads(row["input_modalities"]) == ["text", "image"]
    assert json.loads(row["output_modalities"]) == ["text"]


@pytest.mark.asyncio
async def test_refresh_updates_metadata_on_existing_auto_model(monkeypatch):
    """auto rows are refresh-owned: an idempotent upsert always rewrites them."""
    _add_provider("auto-existing", [{"id": "gpt-4o", "name": "Old Name", "enabled": True}])
    _set_source("auto-existing", "gpt-4o", "auto")
    _write_metadata("auto-existing", "gpt-4o", context_length=8000, input_price=99.0)

    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await refresh_provider_models("auto-existing")

    row = _row("auto-existing", "gpt-4o")
    assert row["context_length"] == 128000
    assert row["input_price"] == pytest.approx(2.5)



@pytest.mark.asyncio
async def test_refresh_does_not_wipe_filled_metadata_when_sources_degrade(monkeypatch):
    """§4.2 保留旧值: when every source degrades to empty (all three down + LiteLLM
    has no entry), refresh must NOT overwrite previously-filled capability data
    with NULLs — old values survive until a future refresh succeeds."""
    _add_provider("degrade", [{"id": "gpt-4o", "name": "GPT-4o", "enabled": True}])
    _set_source("degrade", "gpt-4o", "auto")
    _write_metadata("degrade", "gpt-4o",
        context_length=128000, input_price=2.5, input_modalities=["text", "image"])

    # This refresh resolves to all-empty metadata for the model.
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {})  # every model → EMPTY_METADATA
    await refresh_provider_models("degrade")

    row = _row("degrade", "gpt-4o")
    assert row["context_length"] == 128000   # NOT wiped to NULL
    assert row["input_price"] == pytest.approx(2.5)
    assert json.loads(row["input_modalities"]) == ["text", "image"]

    # A later refresh that DOES resolve refills/updates normally.
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})
    await refresh_provider_models("degrade")
    assert _row("degrade", "gpt-4o")["input_price"] == pytest.approx(2.5)

@pytest.mark.asyncio
async def test_refresh_stores_modalities_as_json_text(monkeypatch):
    _add_provider("json-shape")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await refresh_provider_models("json-shape")

    row = _row("json-shape", "gpt-4o")
    assert isinstance(row["input_modalities"], str)
    assert json.loads(row["input_modalities"]) == ["text", "image"]


@pytest.mark.asyncio
async def test_refresh_leaves_null_when_no_source_matched(monkeypatch):
    """§6 用例 10: 采集失败留 NULL, 下次 refresh 重试."""
    _add_provider("auto-null")
    _stub_discovery(monkeypatch, [{"id": "unknown-model", "name": "Unknown"}])
    _stub_resolver(monkeypatch, {})   # every model resolves to EMPTY_METADATA

    await refresh_provider_models("auto-null")

    row = _row("auto-null", "unknown-model")
    assert row["context_length"] is None
    assert row["input_price"] is None
    assert row["input_modalities"] == "[]"


@pytest.mark.asyncio
async def test_null_metadata_is_retried_on_next_refresh(monkeypatch):
    """First refresh finds nothing; the second one fills the fields in."""
    _add_provider("retry")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])

    _stub_resolver(monkeypatch, {})
    await refresh_provider_models("retry")
    assert _row("retry", "gpt-4o")["input_price"] is None

    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})
    await refresh_provider_models("retry")
    assert _row("retry", "gpt-4o")["input_price"] == pytest.approx(2.5)


# -- custom models are untouched --

@pytest.mark.asyncio
async def test_refresh_never_overwrites_custom_metadata(monkeypatch):
    """custom rows are user-owned: their capability fields survive refresh verbatim."""
    _add_provider("custom-guard", [{"id": "gpt-4o", "name": "Hand Tuned", "enabled": True}])
    _set_source("custom-guard", "gpt-4o", "custom")
    _write_metadata(
        "custom-guard", "gpt-4o",
        context_length=32000,
        max_output_tokens=4096,
        input_price=1.0,
        output_price=3.0,
        input_modalities=["text"],
        output_modalities=["text"],
    )

    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await refresh_provider_models("custom-guard")

    row = _row("custom-guard", "gpt-4o")
    assert row["source"] == "custom"
    assert row["model_name"] == "Hand Tuned"          # existing custom guard
    assert row["context_length"] == 32000             # metadata guard
    assert row["max_output_tokens"] == 4096
    assert row["input_price"] == pytest.approx(1.0)
    assert row["output_price"] == pytest.approx(3.0)
    assert json.loads(row["input_modalities"]) == ["text"]


@pytest.mark.asyncio
async def test_refresh_does_not_fill_empty_custom_metadata(monkeypatch):
    """Even NULL custom fields stay NULL: custom sync is user-triggered only (§2.4)."""
    _add_provider("custom-empty", [{"id": "gpt-4o", "name": "Manual", "enabled": True}])
    _set_source("custom-empty", "gpt-4o", "custom")

    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await refresh_provider_models("custom-empty")

    row = _row("custom-empty", "gpt-4o")
    assert row["context_length"] is None
    assert row["input_price"] is None
    assert row["input_modalities"] == "[]"


@pytest.mark.asyncio
async def test_custom_and_auto_models_coexist_without_cross_contamination(monkeypatch):
    _add_provider("mixed", [
        {"id": "custom-model", "name": "Manual", "enabled": True},
        {"id": "auto-model", "name": "Auto", "enabled": True},
    ])
    _set_source("mixed", "custom-model", "custom")
    _set_source("mixed", "auto-model", "auto")
    _write_metadata("mixed", "custom-model", context_length=1234, input_price=0.5)

    _stub_discovery(monkeypatch, [
        {"id": "custom-model", "name": "Upstream Manual"},
        {"id": "auto-model", "name": "Upstream Auto"},
    ])
    _stub_resolver(monkeypatch, {
        "custom-model": GPT4O_METADATA,
        "auto-model": GPT4O_METADATA,
    })

    await refresh_provider_models("mixed")

    custom = _row("mixed", "custom-model")
    auto = _row("mixed", "auto-model")
    assert custom["context_length"] == 1234
    assert custom["input_price"] == pytest.approx(0.5)
    assert auto["context_length"] == 128000
    assert auto["input_price"] == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_custom_model_metadata_survives_stale_cleanup(monkeypatch):
    """custom rows are never deleted, so their metadata must persist too."""
    _add_provider("stale", [{"id": "custom-only", "name": "Manual Only", "enabled": True}])
    _set_source("stale", "custom-only", "custom")
    _write_metadata("stale", "custom-only", context_length=4096, input_price=0.9)

    _stub_discovery(monkeypatch, [{"id": "other-model", "name": "Other"}])
    _stub_resolver(monkeypatch, {"other-model": GPT4O_METADATA})

    result = await refresh_provider_models("stale")
    assert result["removed"] == 0

    row = _row("stale", "custom-only")
    assert row is not None
    assert row["context_length"] == 4096
    assert row["input_price"] == pytest.approx(0.9)


# -- Batch prefetch (§4.2) --

@pytest.mark.asyncio
async def test_metadata_is_resolved_once_per_discovered_model(monkeypatch):
    _add_provider("batch")
    models = [{"id": f"m{i}", "name": f"M{i}"} for i in range(10)]
    _stub_discovery(monkeypatch, models)
    calls = []
    _stub_resolver(monkeypatch, {}, calls=calls)

    await refresh_provider_models("batch")

    assert len(calls) == 10
    assert len(set(calls)) == 10, f"duplicate resolves: {calls}"


@pytest.mark.asyncio
async def test_metadata_prefetch_is_concurrent_not_serialized(monkeypatch):
    """§4.2: 勿在 for 循环内逐模型 await; prefetch via asyncio.gather outside the loop."""
    _add_provider("concurrent")
    models = [{"id": f"m{i}", "name": f"M{i}"} for i in range(8)]
    _stub_discovery(monkeypatch, models)

    in_flight = 0
    peak = 0

    async def slow_resolve(provider_id, model_id):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return dict(GPT4O_METADATA)

    monkeypatch.setattr("app.services.discovery.resolve_model_metadata_async", slow_resolve)

    await refresh_provider_models("concurrent")
    assert peak > 1, "metadata prefetch ran fully serialized"


@pytest.mark.asyncio
async def test_metadata_prefetch_respects_the_concurrency_limit(monkeypatch):
    """Semaphore(4): prefetch must not open an unbounded number of connections."""
    _add_provider("limited")
    models = [{"id": f"m{i}", "name": f"M{i}"} for i in range(20)]
    _stub_discovery(monkeypatch, models)

    in_flight = 0
    peak = 0

    async def slow_resolve(provider_id, model_id):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.005)
        in_flight -= 1
        return dict(GPT4O_METADATA)

    monkeypatch.setattr("app.services.discovery.resolve_model_metadata_async", slow_resolve)

    await refresh_provider_models("limited")
    assert peak <= 4, f"prefetch concurrency {peak} exceeded the Semaphore(4) limit"


# -- Per-model failure isolation (§4.2 降级) --

@pytest.mark.asyncio
async def test_single_model_resolve_failure_does_not_abort_refresh(monkeypatch):
    """§4.2: 绝不让单模型失败使整个 refresh 回滚."""
    _add_provider("isolate")
    _stub_discovery(monkeypatch, [
        {"id": "good-model", "name": "Good"},
        {"id": "bad-model", "name": "Bad"},
        {"id": "another-good", "name": "Another"},
    ])

    def resolve(provider_id, model_id):
        if model_id == "bad-model":
            raise RuntimeError("resolver exploded")
        return dict(GPT4O_METADATA)

    monkeypatch.setattr("app.services.discovery.resolve_model_metadata", resolve)

    result = await refresh_provider_models("isolate")

    assert "error" not in result
    assert result["added"] == 3, "all three models must still be persisted"
    assert _row("isolate", "good-model")["input_price"] == pytest.approx(2.5)
    assert _row("isolate", "another-good")["input_price"] == pytest.approx(2.5)
    assert _row("isolate", "bad-model")["input_price"] is None


@pytest.mark.asyncio
async def test_resolve_failure_preserves_previous_values(monkeypatch):
    """A failed resolve must keep the old value, never blank it out."""
    _add_provider("preserve", [{"id": "gpt-4o", "name": "GPT-4o", "enabled": True}])
    _set_source("preserve", "gpt-4o", "auto")
    _write_metadata("preserve", "gpt-4o", context_length=64000, input_price=1.5)

    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])

    def resolve(provider_id, model_id):
        raise TimeoutError("all sources down")

    monkeypatch.setattr("app.services.discovery.resolve_model_metadata", resolve)

    result = await refresh_provider_models("preserve")
    assert "error" not in result

    row = _row("preserve", "gpt-4o")
    assert row["context_length"] == 64000
    assert row["input_price"] == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_discovery_failure_still_reports_error_dict(monkeypatch):
    """Metadata handling must not swallow the outer discovery failure contract."""
    _add_provider("outer-fail")

    async def discover(_provider_id):
        raise RuntimeError("upstream /models down")

    monkeypatch.setattr("app.services.discovery.discover_models", discover)
    _stub_resolver(monkeypatch, {})

    result = await refresh_provider_models("outer-fail")
    assert result["error"]
    assert result["count"] == 0
    assert result["added"] == 0


# -- Idempotency (§6 用例 5) --

@pytest.mark.asyncio
async def test_two_consecutive_refreshes_are_idempotent(monkeypatch):
    _add_provider("idem")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await refresh_provider_models("idem")
    first = dict(_row("idem", "gpt-4o"))
    second_result = await refresh_provider_models("idem")
    second = dict(_row("idem", "gpt-4o"))

    assert second_result["added"] == 0
    assert second_result["removed"] == 0
    for column in (
        "context_length", "max_output_tokens", "input_modalities", "output_modalities",
        "input_price", "output_price", "cached_input_price",
    ):
        assert first[column] == second[column], f"{column} drifted between refreshes"


@pytest.mark.asyncio
async def test_refresh_does_not_duplicate_rows(monkeypatch):
    _add_provider("no-dup")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    for _ in range(3):
        await refresh_provider_models("no-dup")

    with get_db() as db:
        count = db.execute(
            "SELECT COUNT(*) AS n FROM provider_models WHERE provider_id = 'no-dup'"
        ).fetchone()["n"]
    assert count == 1


# -- per-provider asyncio.Lock (§4.2 定案 14 / §6 用例 5) --

@pytest.mark.asyncio
async def test_concurrent_refresh_of_same_provider_is_serialized(monkeypatch):
    """§4.2: per-provider asyncio.Lock 保护同 provider 的 refresh/sync."""
    _add_provider("locked")

    concurrent = 0
    peak = 0

    async def discover(_provider_id):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return [{"id": "gpt-4o", "name": "GPT-4o"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await asyncio.gather(
        refresh_provider_models("locked"),
        refresh_provider_models("locked"),
    )
    assert peak == 1, f"two refreshes of the same provider overlapped (peak={peak})"


@pytest.mark.asyncio
async def test_concurrent_refresh_of_different_providers_runs_in_parallel(monkeypatch):
    """The lock must be per-provider, not global: unrelated providers stay concurrent."""
    _add_provider("prov-a")
    _add_provider("prov-b")

    concurrent = 0
    peak = 0

    async def discover(_provider_id):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return [{"id": "gpt-4o", "name": "GPT-4o"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await asyncio.gather(
        refresh_provider_models("prov-a"),
        refresh_provider_models("prov-b"),
    )
    assert peak == 2, "different providers must not block each other"


@pytest.mark.asyncio
async def test_concurrent_refresh_produces_no_dirty_metadata(monkeypatch):
    """After two racing refreshes the row must hold one consistent snapshot."""
    _add_provider("no-dirty")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await asyncio.gather(*(refresh_provider_models("no-dirty") for _ in range(4)))

    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM provider_models WHERE provider_id = 'no-dirty'"
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["context_length"] == 128000
    assert row["input_price"] == pytest.approx(2.5)
    assert json.loads(row["input_modalities"]) == ["text", "image"]


@pytest.mark.asyncio
async def test_lock_is_released_after_a_failed_refresh(monkeypatch):
    """A raising discover must not leave the per-provider lock held forever."""
    _add_provider("lock-release")

    async def failing_discover(_provider_id):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.discovery.discover_models", failing_discover)
    _stub_resolver(monkeypatch, {})

    first = await refresh_provider_models("lock-release")
    assert first["error"]

    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})
    second = await asyncio.wait_for(refresh_provider_models("lock-release"), timeout=5)
    assert second["added"] == 1


@pytest.mark.asyncio
async def test_refresh_all_providers_still_fills_metadata(monkeypatch):
    """refresh_all_providers uses Semaphore(4); metadata must ride along."""
    for pid in ("all-a", "all-b", "all-c"):
        _add_provider(pid)

    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    results = await refresh_all_providers()
    assert len(results) == 3
    for pid in ("all-a", "all-b", "all-c"):
        assert _row(pid, "gpt-4o")["input_price"] == pytest.approx(2.5)


# -- Read path (§4.4) --

@pytest.mark.asyncio
async def test_get_provider_returns_metadata_as_python_types(monkeypatch):
    """_model_from_row must json.loads the modalities columns for API consumers."""
    _add_provider("read-path")
    _stub_discovery(monkeypatch, [{"id": "gpt-4o", "name": "GPT-4o"}])
    _stub_resolver(monkeypatch, {"gpt-4o": GPT4O_METADATA})

    await refresh_provider_models("read-path")

    provider = get_provider("read-path")
    model = {m["id"]: m for m in provider["models"]}["gpt-4o"]
    assert model["context_length"] == 128000
    assert model["max_output_tokens"] == 16384
    assert model["input_modalities"] == ["text", "image"]
    assert model["output_modalities"] == ["text"]
    assert model["input_price"] == pytest.approx(2.5)
    assert model["output_price"] == pytest.approx(10.0)
    assert model["cached_input_price"] == pytest.approx(1.25)


def test_get_provider_tolerates_corrupt_modalities_json():
    """§3.1: 读取端对 modalities 须 json.loads 并容错空串/非法 JSON."""
    _add_provider("corrupt", [{"id": "gpt-4o", "name": "GPT-4o", "enabled": True}])
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET input_modalities = ?, output_modalities = ? "
            "WHERE provider_id = 'corrupt' AND model_id = 'gpt-4o'",
            ("not json at all", ""),
        )

    provider = get_provider("corrupt")
    model = {m["id"]: m for m in provider["models"]}["gpt-4o"]
    assert model["input_modalities"] == []
    assert model["output_modalities"] == []


# -- Global (cross-provider) metadata concurrency cap --

@pytest.mark.asyncio
async def test_metadata_prefetch_cap_is_global_across_providers(monkeypatch):
    """The Semaphore(4) cap must bound ALL providers together, not each one.

    Regression: _METADATA_SEM used to be built inside refresh_provider_models,
    so every provider got a private budget of 4.  refresh_all_providers runs up
    to 4 providers concurrently, making the real in-flight count 4 x 4 = 16 --
    four times the documented cap, and enough to trip upstream rate limits.

    The single-provider test above cannot catch this: with one provider there is
    only ever one semaphore, so a per-call semaphore looks correct.
    """
    for pid in ("p1", "p2", "p3", "p4"):
        _add_provider(pid)
    _stub_discovery(monkeypatch, [{"id": f"m{i}", "name": f"M{i}"} for i in range(10)])

    in_flight = 0
    peak = 0

    async def slow_resolve(provider_id, model_id):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.005)
        in_flight -= 1
        return dict(GPT4O_METADATA)

    monkeypatch.setattr("app.services.discovery.resolve_model_metadata_async", slow_resolve)

    await refresh_all_providers()

    assert peak <= 4, (
        f"cross-provider metadata concurrency reached {peak}, exceeding the "
        f"global Semaphore(4); is _METADATA_SEM per-call again?"
    )
    assert peak > 1, "providers ran fully serialized -- the test lost its teeth"


@pytest.mark.asyncio
async def test_metadata_semaphore_is_shared_not_per_call():
    """Pin the fix structurally, not just behaviourally.

    A per-call semaphore is the exact shape of the bug, and it is an easy thing
    to reintroduce while refactoring.  Two calls inside one event loop must hand
    back the *same* object -- that identity is what makes the cap global.

    The accessor is a function rather than a bare module-level constant because
    asyncio.Semaphore binds to the running loop on first await and keeps that
    binding, so a single shared instance would break the moment a second loop
    used it (which is every asyncio test, and any asyncio.run caller).
    """
    import app.services.discovery as disc

    assert disc._metadata_sem() is disc._metadata_sem(), (
        "_metadata_sem() must return one shared semaphore per event loop; a new "
        "instance per call gives each refresh its own budget, which is the bug."
    )
    assert isinstance(disc._metadata_sem(), asyncio.Semaphore)
    assert disc._METADATA_CONCURRENCY == 4, (
        "the documented cap is 4; changing it means updating the docstring in "
        "resolve_model_metadata_async and the comment in refresh_provider_models"
    )


def test_metadata_semaphore_is_not_shared_across_loops():
    """Each event loop must get its own semaphore instance.

    Guards the failure mode that made a plain module-level constant unusable:
    asyncio.Semaphore binds to the running loop on first acquire, so reusing one
    instance in a second loop raises
    RuntimeError('... is bound to a different event loop').

    Deliberately a sync test using two asyncio.run() calls: the semaphore must be
    acquired inside each loop (that is what triggers the binding), and a nested
    run_until_complete inside an already-running loop is not allowed.
    """
    first = asyncio.run(_grab_sem())
    second = asyncio.run(_grab_sem())

    assert first is not second, (
        "a separate event loop must get its own semaphore; sharing one instance "
        "across loops raises RuntimeError on acquire"
    )


async def _grab_sem():
    import app.services.discovery as disc

    sem = disc._metadata_sem()
    async with sem:  # binding happens here -- a foreign instance raises
        pass
    return sem
