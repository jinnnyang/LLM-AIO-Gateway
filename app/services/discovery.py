import asyncio
import json
import logging
import time
import weakref

import httpx
from app.database import get_provider, update_provider, get_providers, get_db
from app.adapters.responses import iter_sse_frames, responses_headers, responses_url, sse_payload
from app.services.model_metadata import resolve_model_metadata

logger = logging.getLogger(__name__)


async def resolve_model_metadata_async(provider_id: str, model_id: str) -> dict:
    """Async wrapper around the sync resolver.

    The resolver does network I/O (models.dev, OpenRouter, LiteLLM) and must
    NOT block the event loop, so we offload it to a thread pool via
    asyncio.to_thread.  _metadata_sem() bounds how many run at once.
    """
    return await asyncio.to_thread(resolve_model_metadata, provider_id, model_id)

# Per-provider locks guard refresh/sync of the same provider so concurrent
# refreshes cannot interleave writes.  Limitations (§8 定案 13): the lock does
# not span threads (sync endpoints run in a threadpool) and is per-process, so
# multi-worker deployments retain a residual race.  Accepted.
_provider_locks: dict[str, asyncio.Lock] = {}

_METADATA_CONCURRENCY = 4

# Global cap on concurrent metadata resolutions, counted per model.
#
# MUST be module level, not per-call: refresh_all_providers runs up to 4
# providers concurrently, so a semaphore built inside refresh_provider_models
# gives each provider its own budget and the real in-flight count becomes
# 4 providers x 4 models = 16 -- four times what this cap claims, and enough
# to trip rate limits on the shared models.dev / OpenRouter endpoints.
#
# Keyed by event loop: asyncio.Semaphore binds itself to the running loop on
# first await and keeps that binding for good.  A single module-level instance
# would therefore die with "is bound to a different event loop" as soon as a
# second loop uses it (every asyncio test gets a fresh loop, and asyncio.run
# callers do too).  So we cache one semaphore per loop and drop it when the
# loop closes -- this keeps the cap global within a loop, which is the scope
# that actually matters for bounding concurrent upstream calls.
_metadata_sems: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _metadata_sem() -> asyncio.Semaphore:
    """Return the metadata concurrency cap for the running event loop."""
    loop = asyncio.get_running_loop()
    sem = _metadata_sems.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_METADATA_CONCURRENCY)
        _metadata_sems[loop] = sem
    return sem


def _provider_lock(provider_id: str) -> asyncio.Lock:
    lock = _provider_locks.get(provider_id)
    if lock is None:
        lock = asyncio.Lock()
        _provider_locks[provider_id] = lock
    return lock


def _has_any_metadata(metadata: dict) -> bool:
    """True if a resolved metadata dict carries at least one non-empty value.

    Used to skip refreshes that fully degraded to empty (all sources down), so
    previously-filled capability data is not wiped with NULLs (§4.2)."""
    for key in ("context_length", "max_output_tokens",
                "input_price", "output_price", "cached_input_price"):
        if metadata.get(key) is not None:
            return True
    for key in ("input_modalities", "output_modalities"):
        if metadata.get(key):
            return True
    return False


def model_list_urls(api_base: str, provider_type: str) -> list[str]:
    """Return candidate model-list URLs for the given provider.

    For Anthropic-compatible endpoints (non api.anthropic.com), also try
    the parent path - e.g. DeepSeek's /anthropic base has no /models,
    but the root /v1/models works."""
    api_base = api_base.rstrip("/")
    urls = []
    if provider_type == "anthropic":
        if not api_base.endswith("/v1"):
            urls.append(f"{api_base}/v1/models")
        urls.append(f"{api_base}/models")
        # Anthropic-compatible endpoints may host /models on a different base path
        if "api.anthropic.com" not in api_base:
            parent = api_base.rsplit("/", 1)[0]
            if parent and parent != api_base:
                for u in (f"{parent}/v1/models", f"{parent}/models"):
                    if u not in urls:
                        urls.append(u)
    else:
        urls = [f"{api_base}/models"]
    return urls


def auth_headers(api_key: str, provider_type: str) -> list[dict]:
    if api_key:
        headers = [{"Authorization": f"Bearer {api_key}"}]
        if provider_type == "anthropic":
            headers.append({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    else:
        headers = [{}]
    return headers


def parse_models(data: dict) -> list[dict]:
    raw_models = data.get("data")
    if raw_models is None:
        raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        return []

    models = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("identifier") or item.get("name")
        if model_id:
            models.append({
                "id": model_id,
                "name": item.get("display_name") or item.get("name") or model_id
            })
    return models


async def discover_models(provider_id: str) -> list[dict]:
    provider = get_provider(provider_id)
    if not provider:
        return []
    if not provider.get("enabled"):
        return []

    api_base = provider["api_base"].rstrip("/")
    api_key = provider["api_key"]
    provider_type = provider["provider_type"]

    last_error = None
    async with httpx.AsyncClient() as client:
        for url in model_list_urls(api_base, provider_type):
            for headers in auth_headers(api_key, provider_type):
                try:
                    models_by_id = {}
                    cursor = ""
                    seen_cursors = set()
                    for _page in range(100):
                        params = None
                        if cursor:
                            params = {"after_id" if provider_type == "anthropic" else "after": cursor}
                        request_kwargs = {"headers": headers, "timeout": 10.0}
                        if params:
                            request_kwargs["params"] = params
                        resp = await client.get(url, **request_kwargs)
                        resp.raise_for_status()
                        payload = resp.json()
                        page_models = parse_models(payload)
                        for model in page_models:
                            models_by_id[str(model["id"])] = model
                        if not payload.get("has_more"):
                            break
                        next_cursor = str(payload.get("last_id") or "").strip()
                        if not next_cursor and page_models:
                            next_cursor = str(page_models[-1]["id"])
                        if not next_cursor or next_cursor in seen_cursors:
                            raise RuntimeError("model discovery returned an invalid pagination cursor")
                        seen_cursors.add(next_cursor)
                        cursor = next_cursor
                    else:
                        raise RuntimeError("model discovery exceeded 100 pages")
                    if models_by_id:
                        return list(models_by_id.values())
                except Exception as exc:
                    last_error = exc

    if last_error:
        raise last_error

    return []


async def refresh_provider_models(provider_id: str) -> dict:
    async with _provider_lock(provider_id):
        try:
            discovered = await discover_models(provider_id)
        except Exception as exc:
            return {
                "provider_id": provider_id,
                "discovered": [],
                "count": 0,
                "added": 0,
                "updated": 0,
                "removed": 0,
                "error": str(exc)
            }

        added = 0
        updated = 0
        removed = 0

        if discovered:
            discovered_by_id = {d["id"]: d for d in discovered}
            discovered_ids = set(discovered_by_id)
            with get_db() as db:
                existing_rows = db.execute(
                    "SELECT model_id, source FROM provider_models WHERE provider_id = ?",
                    (provider_id,),
                ).fetchall()
                existing_ids = {row["model_id"] for row in existing_rows}
                auto_ids = {
                    row["model_id"]
                    for row in existing_rows
                    if (row["source"] or "auto") == "auto"
                }

                stale_ids = auto_ids - discovered_ids
                if stale_ids:
                    db.executemany(
                        "DELETE FROM provider_models WHERE provider_id = ? AND model_id = ? AND source = 'auto'",
                        [(provider_id, model_id) for model_id in stale_ids],
                    )
                    removed = len(stale_ids)

                for model_id, model in discovered_by_id.items():
                    if model_id in existing_ids:
                        cursor = db.execute(
                            "UPDATE provider_models SET model_name = ? WHERE provider_id = ? AND model_id = ? AND source = 'auto'",
                            (model["name"], provider_id, model_id),
                        )
                        if cursor.rowcount:
                            updated += 1
                    else:
                        db.execute(
                            "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled, source) VALUES (?, ?, ?, 1, 'auto')",
                            (provider_id, model_id, model["name"]),
                        )
                        added += 1

            # ── M3: capability metadata enrichment ──
            # Batch prefetch metadata for all discovered models.  Only
            # source='auto' rows actually get filled; custom rows are
            # protected by the AND source='auto' clause in the UPDATE.
            # Per-model resolve failures are swallowed (try/except):
            # a single failure cannot roll back the entire refresh.
            # Prefetch uses asyncio.gather bounded by the module-level
            # _metadata_sem(), which caps in-flight resolutions across ALL
            # providers (not just this one).

            async def _resolve_one(pid: str, mid: str) -> tuple[str, dict | None]:
                async with _metadata_sem():
                    try:
                        result = await resolve_model_metadata_async(pid, mid)
                        return mid, result
                    except Exception:
                        return mid, None

            if discovered_ids:
                tasks = [_resolve_one(provider_id, mid) for mid in discovered_ids]
                resolved = dict(await asyncio.gather(*tasks))

                for model_id, metadata in resolved.items():
                    if metadata is None:
                        continue
                    # Design §4.2: preserve existing values on failure.  When every
                    # source degrades to empty (all three fail + LiteLLM has no
                    # entry), resolve returns an all-empty dict — writing it would
                    # wipe previously-filled capability data with NULLs.  Skip the
                    # UPDATE so old values survive until a future refresh succeeds.
                    if not _has_any_metadata(metadata):
                        continue
                    with get_db() as db:
                        db.execute(
                            """UPDATE provider_models SET
                                context_length = ?,
                                max_output_tokens = ?,
                                input_modalities = ?,
                                output_modalities = ?,
                                input_price = ?,
                                output_price = ?,
                                cached_input_price = ?
                            WHERE provider_id = ? AND model_id = ? AND source = 'auto'""",
                            (
                                metadata.get("context_length"),
                                metadata.get("max_output_tokens"),
                                json.dumps(metadata.get("input_modalities", [])),
                                json.dumps(metadata.get("output_modalities", [])),
                                metadata.get("input_price"),
                                metadata.get("output_price"),
                                metadata.get("cached_input_price"),
                                provider_id,
                                model_id,
                            ),
                        )

    return {
        "provider_id": provider_id,
        "discovered": discovered,
        "count": len(discovered),
        "added": added,
        "updated": updated,
        "removed": removed,
    }
async def refresh_all_providers() -> list[dict]:
    providers = [provider for provider in get_providers() if provider.get("enabled")]
    limiter = asyncio.Semaphore(4)

    async def refresh(provider_id: str) -> dict:
        async with limiter:
            return await refresh_provider_models(provider_id)

    return list(await asyncio.gather(*(refresh(provider["id"]) for provider in providers)))


async def check_provider_health(provider_id: str, timeout: float = 10.0) -> dict:
    provider = get_provider(provider_id)
    if not provider:
        return {"provider_id": provider_id, "ok": False, "status": "not_found", "error": "Provider not found"}
    if not provider.get("enabled"):
        return {"provider_id": provider_id, "ok": False, "status": "disabled", "error": "Provider is disabled"}

    started = time.perf_counter()
    last_error = None
    attempts = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in model_list_urls(provider.get("api_base", ""), provider.get("provider_type", "openai")):
            for headers in auth_headers(provider.get("api_key", ""), provider.get("provider_type", "openai")):
                attempt = {"url": url, "ok": False, "status_code": None, "model_count": 0, "error": ""}
                try:
                    resp = await client.get(url, headers=headers)
                    attempt["status_code"] = resp.status_code
                    resp.raise_for_status()
                    models = parse_models(resp.json())
                    attempt["ok"] = True
                    attempt["model_count"] = len(models)
                    attempts.append(attempt)
                    return {
                        "provider_id": provider_id,
                        "ok": True,
                        "status": "ok",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "checked_url": url,
                        "status_code": resp.status_code,
                        "model_count": len(models),
                        "attempts": attempts,
                    }
                except Exception as exc:
                    last_error = exc
                    attempt["error"] = str(exc)
                    attempts.append(attempt)

    return {
        "provider_id": provider_id,
        "ok": False,
        "status": "error",
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "error": str(last_error) if last_error else "No health endpoint succeeded",
        "attempts": attempts,
    }


async def check_all_provider_health(timeout: float = 10.0) -> list[dict]:
    results = []
    for provider in get_providers():
        results.append(await check_provider_health(provider["id"], timeout=timeout))
    return results
