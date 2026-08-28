"""
Model capability metadata resolution: context length, prices, and modalities.

Design: docs/模型能力元数据扩展-设计方案.md §2.1-§2.3 / §4.1 / §8 定案 1-14.

Three sources always merge in parallel:
  1. models.dev  (network, $/M token, includes cached_input_price)
  2. OpenRouter  (network, per-token strings → $/M, includes cached_input_price)
  3. LiteLLM     (local, per-token floats → $/M, always participates)

Two-level cache:
  L1: full catalogs (models.dev JSON, OpenRouter JSON) — TTL 24h, lazy-loaded, shared across
      all model lookups within a process.
  L2: per-model resolved results — TTL 5 min, keyed by (provider_id, normalized_model_id).
      L2 *only* consumes L1; it must never issue its own network request.
"""

import json
import logging
import re
import threading
import time
from typing import Optional

import httpx

from app.database import get_db, _json_loads

logger = logging.getLogger(__name__)

# ── Whitelist ──────────────────────────────────────────────────────────────────
# 模态白名单宁漏勿虚 (§2.2 定案 6)
MODALITY_WHITELIST = {
    "text", "image", "audio", "video",
    "vision",   # ← normalized to "image" during merge
    "pdf",      # ← normalized to "file" during merge
    "file",
    "embedding",
}

# 归一化映射：keep only the canonical form
MODALITY_NORMALIZE = {
    "vision": "image",
    "pdf": "file",
}

# ── Normalization: model_id ────────────────────────────────────────────────────

# 1) provider prefixes
_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"vcp-|openai/|anthropic\.|google/|azure/|vertex_ai/"
    r"|bedrock/|replicate/|together/|fireworks/"
    r"|mistral/|cohere/|perplexity/|groq/|xai/|deepseek/"
    r"|ollama/|huggingface/"
    r")",
    re.IGNORECASE,
)

# 2) date/version suffixes
_SUFFIX_PATTERN = re.compile(
    r"(?:"
    r"-(?:latest|snapshot|beta|alpha|rc(?:\d+)?)"
    r"|-\d{8}"
    r"|-\d{6}"
    r"|-\d{4}-[a-z]+\d*"
    r"|-\d{4}"
    r"|-\d{3}"
    r"|-[a-f0-9]{7,40}"
    r")$",
    re.IGNORECASE,
)

# 3) trailing date like -20240229 (already covered by _SUFFIX_PATTERN)
# 4) version suffix like -0125 (already covered by -d{4})


def _normalize_model_id(model_id) -> str:
    """Normalize a model ID for lookup: strip prefixes, date suffixes, lowercase.

    Priority (§2.3 定案 10):
      1. Remove provider prefix (vcp-, openai/, anthropic., etc.)
      2. Remove date/version suffix (-latest, -20240229, -0125, -1106-preview)
      3. Lowercase
      4. Unmatched → log but don't block
    """
    if not model_id:
        return ""

    mid = str(model_id).strip()

    # Step 1: strip provider prefix
    mid = _PREFIX_PATTERN.sub("", mid)

    # Step 2: strip date/version suffix (iteratively, one at a time)
    # e.g. "gpt-4-1106-preview" -> strip "-1106-preview" -> "gpt-4"
    # But actually we need to strip the whole trailing version group.
    # Better: strip the first suffix match from the end-like pattern.
    m = _SUFFIX_PATTERN.search(mid)
    if m:
        mid = mid[: m.start()]

    # Step 3: lowercase
    mid = mid.lower()

    # Step 4: strip any remaining whitespace
    mid = mid.strip()

    return mid


# ── L1 cache ───────────────────────────────────────────────────────────────────

class _L1Cache:
    """Full-catalog cache for models.dev and OpenRouter.

    L1 is lazy-loaded: the first model lookup for each source triggers the fetch.
    TTL is 24 hours.

    Locking (§ P2-1): two lock tiers, so that a network fetch never blocks readers.

      * ``self._lock`` guards the cached state only.  It is held for dict
        assignments and timestamp reads -- never across ``fetcher()``.
      * ``self._refresh_locks[source]`` serialises refreshes per source, so a
        stampede of callers triggers exactly one upstream request (this is the
        thundering-herd guard).

    Refresh policy differs by whether we already have a usable catalog:

      * **Warm and stale** -- serve the stale catalog immediately and let *one*
        caller refresh in the background of its own call.  Other callers take
        the stale value rather than queueing on the refresh lock.  A catalog one
        day old is vastly better than a multi-second stall.
      * **Cold (never fetched)** -- callers must block on the refresh lock,
        because returning ``{}`` here would silently resolve every model to "no
        metadata" and that empty result would get cached in L2.
    """

    _SOURCES = ("models_dev", "openrouter")

    def __init__(self, ttl: int = 86400):
        self._lock = threading.Lock()
        self._refresh_locks = {name: threading.Lock() for name in self._SOURCES}
        self._ttl = ttl  # seconds
        self._models_dev: Optional[dict] = None
        self._models_dev_ts: float = 0.0
        self._openrouter: Optional[dict] = None
        self._openrouter_ts: float = 0.0

    def _stale(self, ts: float) -> bool:
        return (time.time() - ts) > self._ttl

    def _refresh(self, source: str, fetcher, label: str, allow_empty: bool):
        """Fetch and store one catalog.  Caller must hold the refresh lock.

        ``self._lock`` is deliberately *not* held while ``fetcher()`` runs.
        """
        attr, ts_attr = f"_{source}", f"_{source}_ts"
        try:
            data = fetcher()
        except Exception as exc:
            logger.warning("L1 %s fetch failed: %s", label, exc)
            # Cold start: fall back to an empty catalog so callers get a dict,
            # but leave the timestamp at 0 so the next call retries.
            if allow_empty:
                with self._lock:
                    if getattr(self, attr) is None:
                        setattr(self, attr, {})
            return
        with self._lock:
            setattr(self, attr, data)
            setattr(self, ts_attr, time.time())

    def _get(self, source: str, fetcher, label: str) -> dict:
        attr, ts_attr = f"_{source}", f"_{source}_ts"
        refresh_lock = self._refresh_locks[source]

        with self._lock:
            value = getattr(self, attr)
            stale = self._stale(getattr(self, ts_attr))

        if value is not None and not stale:
            return value or {}

        if value is not None:
            # Warm but stale: refresh opportunistically, never block a reader.
            if not refresh_lock.acquire(blocking=False):
                return value or {}
            try:
                with self._lock:
                    if not self._stale(getattr(self, ts_attr)):
                        return getattr(self, attr) or {}  # someone just refreshed
                self._refresh(source, fetcher, label, allow_empty=False)
            finally:
                refresh_lock.release()
            with self._lock:
                return getattr(self, attr) or {}

        # Cold: block, but only one caller actually hits the network.
        with refresh_lock:
            with self._lock:
                if getattr(self, attr) is not None:
                    return getattr(self, attr) or {}  # filled while we queued
            self._refresh(source, fetcher, label, allow_empty=True)
        with self._lock:
            return getattr(self, attr) or {}

    def get_models_dev(self, fetcher) -> dict:
        return self._get("models_dev", fetcher, "models.dev")

    def get_openrouter(self, fetcher) -> dict:
        return self._get("openrouter", fetcher, "OpenRouter")


    def clear(self):
        with self._lock:
            self._models_dev = None
            self._models_dev_ts = 0.0
            self._openrouter = None
            self._openrouter_ts = 0.0


_l1 = _L1Cache()


# ── L2 cache ───────────────────────────────────────────────────────────────────

class _L2Cache:
    """Per-model resolved result cache.

    Key: (provider_id, normalized_model_id).
    TTL: 5 minutes.
    L2 *only* consumes L1; it must never issue its own network request.
    Thread-safe via a Lock.
    """

    def __init__(self, ttl: int = 300):
        self._lock = threading.Lock()
        self._ttl = ttl
        self._data: dict[tuple[str, str], tuple[float, dict]] = {}

    def get(self, provider_id: str, model_id: str) -> Optional[dict]:
        key = (provider_id, model_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, value = entry
            if (time.time() - ts) > self._ttl:
                del self._data[key]
                return None
            return value

    def set(self, provider_id: str, model_id: str, value: dict):
        key = (provider_id, model_id)
        with self._lock:
            self._data[key] = (time.time(), value)

    def invalidate(self, provider_id: str, model_id: str):
        key = (provider_id, model_id)
        with self._lock:
            self._data.pop(key, None)

    def clear(self):
        with self._lock:
            self._data.clear()


_l2 = _L2Cache()


# ── Network fetchers (uncached — L1 cache wraps these) ─────────────────────────

_HTTP_TIMEOUT = 15.0


def _fetch_models_dev_catalog_uncached() -> dict:
    """Fetch the full models.dev catalog and flatten it by model ID.

    The live API (`https://models.dev/api.json`) returns a dict keyed by
    PROVIDER, with models nested under each provider's `models` key (model ids
    already carry the provider prefix, e.g. `deepseek/deepseek-v4-flash`).
    We flatten across all providers into a single dict keyed by model id so
    downstream lookups (`_from_models_dev`) can do a direct `catalog.get()`.

    Returns a dict of model_id -> model dict (field positions verified live):
        {"id": str, "name": str, "limit": {"context": int, "output": int},
         "modalities": {"input": [str, ...], "output": [str, ...]},
         "cost": {"input": float, "output": float, "cache_read": float}}
    """
    url = "https://models.dev/api.json"
    resp = httpx.get(url, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Case 1: provider-keyed dict (the live shape) — flatten nested `models`.
    if isinstance(data, dict):
        flat = {}
        for provider in data.values():
            if not isinstance(provider, dict):
                continue
            models = provider.get("models")
            if isinstance(models, dict):
                for mid, m in models.items():
                    if isinstance(m, dict):
                        flat[mid] = m
            elif isinstance(models, list):
                for m in models:
                    if isinstance(m, dict) and "id" in m:
                        flat[m["id"]] = m
        if flat:
            return flat
        # Provider-keyed but no nested `models` found → fall through to the
        # flat-dict-by-id heuristic below (defensive; not the live shape).
        if all(not isinstance(v, dict) for v in data.values()):
            return data
    # Case 2: already a flat list of model objects — index by id.
    if isinstance(data, list):
        return {m["id"]: m for m in data if isinstance(m, dict) and "id" in m}

    logger.warning("models.dev returned unexpected type: %s", type(data).__name__)
    return {}


def _fetch_models_dev_catalog() -> dict:
    """L1-cached wrapper for models.dev catalog."""
    return _l1.get_models_dev(_fetch_models_dev_catalog_uncached)


def _fetch_openrouter_catalog() -> dict:
    """Fetch the full OpenRouter models catalog.

    Returns a dict with key "data" containing a list of model objects.

    Expected shape (per model in data[]):
        {"id": str, "context_length": int,
         "top_provider": {"max_completion_tokens": int},
         "architecture": {"input_modalities": [str, ...], "output_modalities": [str, ...]},
         "pricing": {"prompt": str, "completion": str, "input_cache_read": str}}
    """
    raw = _l1.get_openrouter(_fetch_openrouter_catalog_uncached)
    # The L1 cache stores the raw response; return it as-is.
    return raw


def _fetch_openrouter_catalog_uncached() -> dict:
    """Uncached fetch for OpenRouter catalog."""
    url = "https://openrouter.ai/api/v1/models"
    resp = httpx.get(url, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _litellm_model_cost() -> dict:
    """Return the litellm.model_cost dictionary.

    This is a local/zero-cost operation — no network request.
    """
    try:
        import litellm
        return getattr(litellm, "model_cost", {})
    except ImportError:
        logger.warning("litellm not installed; model_cost unavailable")
        return {}


# ── Per-source adapters ───────────────────────────────────────────────────────

def _from_models_dev(model_id: str) -> Optional[dict]:
    """Resolve from models.dev catalog.

    models.dev advertises prices in $/M token — do NOT scale again.
    """
    catalog = _fetch_models_dev_catalog()
    if not isinstance(catalog, dict):
        logger.warning("models.dev catalog is not a dict: %s", type(catalog).__name__)
        return None

    norm = _normalize_model_id(model_id)
    # Try normalized key first, then raw id
    entry = catalog.get(norm) or catalog.get(model_id)
    if not entry or not isinstance(entry, dict):
        return None

    result = {}

    # context_length / max_output_tokens
    limit = entry.get("limit") or {}
    if isinstance(limit, dict):
        result["context_length"] = limit.get("context")
        result["max_output_tokens"] = limit.get("output")

    # modalities
    modalities = entry.get("modalities") or {}
    if isinstance(modalities, dict):
        result["input_modalities"] = modalities.get("input") or []
        result["output_modalities"] = modalities.get("output") or []

    # prices (already $/M — no scaling)
    cost = entry.get("cost") or {}
    if isinstance(cost, dict):
        result["input_price"] = cost.get("input")
        result["output_price"] = cost.get("output")
        result["cached_input_price"] = cost.get("cache_read")

    return result if any(v is not None for v in result.values()) else None


def _from_openrouter(model_id: str) -> Optional[dict]:
    """Resolve from OpenRouter catalog.

    OpenRouter publishes per-token price STRINGS: float() then x1e6 to reach $/M.
    """
    catalog = _fetch_openrouter_catalog()
    data = catalog.get("data") if isinstance(catalog, dict) else None
    if not isinstance(data, list):
        return None

    norm = _normalize_model_id(model_id)
    entry = None
    for m in data:
        if not isinstance(m, dict):
            continue
        mid = m.get("id", "")
        if _normalize_model_id(mid) == norm or mid == model_id:
            entry = m
            break

    if not entry:
        return None

    result = {}

    # context_length
    result["context_length"] = entry.get("context_length")

    # max_output_tokens
    top_provider = entry.get("top_provider") or {}
    if isinstance(top_provider, dict):
        result["max_output_tokens"] = top_provider.get("max_completion_tokens")

    # modalities
    arch = entry.get("architecture") or {}
    if isinstance(arch, dict):
        result["input_modalities"] = arch.get("input_modalities") or []
        result["output_modalities"] = arch.get("output_modalities") or []

    # prices (per-token strings → $/M)
    pricing = entry.get("pricing") or {}
    if isinstance(pricing, dict):
        for source_key, target_key in (
            ("prompt", "input_price"),
            ("completion", "output_price"),
            ("input_cache_read", "cached_input_price"),
        ):
            raw = pricing.get(source_key)
            if raw is not None:
                try:
                    result[target_key] = float(raw) * 1_000_000
                except (TypeError, ValueError):
                    pass

    return result if any(v is not None for v in result.values()) else None


def _from_litellm(model_id: str) -> Optional[dict]:
    """Resolve from LiteLLM's local model_cost dictionary.

    LiteLLM stores per-token floats: x1e6 to reach $/M.
    """
    catalog = _litellm_model_cost()
    if not isinstance(catalog, dict):
        return None

    norm = _normalize_model_id(model_id)
    # Try normalized key first, then raw id
    entry = catalog.get(norm) or catalog.get(model_id)
    if not entry or not isinstance(entry, dict):
        return None

    result = {}

    # context_length
    result["context_length"] = entry.get("max_input_tokens")

    # max_output_tokens
    result["max_output_tokens"] = entry.get("max_output_tokens")

    # prices (per-token floats → $/M)
    for source_key, target_key in (
        ("input_cost_per_token", "input_price"),
        ("output_cost_per_token", "output_price"),
        ("cache_read_input_token_cost", "cached_input_price"),
    ):
        val = entry.get(source_key)
        if val is not None:
            try:
                result[target_key] = float(val) * 1_000_000
            except (TypeError, ValueError):
                pass

    # modalities: LiteLLM doesn't declare modalities explicitly.
    # Infer from `mode` and `supports_vision` flags.
    mode = entry.get("mode", "")
    inferred = False
    input_mods = []
    output_mods = []

    if mode == "chat":
        input_mods.append("text")
        output_mods.append("text")
        if entry.get("supports_vision"):
            input_mods.append("image")
        inferred = True
    elif mode == "embedding":
        input_mods.append("text")
        output_mods.append("embedding")
        inferred = True

    if inferred:
        result["input_modalities"] = input_mods
        result["output_modalities"] = output_mods
        result["modalities_inferred"] = True

    return result if any(v is not None for v in result.values()) else None


def _from_provider_models(provider_id: str, model_id: str) -> Optional[dict]:
    """Resolve modalities-only from the provider's own /models endpoint.

    定案 5: 供应商 /models 是 modalities-only 补充源，不参与价格候选计数、
    不影响降级判定。
    """
    # This is a modalities-only source: we never return prices from here.
    try:
        from app.services.discovery import discover_models
        import asyncio

        # discover_models is async; run it to completion. If the caller is already
        # inside a running event loop (e.g. an async endpoint), we cannot block on
        # it here, so skip the provider /models supplement.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        else:
            running_loop = True
        if running_loop is not None:
            logger.debug("provider /models skipped: event loop already running")
            return None

        # No running loop (e.g. discovery's asyncio.to_thread path): create a
        # dedicated loop, run it, and close it so no loop resource leaks. The
        # result survives even if the final close() is noisy.
        loop = asyncio.new_event_loop()
        try:
            discovered = loop.run_until_complete(discover_models(provider_id))
        finally:
            try:
                loop.close()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("provider /models fetch failed for %s: %s", provider_id, exc)
        return None

    # Match the model by id
    norm = _normalize_model_id(model_id)
    match = None
    for m in discovered:
        if not isinstance(m, dict):
            continue
        mid = m.get("id", "")
        if _normalize_model_id(mid) == norm or mid == model_id:
            match = m
            break

    if not match:
        return None

    result = {}
    # Provider /models typically returns only model metadata, not capability fields.
    # If it does return modalities, we extract them.
    mods = match.get("modalities") or match.get("capabilities") or {}
    if isinstance(mods, dict):
        result["input_modalities"] = mods.get("input") or mods.get("input_modalities") or []
        result["output_modalities"] = mods.get("output") or mods.get("output_modalities") or []
    elif isinstance(mods, list):
        # Some APIs return a flat list of supported features
        result["input_modalities"] = mods
        result["output_modalities"] = mods

    return result if result.get("input_modalities") or result.get("output_modalities") else None


# ── Merge helpers ──────────────────────────────────────────────────────────────

def _first_present(sources: list[Optional[dict]], field: str) -> Optional[int]:
    """Return `field` from the first source that declares it.

    Used for fields whose sources are not mutually comparable, so combining them
    numerically would produce a value with no well-defined meaning. `sources`
    must be ordered most-trusted first; a later source is a fallback for a
    missing value, never a competing candidate for a present one.
    """
    for source in sources:
        if not source:
            continue
        value = source.get(field)
        if value is not None:
            return value
    return None


def _merge_price(
    values: list[Optional[float]],
    model_id: str = "",
    field: str = "",
) -> Optional[float]:
    """Merge a list of price candidates from multiple sources.

    Rules (§2.1 定案 4):
      - Filter out None values (not published by that source).
      - >=3 candidates → median.
      - ==2 candidates → higher price (conservative: never under-bill).
        If they differ >20%, log a WARN with model_id and both candidate values.
      - ==1 candidate → single-source adoption.
      - 0 candidates → None.
      - >10x outlier → log but still merge (don't drop).
    """
    candidates = [v for v in values if v is not None]
    if not candidates:
        return None

    candidates.sort()

    if len(candidates) >= 3:
        # Median: middle element for odd, average of two middle for even
        n = len(candidates)
        if n % 2 == 1:
            picked = candidates[n // 2]
        else:
            picked = (candidates[n // 2 - 1] + candidates[n // 2]) / 2.0
        # Log any >10x outlier
        _check_outliers(candidates, model_id, field)
        return picked

    if len(candidates) == 2:
        low, high = candidates[0], candidates[1]
        # Log >10x outlier
        _check_outliers(candidates, model_id, field)
        # Conflict >20% → WARN
        if high > 0 and (high - low) / high > 0.20:
            logger.warning(
                "price conflict model=%s field=%s candidates=%s picked=%s",
                model_id, field, candidates, high,
            )
        return high  # always take the higher price

    # Single candidate
    _check_outliers(candidates, model_id, field)
    return candidates[0]


def _check_outliers(candidates: list[float], model_id: str, field: str):
    """Log a warning if any candidate is >10x the median."""
    if len(candidates) < 3:
        return
    sorted_c = sorted(candidates)
    median = sorted_c[len(sorted_c) // 2]
    if median > 0:
        for v in sorted_c:
            if v / median > 10 or median / v > 10:
                logger.warning(
                    "price outlier model=%s field=%s value=%s median=%s candidates=%s",
                    model_id, field, v, median, candidates,
                )
                break


_MODALITY_CANONICAL = {
    "vision": "image",
    "pdf": "file",
}


def _merge_modalities(candidates: list[dict]) -> dict:
    """Merge modalities from multiple sources.

    Rules (§2.2):
      - Whitelist union: only known modalities survive.
      - Normalize vocabulary: image <-> vision, pdf <-> file.
      - Output is deterministically ordered (sorted).
      - Empty input → {"input": [], "output": []}.
    """
    seen_input = set()
    seen_output = set()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for mods in candidate.get("input_modalities") or []:
            if isinstance(mods, list):
                # If input_modalities is somehow a list of lists, flatten
                for m in mods:
                    _add_modality(seen_input, m)
            elif isinstance(mods, str):
                _add_modality(seen_input, mods)
        for mods in candidate.get("output_modalities") or []:
            if isinstance(mods, list):
                for m in mods:
                    _add_modality(seen_output, m)
            elif isinstance(mods, str):
                _add_modality(seen_output, mods)

    return {
        "input": sorted(seen_input),
        "output": sorted(seen_output),
    }


def _add_modality(seen: set, modality: str):
    """Add a modality to the set, normalizing through the canonical map."""
    canonical = _MODALITY_CANONICAL.get(modality, modality)
    if canonical in MODALITY_WHITELIST:
        seen.add(canonical)


# ── Main entry point ───────────────────────────────────────────────────────────

def resolve_model_metadata(provider_id: str, model_id: str) -> dict:
    """Resolve capability metadata for a model from all available sources.

    Merge order (§2.3 定案 8):
      1. Modalities merge (whitelist union)
      2. Check if output_modalities contains "embedding"
      3. Price merge (embedding models need only input_price)

    Returns a dict with 7 keys:
      context_length, max_output_tokens, input_modalities, output_modalities,
      input_price, output_price, cached_input_price
    """
    norm = _normalize_model_id(model_id)

    # L2 check
    cached = _l2.get(provider_id, norm)
    if cached is not None:
        return cached

    # Collect candidates from all three sources
    try:
        models_dev = _from_models_dev(model_id)
    except Exception as exc:
        logger.debug("models.dev resolve failed for %s: %s", model_id, exc)
        models_dev = None

    try:
        openrouter = _from_openrouter(model_id)
    except Exception as exc:
        logger.debug("OpenRouter resolve failed for %s: %s", model_id, exc)
        openrouter = None

    try:
        litellm = _from_litellm(model_id)
    except Exception as exc:
        logger.debug("LiteLLM resolve failed for %s: %s", model_id, exc)
        litellm = None

    # Provider /models (modalities-only supplement)
    try:
        provider_mods = _from_provider_models(provider_id, model_id)
    except Exception as exc:
        logger.debug("provider /models resolve failed for %s/%s: %s", provider_id, model_id, exc)
        provider_mods = None

    # ── Step 1: Merge modalities (whitelist union) ──
    # LiteLLM's mode/supports_vision inference is display-only and must NOT
    # participate in routing decisions (§2.3 定案 5). Its inferred modalities are
    # excluded from the merged (stored) result; only declared modalities from
    # models.dev, OpenRouter and the provider /models source contribute. LiteLLM
    # still participates in the price merge below (separate candidate list).
    litellm_declared = litellm if not (litellm or {}).get("modalities_inferred") else None
    modality_candidates = [
        m for m in [models_dev, openrouter, litellm_declared, provider_mods]
        if m is not None
    ]
    merged_mods = _merge_modalities(modality_candidates)
    input_mods = merged_mods["input"]
    output_mods = merged_mods["output"]

    # ── Step 2: Check if embedding ──
    is_embedding = "embedding" in output_mods

    # ── Step 3: Merge prices ──
    price_sources = [m for m in [models_dev, openrouter, litellm] if m is not None]

    if is_embedding:
        # Embedding models: only input_price matters
        input_price = _merge_price(
            [m.get("input_price") for m in price_sources],
            model_id=model_id, field="input_price",
        )
        output_price = None
        cached_input_price = _merge_price(
            [m.get("cached_input_price") for m in price_sources],
            model_id=model_id, field="cached_input_price",
        )
    else:
        input_price = _merge_price(
            [m.get("input_price") for m in price_sources],
            model_id=model_id, field="input_price",
        )
        output_price = _merge_price(
            [m.get("output_price") for m in price_sources],
            model_id=model_id, field="output_price",
        )
        cached_input_price = _merge_price(
            [m.get("cached_input_price") for m in price_sources],
            model_id=model_id, field="cached_input_price",
        )

    # ── Step 4: Merge context fields ──
    # Source priority, NOT max(): the three sources do not report the same
    # quantity, so they are not comparable.
    #   models.dev  limit.context        -> total window (input + output)
    #   OpenRouter  context_length       -> total window
    #   LiteLLM     max_input_tokens     -> input ceiling only (<= total window)
    # Mixing them under max() could store an input ceiling as a total window (or
    # vice versa), always erring high. Falling back in declared-trust order keeps
    # the value's meaning intact; a lower-priority source is consulted only when
    # the ones above it have nothing to say.
    context_sources = [models_dev, openrouter, litellm]

    context_length = _first_present(context_sources, "context_length")
    max_output_tokens = _first_present(context_sources, "max_output_tokens")

    # ── Build result ──
    result = {
        "context_length": context_length,
        "max_output_tokens": max_output_tokens,
        "input_modalities": input_mods,
        "output_modalities": output_mods,
        "input_price": input_price,
        "output_price": output_price,
        "cached_input_price": cached_input_price,
    }

    # Store in L2 cache
    _l2.set(provider_id, norm, result)

    return result



# ── Cache management ──────────────────────────────────────────────────────────

def clear_caches():
    """Clear both L1 and L2 caches."""
    _l1.clear()
    _l2.clear()


def invalidate_model_cache(provider_id: str, model_id: str):
    """Invalidate a single L2 cache entry.

    The model_id is normalized before lookup, so unnormalized ids work.
    """
    norm = _normalize_model_id(model_id)
    _l2.invalidate(provider_id, norm)