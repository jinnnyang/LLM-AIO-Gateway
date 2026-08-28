"""
API tests for the model capability metadata endpoints (M4).

Design reference: docs/plans/模型能力元数据扩展-设计方案.md §4.3 / §4.4 / §6 用例 4/8/11.

Contract these tests pin down:
- GET .../metadata-candidates is read-only: it never writes the database.
- PUT .../capabilities is the only write path: both custom and auto rows are
  writable; auto rows just get overwritten again by the next upstream refresh.
- mode=fill_empty: only fills NULL fields, never overwrites existing values.
- PUT (no mode): full overwrite (default semantics).
- The existing PUT /providers/{id} (models array) must NOT silently clear metadata
  when the client omits the new fields (design §4.4: "不扩展现有 update 模型端点").
- XSS: candidate values from external sources must be escaped before innerHTML (tested
  at the JS level).
- M1 backward compatibility: custom fill + save works correctly.
"""
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.database import init_db, get_db, add_provider, add_admin
from app.security import create_session, hash_password


client = TestClient(app)


SAMPLE_PROVIDER = {
    "id": "test-provider",
    "name": "Test Provider",
    "provider_type": "openai",
    "api_base": "https://api.test.com/v1",
    "api_key": "sk-test-key",
    "enabled": True,
    "models": [],
}

GPT4O_MANUAL = {
    "id": "gpt-4o",
    "name": "GPT-4o",
    "enabled": True,
    "source": "custom",
}

GPT4O_AUTO = {
    "id": "gpt-4o",
    "name": "GPT-4o",
    "enabled": True,
    "source": "auto",
}


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Isolated database with admin session for every test."""
    db_path = str(tmp_path / "test_metadata_api.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
    }
    config.save()
    init_db(db_path)
    add_admin("admin", hash_password("secret"), "Admin")
    token = create_session("admin")
    yield {"headers": {"Authorization": f"Bearer {token}"}}
    # App state is per-module; resetting fully would require module reload.
    # Tests that share state are ordered within this file — individual isolation
    # is assured by tmp_path -> unique db_path.


def _seed_custom_model(provider_id, model_id="gpt-4o"):
    """Create a provider with a custom model and optional metadata."""
    add_provider({
        **SAMPLE_PROVIDER,
        "id": provider_id,
        "models": [{
            "id": model_id,
            "name": "GPT-4o Manual",
            "enabled": True,
            "source": "custom",
        }],
    })
    return provider_id


def _seed_auto_model(provider_id, model_id="gpt-4o"):
    """Create a provider with an auto-discovered model."""
    add_provider({
        **SAMPLE_PROVIDER,
        "id": provider_id,
        "models": [{
            "id": model_id,
            "name": "GPT-4o Auto",
            "enabled": True,
            "source": "auto",
        }],
    })
    return provider_id


def _set_metadata(provider_id, model_id, **fields):
    """Directly write capability columns for a model."""
    assignments = ", ".join(f"{k} = ?" for k in fields)
    values = [
        json.dumps(v) if k.endswith("_modalities") else v
        for k, v in fields.items()
    ]
    with get_db() as db:
        db.execute(
            f"UPDATE provider_models SET {assignments} WHERE provider_id = ? AND model_id = ?",
            (*values, provider_id, model_id),
        )


def _row(provider_id, model_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        ).fetchone()


# -- GET /metadata-candidates (read-only, §4.3) --

def test_metadata_candidates_requires_login(temp_db):
    pid = _seed_custom_model("candidates-auth")
    response = client.get(f"/admin/providers/{pid}/models/gpt-4o/metadata-candidates")
    assert response.status_code == 401


def test_metadata_candidates_returns_200_with_candidates(temp_db, monkeypatch):
    pid = _seed_custom_model("candidates-ok")
    from app.services import model_metadata as mm

    monkeypatch.setattr(mm, "resolve_model_metadata", lambda pid, mid: {
        "context_length": 128000,
        "max_output_tokens": 16384,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "input_price": 2.5,
        "output_price": 10.0,
        "cached_input_price": 1.25,
    })

    response = client.get(
        f"/admin/providers/{pid}/models/gpt-4o/metadata-candidates",
        headers=temp_db["headers"],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidates"]["context_length"] == 128000
    assert data["candidates"]["input_price"] == pytest.approx(2.5)
    assert data["candidates"]["input_modalities"] == ["text", "image"]


def test_metadata_candidates_is_read_only(temp_db, monkeypatch):
    """§4.3: 只返回候选值与'将被覆盖的现有值'供前端 diff，不写库。"""
    pid = _seed_custom_model("candidates-readonly")
    _set_metadata(pid, "gpt-4o", context_length=999, input_price=1.0)

    from app.services import model_metadata as mm
    monkeypatch.setattr(mm, "resolve_model_metadata", lambda pid, mid: {
        "context_length": 128000,
        "input_price": 2.5,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "max_output_tokens": None,
        "output_price": None,
        "cached_input_price": None,
    })

    response = client.get(
        f"/admin/providers/{pid}/models/gpt-4o/metadata-candidates",
        headers=temp_db["headers"],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidates"]["context_length"] == 128000
    assert data["current"]["context_length"] == 999
    assert data["candidates"]["input_price"] == pytest.approx(2.5)
    assert data["current"]["input_price"] == pytest.approx(1.0)

    # DB must not have changed
    assert _row(pid, "gpt-4o")["context_length"] == 999


def test_metadata_candidates_includes_diff_metadata(temp_db, monkeypatch):
    """The response should include a diff hint so the frontend can show what changed."""
    pid = _seed_custom_model("candidates-diff")
    _set_metadata(pid, "gpt-4o", context_length=64000)

    from app.services import model_metadata as mm
    monkeypatch.setattr(mm, "resolve_model_metadata", lambda pid, mid: {
        "context_length": 128000,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "max_output_tokens": None,
        "input_price": None,
        "output_price": None,
        "cached_input_price": None,
    })

    response = client.get(
        f"/admin/providers/{pid}/models/gpt-4o/metadata-candidates",
        headers=temp_db["headers"],
    )
    data = response.json()
    assert "candidates" in data
    assert "current" in data
    assert data["current"]["context_length"] == 64000
    assert data["candidates"]["context_length"] == 128000


def test_metadata_candidates_404_for_unknown_provider(temp_db):
    response = client.get(
        "/admin/providers/ghost/models/gpt-4o/metadata-candidates",
        headers=temp_db["headers"],
    )
    assert response.status_code == 404


def test_metadata_candidates_404_for_unknown_model(temp_db):
    pid = _seed_custom_model("candidates-404")
    response = client.get(
        f"/admin/providers/{pid}/models/ghost/metadata-candidates",
        headers=temp_db["headers"],
    )
    assert response.status_code == 404


# -- PUT /capabilities (custom-only write path, §4.4) --

def test_put_capabilities_requires_login(temp_db):
    pid = _seed_custom_model("cap-auth")
    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={"context_length": 128000},
    )
    assert response.status_code == 401


def test_put_capabilities_writes_for_custom_model(temp_db):
    """§6 用例 11: custom model -> capabilities endpoint correctly writes metadata."""
    pid = _seed_custom_model("cap-custom")
    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={
            "context_length": 128000,
            "max_output_tokens": 16384,
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "input_price": 2.5,
            "output_price": 10.0,
            "cached_input_price": 1.25,
        },
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 128000
    assert row["max_output_tokens"] == 16384
    assert row["input_price"] == pytest.approx(2.5)
    assert row["output_price"] == pytest.approx(10.0)
    assert row["cached_input_price"] == pytest.approx(1.25)
    assert json.loads(row["input_modalities"]) == ["text", "image"]


def test_put_capabilities_writes_auto_model(temp_db):
    """Auto rows are editable; their values just get overwritten again by the
    next upstream refresh."""
    pid = _seed_auto_model("cap-write")
    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={"context_length": 128000},
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    # DB must have been updated
    assert _row(pid, "gpt-4o")["context_length"] == 128000


def test_put_capabilities_404_for_unknown_provider(temp_db):
    response = client.put(
        "/admin/providers/ghost/models/gpt-4o/capabilities",
        json={"context_length": 128000},
        headers=temp_db["headers"],
    )
    assert response.status_code == 404


def test_put_capabilities_404_for_unknown_model(temp_db):
    pid = _seed_custom_model("cap-404")
    response = client.put(
        f"/admin/providers/{pid}/models/ghost/capabilities",
        json={"context_length": 128000},
        headers=temp_db["headers"],
    )
    assert response.status_code == 404


def test_put_capabilities_partial_update_does_not_clear_other_fields(temp_db):
    """PUT with only one field must not set the others to NULL."""
    pid = _seed_custom_model("cap-partial")
    _set_metadata(pid, "gpt-4o", context_length=64000, input_price=1.0)

    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={"context_length": 128000},
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 128000
    assert row["input_price"] == pytest.approx(1.0)  # must not be cleared


# -- mode=fill_empty (后端防线, §4.4 四轮 #3) --

def test_fill_empty_does_not_overwrite_existing_values(temp_db):
    """mode=fill_empty: 仅填空不覆盖非 NULL, 防绕过前端全量覆盖。"""
    pid = _seed_custom_model("fill-empty")
    _set_metadata(pid, "gpt-4o", context_length=64000, input_price=1.0)

    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={
            "mode": "fill_empty",
            "context_length": 999999,
            "max_output_tokens": 16384,
            "input_price": 0.01,
            "output_price": 10.0,
        },
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 64000          # unchanged (was non-NULL)
    assert row["input_price"] == pytest.approx(1.0)  # unchanged
    assert row["max_output_tokens"] == 16384         # filled (was NULL)
    assert row["output_price"] == pytest.approx(10.0)  # filled (was NULL)


def test_fill_empty_fills_null_fields(temp_db):
    pid = _seed_custom_model("fill-null")
    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={
            "mode": "fill_empty",
            "context_length": 128000,
            "max_output_tokens": 16384,
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 128000
    assert row["max_output_tokens"] == 16384
    assert json.loads(row["input_modalities"]) == ["text", "image"]


def test_fill_empty_is_noop_when_all_fields_are_present(temp_db):
    pid = _seed_custom_model("fill-noop")
    _set_metadata(pid, "gpt-4o", context_length=1, input_price=0.01)

    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={
            "mode": "fill_empty",
            "context_length": 999999,
            "input_price": 99.0,
        },
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 1
    assert row["input_price"] == pytest.approx(0.01)


def test_default_mode_is_full_overwrite(temp_db):
    """PUT without mode=fill_empty is a full overwrite (default semantics)."""
    pid = _seed_custom_model("overwrite-default")
    _set_metadata(pid, "gpt-4o", context_length=64000, input_price=1.0)

    response = client.put(
        f"/admin/providers/{pid}/models/gpt-4o/capabilities",
        json={"context_length": 128000},
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 128000
    # input_price was NOT in the request body, so NULL is not written.
    # The design says "按键存在才写" — only keys present in the request body
    # are touched. input_price is absent → stays as-is.
    assert row["input_price"] == pytest.approx(1.0)


# -- PUT /providers/{id} (existing models array) must not clear metadata (§4.4 / §6 用例 4/11) --

def test_put_provider_models_does_not_clear_metadata(temp_db):
    """§6 用例 11: PUT 全量 models 数组不洗掉 metadata."""
    pid = _seed_custom_model("put-protect")
    _set_metadata(pid, "gpt-4o", context_length=64000, input_price=1.0)

    # This is the existing PUT /admin/providers/{id} used by saveModelEdit.
    # It must NOT silently clear the metadata fields.
    response = client.put(
        f"/admin/providers/{pid}",
        json={
            "models": [
                {"id": "gpt-4o", "name": "Updated Name", "enabled": True, "source": "custom"},
            ],
        },
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 64000
    assert row["input_price"] == pytest.approx(1.0)
    assert row["model_name"] == "Updated Name"


def test_put_provider_models_does_not_wipe_auto_metadata(temp_db):
    """Even for auto models, the existing PUT must not clear metadata.

    This is symmetric: the PUT /providers/{id} handler writes only model_name,
    enabled, preprocessor, image_generation, source — it must never touch the
    capability columns.
    """
    pid = _seed_auto_model("put-auto-protect")
    _set_metadata(pid, "gpt-4o", context_length=128000, input_price=2.5)

    response = client.put(
        f"/admin/providers/{pid}",
        json={
            "models": [
                {"id": "gpt-4o", "name": "Auto Updated", "enabled": True},
            ],
        },
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 128000
    assert row["input_price"] == pytest.approx(2.5)


def test_put_provider_without_models_array_does_not_touch_metadata(temp_db):
    """Updating the provider name alone must not touch any model metadata."""
    pid = _seed_custom_model("put-name-only")
    _set_metadata(pid, "gpt-4o", context_length=64000)

    response = client.put(
        f"/admin/providers/{pid}",
        json={"name": "Renamed Provider"},
        headers=temp_db["headers"],
    )
    assert response.status_code == 200

    row = _row(pid, "gpt-4o")
    assert row["context_length"] == 64000


# -- GET /providers returns new fields (§4.4) --

def test_list_providers_returns_metadata_fields(temp_db):
    """Existing GET 模型列表返回新字段，modalities 须 json.loads。"""
    pid = _seed_custom_model("list-metadata")
    _set_metadata(pid, "gpt-4o",
                  context_length=128000,
                  max_output_tokens=16384,
                  input_modalities=["text", "image"],
                  output_modalities=["text"],
                  input_price=2.5,
                  output_price=10.0,
                  cached_input_price=1.25)

    response = client.get("/admin/providers", headers=temp_db["headers"])
    assert response.status_code == 200
    providers = {p["id"]: p for p in response.json()}
    model = {m["id"]: m for m in providers[pid]["models"]}["gpt-4o"]

    assert model["context_length"] == 128000
    assert model["max_output_tokens"] == 16384
    assert model["input_modalities"] == ["text", "image"]
    assert model["output_modalities"] == ["text"]
    assert model["input_price"] == pytest.approx(2.5)
    assert model["output_price"] == pytest.approx(10.0)
    assert model["cached_input_price"] == pytest.approx(1.25)


def test_null_metadata_appears_as_none_in_list(temp_db):
    """Models without metadata must not crash the list endpoint."""
    pid = _seed_custom_model("list-null")
    response = client.get("/admin/providers", headers=temp_db["headers"])
    providers = {p["id"]: p for p in response.json()}
    model = {m["id"]: m for m in providers[pid]["models"]}["gpt-4o"]
    assert model["context_length"] is None
    assert model["input_modalities"] == []
    assert model["input_price"] is None


# -- XSS: frontend escaping (§6 用例 8) --

def test_static_js_escapes_candidate_values():
    """§6 用例 8: 恶意候选值回填进 innerHTML 被转义.

    We scan the frontend source for the pattern that renders candidate metadata
    values and verify it uses textContent or innerText or a sanitizer, not raw
    innerHTML assignment from a user-controlled variable.
    """
    import pathlib
    js_path = pathlib.Path(__file__).resolve().parents[1] / "app" / "web" / "static" / "app.js"
    src = js_path.read_text(encoding="utf-8")

    # Look for the candidate-rendering pattern. The candidate value comes from
    # an external API response (models.dev / OpenRouter) and must be HTML-escaped.
    # If the code uses textContent, innerText, or calls a sanitizer it's safe.
    # If it uses innerHTML with a value from an external source, it's a finding.
    unsafe_patterns = []
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        # Only guard candidate/capability-rendering code (design §6 用例 8).
        # The candidate value comes from an external API (models.dev /
        # OpenRouter) and must be HTML-escaped. Pre-existing admin builders
        # (providerFormHtml, .map(...) chains, locally-built `html` strings)
        # escape internally and are out of scope — a whole-file scan of
        # .innerHTML assignments can't distinguish them from unsafe code.
        # Keep candidate_terms aligned with the identifiers the M4 frontend
        # sub-step uses (candidate / metadata / capability / sync); extend if
        # the frontend names its variables differently.
        candidate_terms = ("candidate", "metadata", "capabilit", "sync")
        if ".innerHTML" not in stripped or "=" not in stripped:
            continue
        if not any(term in stripped.lower() for term in candidate_terms):
            continue
        rhs = stripped.split("=", 1)[1].strip()
        # If the RHS is a string literal (quoted), it's safe.
        if rhs.startswith(('"', "'", "`")):
            continue
        # If the RHS is a constant/known safe value, skip.
        safe_rhs = {"''", '""', "null", "undefined", "true", "false", "0", '""'}
        if rhs.rstrip(";") in safe_rhs:
            continue
        # If the RHS is a sanitizer call, skip. Project sanitizers: escHtml
        # (HTML entity escape, app.js) and jsEsc (JS string escape).
        if any(s in rhs for s in ("escapeHtml", "escHtml", "jsEsc", "sanitize", "textContent", "DOMPurify")):
            continue
        unsafe_patterns.append(f"  line {i}: {stripped}")

    if unsafe_patterns:
        pytest.fail(
            f"Potential XSS: {len(unsafe_patterns)} innerHTML assignment(s) "
            f"with non-literal RHS found in {js_path}:\n"
            + "\n".join(unsafe_patterns)
        )