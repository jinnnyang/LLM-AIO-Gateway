"""
HTTP API tests for the cascading model-id rename endpoint.

Endpoint: PUT /admin/providers/{provider_id}/models/{model_id}/rename

Contract these tests pin down:
- The endpoint requires an admin session (no session -> 401, never a silent write).
- Error codes map to HTTP correctly: unknown provider/model -> 404, everything
  else (duplicate id, auto row, malformed new id) -> 400.
- A successful rename returns the cascade counts and the warning list so the UI
  can surface references it deliberately did not rewrite.
- The rename is by BARE upstream id; the "provider/" prefix is addressing syntax.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.database import init_db, get_db, add_provider, add_admin, get_provider
from app.security import create_session, hash_password


client = TestClient(app)

SAMPLE_PROVIDER = {
    "id": "renprov",
    "name": "Rename Provider",
    "provider_type": "openai",
    "api_base": "https://api.test.com/v1",
    "api_key": "sk-test-key",
    "enabled": True,
    "models": [],
}


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Isolated database with an admin session for every test."""
    db_path = str(tmp_path / "test_rename_api.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs",
                    "retention_days": 30, "console": False},
    }
    config.save()
    init_db(db_path)
    add_admin("admin", hash_password("secret"), "Admin")
    token = create_session("admin")
    yield {"headers": {"Authorization": f"Bearer {token}"}}


def _seed(provider_id="renprov", model_id="old-model", source="custom"):
    add_provider({
        **SAMPLE_PROVIDER,
        "id": provider_id,
        "models": [{"id": model_id, "name": model_id, "enabled": True,
                    "source": source}],
    })
    return provider_id


def _url(provider_id, model_id):
    return "/admin/providers/%s/models/%s/rename" % (provider_id, model_id)


def _model_ids(provider_id):
    provider = get_provider(provider_id)
    return [m["id"] for m in (provider or {}).get("models", [])]


# --------------------------------------------------------------------------
# authentication
# --------------------------------------------------------------------------

def test_rename_requires_admin_session(temp_db):
    """An unauthenticated rename must be rejected AND must not write."""
    _seed()
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "hacked"})
    assert res.status_code in (401, 403)
    assert _model_ids("renprov") == ["old-model"]


def test_rename_rejects_bogus_token(temp_db):
    _seed()
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "hacked"},
                     headers={"Authorization": "Bearer not-a-real-token"})
    assert res.status_code in (401, 403)
    assert _model_ids("renprov") == ["old-model"]


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------

def test_rename_succeeds_and_reports_shape(temp_db):
    _seed()
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "new-model"}, **temp_db)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "renamed"
    assert body["provider_id"] == "renprov"
    assert body["old_model_id"] == "old-model"
    assert body["model"]["id"] == "new-model"
    # Both keys must always be present: the UI reads them unconditionally.
    assert isinstance(body["updated"], dict)
    assert isinstance(body["warnings"], list)
    assert _model_ids("renprov") == ["new-model"]


def test_rename_accepts_legacy_model_id_key(temp_db):
    """Older clients send `model_id` instead of `new_model_id`."""
    _seed()
    res = client.put(_url("renprov", "old-model"),
                     json={"model_id": "new-model"}, **temp_db)
    assert res.status_code == 200
    assert _model_ids("renprov") == ["new-model"]


def test_rename_reports_cascade_counts(temp_db):
    _seed()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, "
            "target_model, target_provider) VALUES (?,?,?,?,?,?)",
            ("r1", "r", 1, "renprov/old-model", "renprov/old-model", "renprov"),
        )
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "new-model"}, **temp_db)
    assert res.status_code == 200
    updated = res.json()["updated"]
    assert updated["routing_rules.match_model"] == 1
    assert updated["routing_rules.target_model"] == 1


def test_rename_surfaces_wildcard_warning(temp_db):
    """A wildcard that stops matching is reported, not silently left broken."""
    _seed()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, "
            "target_model, target_provider) VALUES (?,?,?,?,?,?)",
            ("r-wild", "wild", 1, "old-*", "", "renprov"),
        )
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "brand-new"}, **temp_db)
    assert res.status_code == 200
    kinds = [w["kind"] for w in res.json()["warnings"]]
    assert "wildcard_no_longer_matches" in kinds


# --------------------------------------------------------------------------
# error mapping
# --------------------------------------------------------------------------

def test_unknown_provider_is_404(temp_db):
    res = client.put(_url("ghost-provider", "old-model"),
                     json={"new_model_id": "x"}, **temp_db)
    assert res.status_code == 404


def test_unknown_model_is_404(temp_db):
    _seed()
    res = client.put(_url("renprov", "ghost-model"),
                     json={"new_model_id": "x"}, **temp_db)
    assert res.status_code == 404


def test_duplicate_target_is_400(temp_db):
    add_provider({
        **SAMPLE_PROVIDER,
        "models": [
            {"id": "a", "name": "a", "enabled": True, "source": "custom"},
            {"id": "b", "name": "b", "enabled": True, "source": "custom"},
        ],
    })
    res = client.put(_url("renprov", "a"),
                     json={"new_model_id": "b"}, **temp_db)
    assert res.status_code == 400
    assert sorted(_model_ids("renprov")) == ["a", "b"]


def test_auto_model_is_400(temp_db):
    """Auto rows are owned by refresh discovery, so their id is not editable."""
    _seed(source="auto")
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "new-model"}, **temp_db)
    assert res.status_code == 400
    assert _model_ids("renprov") == ["old-model"]


@pytest.mark.parametrize("bad", ["", "   "])
def test_invalid_new_id_is_400(temp_db, bad):
    _seed()
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": bad}, **temp_db)
    assert res.status_code == 400
    assert _model_ids("renprov") == ["old-model"]


def test_prefix_of_existing_provider_is_400(temp_db):
    """A head that names a real provider is a move, not a rename -> reject.

    A head that is NOT a provider (e.g. "des/") is an upstream namespace and
    must be allowed, so this check is existence-based rather than syntactic.
    """
    _seed()
    add_provider({**SAMPLE_PROVIDER, "id": "openai",
                  "models": [{"id": "gpt-4o", "name": "gpt-4o",
                              "enabled": True, "source": "custom"}]})
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "openai/gpt-4o"}, **temp_db)
    assert res.status_code == 400
    assert _model_ids("renprov") == ["old-model"]


def test_non_provider_prefix_is_accepted(temp_db):
    _seed()
    res = client.put(_url("renprov", "old-model"),
                     json={"new_model_id": "des/deepseek"}, **temp_db)
    assert res.status_code == 200
    assert _model_ids("renprov") == ["des/deepseek"]



def test_missing_body_key_is_400(temp_db):
    _seed()
    res = client.put(_url("renprov", "old-model"), json={}, **temp_db)
    assert res.status_code == 400
    assert _model_ids("renprov") == ["old-model"]


# --------------------------------------------------------------------------
# slash-bearing upstream ids
#
# When a provider namespaces its models with a path, the BARE upstream id is
# itself "des/deepseek". Two consequences drive these tests:
#
# 1. The old id cannot travel in the URL path. ASGI servers decode %2F BEFORE
#    route matching, so an encoded slash splits the path segment and the
#    request lands on no route at all (405). That is why the rename endpoint
#    accepts the old id in the BODY.
# 2. Sibling endpoints (capabilities, delete, metadata-candidates) still take
#    the id in the path, so they declare {model_id:path} to allow slashes.
# --------------------------------------------------------------------------

SLASH_ID = "des/deepseek"


def _seed_slash(model_id=SLASH_ID, source="custom"):
    add_provider({
        **SAMPLE_PROVIDER,
        "models": [{"id": model_id, "name": model_id, "enabled": True,
                    "source": source}],
    })


def test_body_rename_handles_slash_id(temp_db):
    _seed_slash()
    res = client.put("/admin/providers/renprov/models/rename",
                     json={"old_model_id": SLASH_ID,
                           "new_model_id": "des/deepseek-v3"}, **temp_db)
    assert res.status_code == 200
    assert res.json()["model"]["id"] == "des/deepseek-v3"
    assert _model_ids("renprov") == ["des/deepseek-v3"]


def test_body_rename_accepts_own_composite_old_id(temp_db):
    """The UI shows "renprov/des/deepseek"; sending that must work too."""
    _seed_slash()
    res = client.put("/admin/providers/renprov/models/rename",
                     json={"old_model_id": "renprov/" + SLASH_ID,
                           "new_model_id": "des/deepseek-v3"}, **temp_db)
    assert res.status_code == 200
    assert _model_ids("renprov") == ["des/deepseek-v3"]


def test_body_rename_requires_admin_session(temp_db):
    _seed_slash()
    res = client.put("/admin/providers/renprov/models/rename",
                     json={"old_model_id": SLASH_ID, "new_model_id": "hacked"})
    assert res.status_code in (401, 403)
    assert _model_ids("renprov") == [SLASH_ID]


def test_body_rename_missing_old_id_is_400(temp_db):
    _seed_slash()
    res = client.put("/admin/providers/renprov/models/rename",
                     json={"new_model_id": "x"}, **temp_db)
    assert res.status_code == 400
    assert _model_ids("renprov") == [SLASH_ID]


def test_body_rename_rejects_foreign_provider_prefix(temp_db):
    _seed_slash()
    add_provider({**SAMPLE_PROVIDER, "id": "openai",
                  "models": [{"id": "gpt-4o", "name": "gpt-4o",
                              "enabled": True, "source": "custom"}]})
    res = client.put("/admin/providers/renprov/models/rename",
                     json={"old_model_id": SLASH_ID,
                           "new_model_id": "openai/gpt-4o"}, **temp_db)
    assert res.status_code == 400
    assert _model_ids("renprov") == [SLASH_ID]


def test_capabilities_endpoint_accepts_slash_id(temp_db):
    """Without {model_id:path} this returned 405 and metadata could not be saved."""
    _seed_slash()
    res = client.put("/admin/providers/renprov/models/%s/capabilities" % SLASH_ID,
                     json={"context_length": 12345}, **temp_db)
    assert res.status_code == 200
    provider = get_provider("renprov")
    assert provider["models"][0]["context_length"] == 12345


def test_delete_endpoint_accepts_slash_id(temp_db):
    """Without {model_id:path} a slash model could not be deleted at all."""
    _seed_slash()
    res = client.delete("/admin/providers/renprov/models/%s" % SLASH_ID, **temp_db)
    assert res.status_code == 200
    assert _model_ids("renprov") == []


def test_metadata_candidates_endpoint_accepts_slash_id(temp_db):
    _seed_slash()
    res = client.get(
        "/admin/providers/renprov/models/%s/metadata-candidates" % SLASH_ID,
        **temp_db)
    assert res.status_code == 200


def test_slash_path_endpoints_still_require_auth(temp_db):
    """The :path converter must not have widened the routes past the auth gate."""
    _seed_slash()
    caps = client.put(
        "/admin/providers/renprov/models/%s/capabilities" % SLASH_ID,
        json={"context_length": 1})
    dele = client.delete("/admin/providers/renprov/models/%s" % SLASH_ID)
    assert caps.status_code in (401, 403)
    assert dele.status_code in (401, 403)
    assert _model_ids("renprov") == [SLASH_ID]


def test_deep_multi_segment_id_via_path_endpoints(temp_db):
    _seed_slash(model_id="a/b/c")
    caps = client.put("/admin/providers/renprov/models/a/b/c/capabilities",
                      json={"context_length": 7}, **temp_db)
    assert caps.status_code == 200
    dele = client.delete("/admin/providers/renprov/models/a/b/c", **temp_db)
    assert dele.status_code == 200
    assert _model_ids("renprov") == []


def test_flat_id_endpoints_unaffected(temp_db):
    """Regression guard: the :path change must not break plain ids."""
    _seed()
    assert client.get(
        _url("renprov", "old-model").replace("/rename", "/metadata-candidates"),
        **temp_db).status_code == 200
    assert client.put(
        "/admin/providers/renprov/models/old-model/capabilities",
        json={"context_length": 9}, **temp_db).status_code == 200
    assert client.put(_url("renprov", "old-model"),
                      json={"new_model_id": "new-model"},
                      **temp_db).status_code == 200
    assert client.delete("/admin/providers/renprov/models/new-model",
                         **temp_db).status_code == 200
