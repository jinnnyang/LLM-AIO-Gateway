"""Tests for cascading model-id rename (rename_provider_model).

Confirmed semantics (2026-08-28):
1. provider_models.model_id stores the BARE upstream id. The "provider/" prefix
   is addressing syntax derived from provider_id, never part of the stored id,
   so a rename never touches it.
2. Only custom models can be renamed; auto rows are owned by refresh discovery.
3. Renaming cascades to EXACT references in: routing_rules(match_model,
   target_model), fallback_policies(match_model, chain),
   image_generators(provider_model), user_api_keys(allowed_models).
4. It deliberately does NOT touch preprocessors.model or image_generators.model
   (independent upstream endpoints with their own api_base/api_key), nor
   request_logs/request_records (audit history).
5. Wildcard rules are never rewritten; losing a match is reported as a warning.
6. A bare id also exposed by another provider cannot be attributed, so bare
   references are left alone and reported as a warning.
7. Capability metadata survives the rename; responses_* probe cache is cleared.
"""
import json

import pytest

import app.database as db_mod
from app.database import (
    add_provider, get_provider, init_db, rename_provider_model,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Isolate each test in a throwaway SQLite file."""
    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    db_mod._initialized = False
    init_db(str(tmp_path / "rename.db"))
    try:
        yield
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized


def _conn():
    return db_mod.get_db()


def _seed_provider(pid="vcp", models=None):
    add_provider({
        "id": pid,
        "name": pid,
        "api_base": "https://example.invalid/v1",
        "api_key": "sk-test",
        "models": models if models is not None else [
            {"id": "old-model", "name": "Old Model"},
        ],
    })


def _model_row(pid, mid):
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM provider_models WHERE provider_id=? AND model_id=?",
            (pid, mid),
        ).fetchone()


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------

def test_rename_updates_primary_row_and_keeps_capabilities():
    _seed_provider()
    with _conn() as conn:
        conn.execute(
            "UPDATE provider_models SET context_length=9999, input_price=1.5 "
            "WHERE provider_id='vcp' AND model_id='old-model'"
        )

    res = rename_provider_model("vcp", "old-model", "new-model")

    assert res["ok"] is True
    assert _model_row("vcp", "old-model") is None
    row = _model_row("vcp", "new-model")
    assert row is not None
    # Capability metadata must survive: it is the whole point of renaming in
    # place instead of delete + recreate.
    assert row["context_length"] == 9999
    assert row["input_price"] == 1.5


def test_rename_accepts_composite_old_id():
    """Callers holding a display id like 'vcp/old-model' must work too."""
    _seed_provider()
    res = rename_provider_model("vcp", "vcp/old-model", "new-model")
    assert res["ok"] is True
    assert _model_row("vcp", "new-model") is not None


def test_rename_to_same_id_is_a_noop():
    _seed_provider()
    res = rename_provider_model("vcp", "old-model", "old-model")
    assert res["ok"] is True
    assert res["updated"] == {}
    assert _model_row("vcp", "old-model") is not None


def test_default_display_name_follows_rename_but_custom_name_is_kept():
    _seed_provider(models=[
        {"id": "auto-named", "name": "auto-named"},   # name == id -> default
        {"id": "hand-named", "name": "My Nice Name"},  # user's own label
    ])

    rename_provider_model("vcp", "auto-named", "auto-renamed")
    rename_provider_model("vcp", "hand-named", "hand-renamed")

    assert _model_row("vcp", "auto-renamed")["model_name"] == "auto-renamed"
    assert _model_row("vcp", "hand-renamed")["model_name"] == "My Nice Name"


def test_rename_clears_responses_probe_cache():
    """The probe result describes the OLD upstream id, so carrying it over to a
    renamed model would assert capabilities that were never tested. Cleared
    means "back to the schema default", not NULL: these columns are NOT NULL."""
    _seed_provider()
    probe_defaults = {
        "responses_status": "unknown",
        "responses_checked_at": "",
        "responses_expires_at": "",
        "responses_streaming": 0,
        "responses_streaming_status": "unknown",
        "responses_tool_types": "[]",
        "responses_error": "",
    }
    with _conn() as conn:
        conn.execute(
            "UPDATE provider_models SET responses_status='supported', "
            "responses_checked_at='2026-01-01T00:00:00Z', "
            "responses_expires_at='2027-01-01T00:00:00Z', "
            "responses_streaming=1, responses_streaming_status='supported', "
            "responses_tool_types='[\"web_search\"]', responses_error='boom' "
            "WHERE provider_id='vcp' AND model_id='old-model'"
        )

    rename_provider_model("vcp", "old-model", "new-model")

    row = _model_row("vcp", "new-model")
    for col, default in probe_defaults.items():
        assert row[col] == default, "%s should be reset to %r" % (col, default)


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------

def test_rename_rejects_auto_model():
    _seed_provider()
    with _conn() as conn:
        conn.execute(
            "UPDATE provider_models SET source='auto' "
            "WHERE provider_id='vcp' AND model_id='old-model'"
        )
    res = rename_provider_model("vcp", "old-model", "new-model")
    assert res == {"ok": False, "error": "auto_model"}


def test_rename_rejects_duplicate_id():
    _seed_provider(models=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}])
    res = rename_provider_model("vcp", "a", "b")
    assert res == {"ok": False, "error": "duplicate_model"}
    # neither row may be destroyed by the failed attempt
    assert _model_row("vcp", "a") is not None
    assert _model_row("vcp", "b") is not None


def test_rename_rejects_unknown_provider_and_model():
    _seed_provider()
    assert rename_provider_model("nope", "old-model", "x") == {
        "ok": False, "error": "provider_not_found"}
    assert rename_provider_model("vcp", "ghost", "x") == {
        "ok": False, "error": "model_not_found"}


@pytest.mark.parametrize("bad", ["", "   "])
def test_rename_rejects_empty_new_id(bad):
    """An empty new id is meaningless and must not clear the stored id."""
    _seed_provider()
    res = rename_provider_model("vcp", "old-model", bad)
    assert res == {"ok": False, "error": "invalid_new_id"}
    assert _model_row("vcp", "old-model") is not None


def test_rename_rejects_prefix_of_a_real_provider():
    """A foreign 'provider/' prefix must not be silently stripped: that would
    make 'move to another provider' look like a plain rename.

    The head is only rejected when it names a provider that actually exists --
    an arbitrary head like "des/" is an upstream namespace and is allowed.
    """
    _seed_provider()
    _seed_provider(pid="openai", models=[{"id": "gpt-4o", "name": "gpt-4o"}])
    res = rename_provider_model("vcp", "old-model", "openai/gpt-4o")
    assert res == {"ok": False, "error": "foreign_provider_prefix"}
    assert _model_row("vcp", "old-model") is not None



def test_rename_allows_own_prefix_on_new_id():
    _seed_provider()
    res = rename_provider_model("vcp", "old-model", "vcp/new-model")
    assert res["ok"] is True
    assert _model_row("vcp", "new-model") is not None


# --------------------------------------------------------------------------
# cascade
# --------------------------------------------------------------------------

def _seed_references():
    _seed_provider()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, target_model, "
            "target_provider) VALUES (?,?,?,?,?,?)",
            ("r-exact", "exact composite", 1, "vcp/old-model", "vcp/old-model", "vcp"),
        )
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, target_model, "
            "target_provider) VALUES (?,?,?,?,?,?)",
            ("r-bare", "exact bare", 1, "old-model", "", "vcp"),
        )
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, target_model, "
            "target_provider) VALUES (?,?,?,?,?,?)",
            ("r-wild", "wildcard", 1, "old-*", "", "vcp"),
        )
        conn.execute(
            "INSERT INTO fallback_policies (id, name, enabled, match_model, chain) "
            "VALUES (?,?,?,?,?)",
            ("f1", "fb", 1, "vcp/old-model",
             json.dumps(["vcp/old-model", {"model": "vcp/old-model",
                                           "provider_id": "vcp"}])),
        )
        conn.execute(
            "INSERT INTO image_generators (id, backend_type, provider_model, model, "
            "enabled) VALUES (?,?,?,?,?)",
            ("img1", "openai", "vcp/old-model", "sd-xl", 1),
        )
        conn.execute(
            "INSERT INTO preprocessors (id, api_base, model, enabled) VALUES (?,?,?,?)",
            ("pre1", "https://other.invalid/v1", "old-model", 1),
        )
        conn.execute(
            "INSERT OR IGNORE INTO users (username, display_name) VALUES (?,?)",
            ("alice", "Alice"),
        )
        conn.execute(
            "INSERT INTO user_api_keys (key, username, allowed_models) VALUES (?,?,?)",
            ("sk-alice", "alice", json.dumps(["vcp/old-model", "other/keep-me"])),
        )


def test_cascade_rewrites_exact_references():
    _seed_references()
    res = rename_provider_model("vcp", "old-model", "new-model")
    assert res["ok"] is True

    with _conn() as conn:
        rules = {r["id"]: r for r in conn.execute("SELECT * FROM routing_rules")}
        assert rules["r-exact"]["match_model"] == "vcp/new-model"
        assert rules["r-exact"]["target_model"] == "vcp/new-model"
        assert rules["r-bare"]["match_model"] == "new-model"

        fb = conn.execute("SELECT * FROM fallback_policies").fetchone()
        assert fb["match_model"] == "vcp/new-model"
        chain = json.loads(fb["chain"])
        assert chain[0] == "vcp/new-model"
        assert chain[1]["model"] == "vcp/new-model"

        img = conn.execute("SELECT * FROM image_generators").fetchone()
        assert img["provider_model"] == "vcp/new-model"

        key = conn.execute("SELECT * FROM user_api_keys").fetchone()
        assert json.loads(key["allowed_models"]) == ["vcp/new-model", "other/keep-me"]


def test_cascade_leaves_independent_upstream_fields_alone():
    """preprocessors.model and image_generators.model point at their OWN
    upstream endpoint (they carry api_base/api_key), so they are not gateway
    model references and must never be rewritten."""
    _seed_references()
    rename_provider_model("vcp", "old-model", "new-model")
    with _conn() as conn:
        assert conn.execute("SELECT model FROM preprocessors").fetchone()[0] == "old-model"
        assert conn.execute("SELECT model FROM image_generators").fetchone()[0] == "sd-xl"


def test_cascade_reports_counts():
    _seed_references()
    res = rename_provider_model("vcp", "old-model", "new-model")
    updated = res["updated"]
    assert updated["routing_rules.match_model"] == 2
    assert updated["routing_rules.target_model"] == 1
    assert updated["fallback_policies.match_model"] == 1
    assert updated["fallback_policies.chain"] == 1
    assert updated["image_generators.provider_model"] == 1
    assert updated["user_api_keys.allowed_models"] == 1


# --------------------------------------------------------------------------
# warnings
# --------------------------------------------------------------------------

def test_wildcard_losing_its_match_is_reported_not_rewritten():
    _seed_references()
    res = rename_provider_model("vcp", "old-model", "brand-new")

    with _conn() as conn:
        wild = conn.execute(
            "SELECT match_model FROM routing_rules WHERE id='r-wild'").fetchone()
    assert wild["match_model"] == "old-*", "user's pattern must not be edited"

    kinds = [w["kind"] for w in res["warnings"]]
    assert "wildcard_no_longer_matches" in kinds
    w = next(x for x in res["warnings"] if x["kind"] == "wildcard_no_longer_matches")
    assert w["pattern"] == "old-*"


def test_wildcard_still_matching_produces_no_warning():
    _seed_references()
    # 'old-*' still matches 'old-renamed', so nothing silently breaks.
    res = rename_provider_model("vcp", "old-model", "old-renamed")
    kinds = [w["kind"] for w in res["warnings"]]
    assert "wildcard_no_longer_matches" not in kinds


def test_bare_id_shared_with_another_provider_is_reported_not_rewritten():
    """An unprefixed reference cannot be attributed when two providers expose
    the same bare id, so rewriting it could silently reroute other traffic."""
    _seed_provider("vcp", models=[{"id": "shared", "name": "shared"}])
    _seed_provider("openai", models=[{"id": "shared", "name": "shared"}])
    with _conn() as conn:
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, target_model, "
            "target_provider) VALUES (?,?,?,?,?,?)",
            ("r-bare", "bare", 1, "shared", "", "vcp"),
        )

    res = rename_provider_model("vcp", "shared", "vcp-only")
    assert res["ok"] is True

    with _conn() as conn:
        rule = conn.execute("SELECT match_model FROM routing_rules").fetchone()
    assert rule["match_model"] == "shared", "ambiguous bare ref must be untouched"

    w = next(x for x in res["warnings"] if x["kind"] == "ambiguous_bare_id")
    assert "openai" in w["providers"]


def test_provider_still_reachable_after_rename():
    """End-to-end: the renamed model must be resolvable through get_provider."""
    _seed_provider()
    rename_provider_model("vcp", "old-model", "new-model")
    provider = get_provider("vcp")
    ids = [m["id"] for m in provider["models"]]
    assert "new-model" in ids
    assert "old-model" not in ids


# ---------------------------------------------------------------------------
# slash-bearing upstream ids
#
# Some providers namespace their models with a path, so the BARE upstream id is
# itself "des/deepseek" and the gateway-facing composite becomes
# "vcp/des/deepseek". A slash inside the bare id is data, not addressing
# syntax, so parse_model_id() must never be used to interpret it -- only the
# provider_id context can tell the two meanings apart.
# ---------------------------------------------------------------------------

SLASH_ID = "des/deepseek"


def test_slash_bare_id_can_be_renamed():
    _seed_provider(models=[{"id": SLASH_ID, "name": SLASH_ID}])
    res = rename_provider_model("vcp", SLASH_ID, "des/deepseek-v3")
    assert res["ok"] is True
    assert res["model"]["id"] == "des/deepseek-v3"
    assert [m["id"] for m in get_provider("vcp")["models"]] == ["des/deepseek-v3"]


def test_slash_id_accepts_own_composite_form():
    """The caller may pass the full "vcp/des/deepseek" it sees in the UI."""
    _seed_provider(models=[{"id": SLASH_ID, "name": SLASH_ID}])
    res = rename_provider_model("vcp", "vcp/" + SLASH_ID, "vcp/des/deepseek-v3")
    assert res["ok"] is True
    # Only the gateway prefix is stripped; the upstream path stays intact.
    assert res["model"]["id"] == "des/deepseek-v3"


def test_own_prefix_is_stripped_only_once():
    """"vcp/vcp/x" means upstream path "vcp/x", not a doubled prefix."""
    _seed_provider(models=[{"id": "plain", "name": "plain"}])
    res = rename_provider_model("vcp", "plain", "vcp/vcp/x")
    assert res["ok"] is True
    assert res["model"]["id"] == "vcp/x"


def test_flat_id_can_become_slash_id_and_back():
    _seed_provider(models=[{"id": "flat", "name": "flat"}])
    assert rename_provider_model("vcp", "flat", SLASH_ID)["ok"] is True
    assert rename_provider_model("vcp", SLASH_ID, "flat-again")["ok"] is True
    assert [m["id"] for m in get_provider("vcp")["models"]] == ["flat-again"]


def test_deep_multi_segment_id_is_allowed():
    _seed_provider(models=[{"id": "plain", "name": "plain"}])
    res = rename_provider_model("vcp", "plain", "a/b/c")
    assert res["ok"] is True
    assert res["model"]["id"] == "a/b/c"


def test_slash_rename_keeps_capability_metadata():
    # add_provider() does not accept capability columns, so write them directly
    # (same approach as test_rename_preserves_metadata above).
    _seed_provider(models=[{"id": SLASH_ID, "name": SLASH_ID}])
    with _conn() as conn:
        conn.execute(
            "UPDATE provider_models SET context_length=65536, input_price=0.5 "
            "WHERE provider_id='vcp' AND model_id=?", (SLASH_ID,),
        )
    assert rename_provider_model("vcp", SLASH_ID, "des/deepseek-v3")["ok"] is True
    row = _model_row("vcp", "des/deepseek-v3")
    assert row["context_length"] == 65536
    assert row["input_price"] == 0.5



def test_slash_rename_cascades_to_references():
    _seed_provider(models=[{"id": SLASH_ID, "name": SLASH_ID}])
    with _conn() as conn:
        conn.execute(
            "INSERT INTO routing_rules (id, name, enabled, match_model, "
            "target_model, target_provider) VALUES (?,?,?,?,?,?)",
            ("r1", "r", 1, "vcp/" + SLASH_ID, "vcp/" + SLASH_ID, "vcp"),
        )
    res = rename_provider_model("vcp", SLASH_ID, "des/deepseek-v3")
    assert res["ok"] is True
    assert res["updated"]["routing_rules.match_model"] == 1
    with _conn() as conn:
        rule = conn.execute("SELECT match_model, target_model FROM routing_rules").fetchone()
    assert rule["match_model"] == "vcp/des/deepseek-v3"
    assert rule["target_model"] == "vcp/des/deepseek-v3"


def test_duplicate_slash_id_is_rejected():
    _seed_provider(models=[
        {"id": SLASH_ID, "name": SLASH_ID},
        {"id": "des/other", "name": "des/other"},
    ])
    res = rename_provider_model("vcp", SLASH_ID, "des/other")
    assert res["ok"] is False
    assert res["error"] == "duplicate_model"


def test_foreign_provider_prefix_is_rejected():
    """A rename must not look like moving the model to another provider."""
    _seed_provider(models=[{"id": "plain", "name": "plain"}])
    _seed_provider(pid="openai", models=[{"id": "gpt-4o", "name": "gpt-4o"}])
    res = rename_provider_model("vcp", "plain", "openai/gpt-4o")
    assert res["ok"] is False
    assert res["error"] == "foreign_provider_prefix"
    assert [m["id"] for m in get_provider("vcp")["models"]] == ["plain"]


def test_slash_head_that_is_not_a_provider_is_accepted():
    """"des/" is an upstream namespace, not a provider id, so it is fine."""
    _seed_provider(models=[{"id": "plain", "name": "plain"}])
    res = rename_provider_model("vcp", "plain", SLASH_ID)
    assert res["ok"] is True
    assert res["model"]["id"] == SLASH_ID


# -- Resolved-metadata (L2) cache invalidation (P3-6) --------------------------
# rename_provider_model already clears the responses_* probe cache in the DB.
# The in-memory L2 cache in app.services.model_metadata is keyed by
# (provider_id, normalized_model_id) and was previously left to expire on its
# own 300s TTL, so the old id stayed resolvable and the new id could serve a
# stale pre-rename miss.

def _l2_key(provider_id, model_id):
    from app.services import model_metadata as mm
    return (provider_id, mm._normalize_model_id(model_id))


def test_rename_drops_the_old_l2_cache_entry():
    from app.services import model_metadata as mm
    mm.clear_caches()
    _seed_provider(models=[{"id": "old-name", "name": "old-name"}])
    mm._l2.set("vcp", "old-name", {"context_length": 128000})
    assert mm._l2.get("vcp", "old-name") is not None

    res = rename_provider_model("vcp", "old-name", "new-name")
    assert res["ok"] is True
    assert mm._l2.get("vcp", "old-name") is None, (
        "old L2 entry survived the rename; it now points at a model that "
        "no longer exists"
    )


def test_rename_drops_a_stale_entry_under_the_new_id():
    """A pre-rename miss cached under the new id must not outlive the rename."""
    from app.services import model_metadata as mm
    mm.clear_caches()
    _seed_provider(models=[{"id": "old-name", "name": "old-name"}])
    mm._l2.set("vcp", "new-name", {})
    assert mm._l2.get("vcp", "new-name") is not None

    assert rename_provider_model("vcp", "old-name", "new-name")["ok"] is True
    assert mm._l2.get("vcp", "new-name") is None


def test_rename_leaves_other_models_cached():
    """Invalidation is scoped to the two ids involved, not a blanket clear."""
    from app.services import model_metadata as mm
    mm.clear_caches()
    _seed_provider(models=[
        {"id": "old-name", "name": "old-name"},
        {"id": "bystander", "name": "bystander"},
    ])
    mm._l2.set("vcp", "bystander", {"context_length": 8192})
    mm._l2.set("other", "old-name", {"context_length": 4096})

    assert rename_provider_model("vcp", "old-name", "new-name")["ok"] is True
    assert mm._l2.get("vcp", "bystander") == {"context_length": 8192}
    assert mm._l2.get("other", "old-name") == {"context_length": 4096}, (
        "another provider's identically named model was invalidated"
    )


def test_rejected_rename_keeps_the_cache_intact():
    """A rename that never happened must not drop live cache entries."""
    from app.services import model_metadata as mm
    mm.clear_caches()
    _seed_provider(models=[
        {"id": "old-name", "name": "old-name"},
        {"id": "taken", "name": "taken"},
    ])
    mm._l2.set("vcp", "old-name", {"context_length": 128000})

    res = rename_provider_model("vcp", "old-name", "taken")
    assert res["ok"] is False and res["error"] == "duplicate_model"
    assert mm._l2.get("vcp", "old-name") == {"context_length": 128000}


def test_slash_id_rename_invalidates_the_normalized_key():
    """L2 keys are normalized, so a slash id must still be invalidated."""
    from app.services import model_metadata as mm
    mm.clear_caches()
    _seed_provider(models=[{"id": SLASH_ID, "name": SLASH_ID}])
    mm._l2.set("vcp", SLASH_ID, {"context_length": 64000})
    assert mm._l2.get("vcp", SLASH_ID) is not None

    assert rename_provider_model("vcp", SLASH_ID, "des/deepseek-v3")["ok"] is True
    assert mm._l2.get("vcp", SLASH_ID) is None


def test_composite_input_invalidates_the_bare_key():
    """Callers may pass "provider/model"; the cache key is the bare id."""
    from app.services import model_metadata as mm
    mm.clear_caches()
    _seed_provider(models=[{"id": "old-name", "name": "old-name"}])
    mm._l2.set("vcp", "old-name", {"context_length": 128000})

    res = rename_provider_model("vcp", "vcp/old-name", "vcp/new-name")
    assert res["ok"] is True
    assert mm._l2.get("vcp", "old-name") is None
