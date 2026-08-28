"""Regression tests for slash-bearing upstream model ids in lookup helpers.

A provider may serve nested upstream paths such as "des/deepseek", stored bare
in provider_models.model_id. Its full gateway address is "vcp/des/deepseek".

parse_model_id() cannot disambiguate those: given the bare "des/deepseek" it
reads "des" as the provider and truncates the model to "deepseek". The lookup
helpers below therefore use _strip_own_prefix(provider_id, model), which only
removes this provider's own prefix and only once.

Before the fix these helpers survived by accident: their SQL is
"model_id IN (?, ?) OR model_name = ?" with the unprocessed value as the first
parameter, so the OR masked the truncated one. Any narrowing of that SQL would
have turned the latent bug into silent data loss, so these tests pin the
behaviour of each parameter independently.
"""
import pytest

import app.database as db_mod
from app.database import (
    add_provider,
    get_db,
    get_model_image_generation,
    get_model_responses_capability,
    init_db,
    set_model_responses_capability,
    update_model_responses_capability,
    update_model_responses_tool_types,
)

PROVIDER_ID = "vcp"
BARE = "des/deepseek"
COMPOSITE = f"{PROVIDER_ID}/{BARE}"
# What parse_model_id() would have produced: a different, non-existent row.
TRUNCATED = "deepseek"


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    db_mod._initialized = False
    init_db(str(tmp_path / "slash.db"))
    try:
        yield
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized


def _seed(model_id: str = BARE) -> None:
    add_provider({
        "id": PROVIDER_ID,
        "name": "VCP",
        "api_base": "https://example.invalid/v1",
        "api_key": "k",
        "models": [{
            "id": model_id,
            "name": model_id,
            "enabled": True,
            "source": "custom",
        }],
    })


def _set_image_generation(model_id: str, value: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET image_generation = ? WHERE provider_id = ? AND model_id = ?",
            (value, PROVIDER_ID, model_id),
        )


def _row(model_id: str = BARE) -> dict:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ? AND model_id = ?",
            (PROVIDER_ID, model_id),
        ).fetchone()
    assert row is not None, f"expected a row for {model_id!r}"
    return dict(row)


# -- the seam itself -------------------------------------------------------


def test_strip_own_prefix_keeps_upstream_slashes():
    assert db_mod._strip_own_prefix(PROVIDER_ID, COMPOSITE) == BARE
    assert db_mod._strip_own_prefix(PROVIDER_ID, BARE) == BARE


def test_parse_model_id_would_truncate_a_bare_slash_id():
    """Pins WHY _strip_own_prefix exists, so the fix is not "simplified" back."""
    assert db_mod.parse_model_id(BARE).model_name == TRUNCATED


# -- get_model_image_generation -------------------------------------------


@pytest.mark.parametrize("lookup", [BARE, COMPOSITE])
def test_image_generation_reads_slash_model(lookup):
    _seed()
    _set_image_generation(BARE, "1")
    assert get_model_image_generation(PROVIDER_ID, lookup) is True


def test_image_generation_does_not_match_the_truncated_id():
    """A real "deepseek" row must not answer for "des/deepseek"."""
    _seed(model_id=TRUNCATED)
    _set_image_generation(TRUNCATED, "1")
    assert get_model_image_generation(PROVIDER_ID, BARE) is False


# -- get_model_responses_capability ---------------------------------------


@pytest.mark.parametrize("lookup", [BARE, COMPOSITE])
def test_responses_capability_reads_slash_model(lookup):
    _seed()
    set_model_responses_capability(PROVIDER_ID, BARE, status="supported")
    found = get_model_responses_capability(PROVIDER_ID, lookup)
    assert found is not None
    assert found["id"] == BARE


def test_responses_capability_does_not_match_the_truncated_id():
    _seed(model_id=TRUNCATED)
    found = get_model_responses_capability(PROVIDER_ID, BARE)
    assert found is None


# -- writers ---------------------------------------------------------------


@pytest.mark.parametrize("target", [BARE, COMPOSITE])
def test_set_responses_capability_writes_slash_model(target):
    _seed()
    set_model_responses_capability(
        PROVIDER_ID, target, status="supported", streaming=True,
    )
    row = _row()
    assert row["responses_status"] == "supported"
    assert row["responses_streaming"] == 1


def test_set_responses_capability_does_not_write_the_truncated_row():
    """Both a slash row and a truncated row exist; only the slash row changes."""
    _seed()
    with get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled, source) VALUES (?, ?, ?, 1, 'custom')",
            (PROVIDER_ID, TRUNCATED, TRUNCATED),
        )
    set_model_responses_capability(PROVIDER_ID, BARE, status="supported")
    assert _row(BARE)["responses_status"] == "supported"
    assert _row(TRUNCATED)["responses_status"] == "unknown"


@pytest.mark.parametrize("target", [BARE, COMPOSITE])
def test_update_responses_capability_writes_slash_model(target):
    _seed()
    update_model_responses_capability(PROVIDER_ID, target, status="degraded")
    assert _row()["responses_status"] == "degraded"


@pytest.mark.parametrize("target", [BARE, COMPOSITE])
def test_update_tool_types_writes_slash_model(target):
    _seed()
    update_model_responses_tool_types(PROVIDER_ID, target, ["web_search"])
    assert _row()["responses_tool_types"] == '["web_search"]'


def test_update_tool_types_does_not_write_the_truncated_row():
    _seed()
    with get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled, source) VALUES (?, ?, ?, 1, 'custom')",
            (PROVIDER_ID, TRUNCATED, TRUNCATED),
        )
    update_model_responses_tool_types(PROVIDER_ID, BARE, ["web_search"])
    assert _row(BARE)["responses_tool_types"] == '["web_search"]'
    assert _row(TRUNCATED)["responses_tool_types"] == "[]"


# -- flat ids keep working ------------------------------------------------


def test_flat_model_id_still_resolves_both_forms():
    _seed(model_id="gpt-4o")
    _set_image_generation("gpt-4o", "1")
    assert get_model_image_generation(PROVIDER_ID, "gpt-4o") is True
    assert get_model_image_generation(PROVIDER_ID, f"{PROVIDER_ID}/gpt-4o") is True
    set_model_responses_capability(PROVIDER_ID, f"{PROVIDER_ID}/gpt-4o", status="supported")
    assert _row("gpt-4o")["responses_status"] == "supported"
