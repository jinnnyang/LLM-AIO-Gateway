"""
DB migration tests for the model capability metadata columns.

Design reference: docs/plans/模型能力元数据扩展-设计方案.md §3.1 / §6 用例 11.

Covers:
- New install: CREATE TABLE already carries the 7 new columns.
- Legacy install: PRAGMA-guarded ALTER TABLE adds the missing columns.
- Idempotency: running init_db repeatedly never fails and never duplicates.
- DEFAULT parity: migrated columns must declare the same DEFAULT as CREATE TABLE,
  otherwise new and upgraded databases diverge in behaviour.
- Price columns must be REAL and NULLable (never NOT NULL DEFAULT 0), so that
  "not collected" is distinguishable from "genuinely free".
- Existing rows survive the migration untouched (backward compatibility).
"""
import sqlite3

import pytest

import app.database as db_mod
from app.database import init_db, get_db, get_provider, add_provider


METADATA_COLUMNS = {
    "context_length",
    "max_output_tokens",
    "input_modalities",
    "output_modalities",
    "input_price",
    "output_price",
    "cached_input_price",
}

# (name, declared type, notnull, default) as reported by PRAGMA table_info.
EXPECTED_COLUMN_DDL = {
    "context_length": ("INTEGER", 0, None),
    "max_output_tokens": ("INTEGER", 0, None),
    "input_modalities": ("TEXT", 1, "'[]'"),
    "output_modalities": ("TEXT", 1, "'[]'"),
    "input_price": ("REAL", 0, None),
    "output_price": ("REAL", 0, None),
    "cached_input_price": ("REAL", 0, None),
}

# provider_models had 16 columns before this feature; 16 + 7 = 23.
EXPECTED_TOTAL_COLUMNS = 23


def _table_info(conn_or_db, table="provider_models"):
    rows = conn_or_db.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1]: {"type": row[2], "notnull": row[3], "default": row[4]} for row in rows}


def _fresh_db(path):
    """init_db on a brand new file, resetting the module-level init guard."""
    db_mod._initialized = False
    init_db(str(path))
    return path


@pytest.fixture(autouse=True)
def restore_db_state():
    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    yield
    db_mod.DB_PATH = previous_path
    db_mod._initialized = previous_initialized


# -- New installs --

def test_new_database_create_table_has_metadata_columns(tmp_path):
    """A brand new database must get the columns from CREATE TABLE, not from ALTER."""
    _fresh_db(tmp_path / "new.db")
    with get_db() as db:
        info = _table_info(db)
    missing = METADATA_COLUMNS - set(info)
    assert not missing, f"CREATE TABLE is missing metadata columns: {sorted(missing)}"


def test_new_database_total_column_count(tmp_path):
    """16 pre-existing columns + 7 new = 23. Guards accidental extra/renamed columns."""
    _fresh_db(tmp_path / "count.db")
    with get_db() as db:
        info = _table_info(db)
    assert len(info) == EXPECTED_TOTAL_COLUMNS, sorted(info)


def test_metadata_columns_have_expected_types_and_defaults(tmp_path):
    _fresh_db(tmp_path / "ddl.db")
    with get_db() as db:
        info = _table_info(db)
    for column, (expected_type, expected_notnull, expected_default) in EXPECTED_COLUMN_DDL.items():
        actual = info[column]
        assert actual["type"].upper() == expected_type, f"{column}: {actual}"
        assert actual["notnull"] == expected_notnull, f"{column}: {actual}"
        assert actual["default"] == expected_default, f"{column}: {actual}"


def test_price_columns_are_nullable_real(tmp_path):
    """Price columns must never be NOT NULL DEFAULT 0: 0 would be read as a real price."""
    _fresh_db(tmp_path / "price.db")
    with get_db() as db:
        info = _table_info(db)
    for column in ("input_price", "output_price", "cached_input_price"):
        assert info[column]["type"].upper() == "REAL"
        assert info[column]["notnull"] == 0, f"{column} must allow NULL"
        assert info[column]["default"] in (None, "NULL"), f"{column} must not default to 0"


def test_metadata_columns_default_to_null_and_empty_json(tmp_path):
    """Freshly inserted rows: numeric fields NULL, modalities '[]'."""
    _fresh_db(tmp_path / "defaults.db")
    add_provider({
        "id": "p-default",
        "name": "Default",
        "provider_type": "openai",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": [{"id": "m1", "name": "M1", "enabled": True}],
    })
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ? AND model_id = ?",
            ("p-default", "m1"),
        ).fetchone()
    assert row["context_length"] is None
    assert row["max_output_tokens"] is None
    assert row["input_price"] is None
    assert row["output_price"] is None
    assert row["cached_input_price"] is None
    assert row["input_modalities"] == "[]"
    assert row["output_modalities"] == "[]"


# -- Legacy installs --

LEGACY_SCHEMA = """
CREATE TABLE providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    provider_type TEXT NOT NULL DEFAULT 'openai',
    api_base TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT '',
    preprocessor TEXT NOT NULL DEFAULT '',
    image_generation TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'auto',
    responses_status TEXT NOT NULL DEFAULT 'unknown',
    responses_checked_at TEXT NOT NULL DEFAULT '',
    responses_expires_at TEXT NOT NULL DEFAULT '',
    responses_streaming INTEGER NOT NULL DEFAULT 0,
    responses_streaming_status TEXT NOT NULL DEFAULT 'unknown',
    responses_tool_types TEXT NOT NULL DEFAULT '[]',
    responses_error TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
    UNIQUE(provider_id, model_id)
);
"""


def _make_legacy_db(path):
    """Build a pre-feature database with the 16-column provider_models table."""
    conn = sqlite3.connect(str(path))
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO providers (id, name, provider_type, api_base, api_key, enabled) "
        "VALUES ('legacy', 'Legacy', 'openai', 'https://api.legacy/v1', 'sk-legacy', 1)"
    )
    conn.execute(
        "INSERT INTO provider_models (provider_id, model_id, model_name, enabled, preprocessor, source) "
        "VALUES ('legacy', 'legacy-model', 'Legacy Model', 1, 'vision-model', 'manual')"
    )
    conn.commit()
    conn.close()
    return path


def test_legacy_database_is_missing_metadata_columns_before_migration(tmp_path):
    """Sanity check on the fixture itself, so the upgrade test proves something."""
    path = _make_legacy_db(tmp_path / "legacy_pre.db")
    conn = sqlite3.connect(str(path))
    try:
        info = _table_info(conn)
    finally:
        conn.close()
    assert not (METADATA_COLUMNS & set(info))
    assert len(info) == EXPECTED_TOTAL_COLUMNS - len(METADATA_COLUMNS)


def test_legacy_database_upgrade_adds_all_metadata_columns(tmp_path):
    path = _make_legacy_db(tmp_path / "legacy.db")
    _fresh_db(path)
    with get_db() as db:
        info = _table_info(db)
    missing = METADATA_COLUMNS - set(info)
    assert not missing, f"migration did not add: {sorted(missing)}"


def test_legacy_upgrade_matches_fresh_install_ddl(tmp_path):
    """ALTER TABLE DEFAULT must be byte-identical to CREATE TABLE DEFAULT.

    Design §3.1: '迁移 DEFAULT 必须与 CREATE TABLE 完全一致，否则新旧库行为分叉'.
    """
    legacy_path = _make_legacy_db(tmp_path / "legacy_ddl.db")
    _fresh_db(legacy_path)
    with get_db() as db:
        migrated = _table_info(db)

    _fresh_db(tmp_path / "fresh_ddl.db")
    with get_db() as db:
        fresh = _table_info(db)

    for column in METADATA_COLUMNS:
        assert migrated[column] == fresh[column], (
            f"{column} diverges: migrated={migrated[column]} fresh={fresh[column]}"
        )


def test_legacy_upgrade_preserves_existing_rows(tmp_path):
    """Backward compatibility: pre-existing models keep their data and get NULL metadata."""
    path = _make_legacy_db(tmp_path / "legacy_rows.db")
    _fresh_db(path)

    provider = get_provider("legacy")
    assert provider is not None
    models = {m["id"]: m for m in provider["models"]}
    assert set(models) == {"legacy-model"}
    assert models["legacy-model"]["name"] == "Legacy Model"
    assert models["legacy-model"]["enabled"] is True
    assert models["legacy-model"]["preprocessor"] == "vision-model"
    # legacy 'manual' rows are upgraded to 'custom' by the terminology migration
    assert models["legacy-model"]["source"] == "custom"


def test_legacy_upgrade_backfills_metadata_defaults(tmp_path):
    """Rows that predate the feature must read as NULL / [] rather than crash."""
    path = _make_legacy_db(tmp_path / "legacy_backfill.db")
    _fresh_db(path)
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM provider_models WHERE provider_id = 'legacy' AND model_id = 'legacy-model'"
        ).fetchone()
    assert row["context_length"] is None
    assert row["max_output_tokens"] is None
    assert row["input_price"] is None
    assert row["output_price"] is None
    assert row["cached_input_price"] is None
    assert row["input_modalities"] == "[]"
    assert row["output_modalities"] == "[]"


# -- Idempotency --

def test_migration_is_idempotent_across_repeated_init(tmp_path):
    """init_db runs on every process start: re-running must not raise or duplicate."""
    path = _make_legacy_db(tmp_path / "idempotent.db")
    for _ in range(3):
        _fresh_db(path)
    with get_db() as db:
        info = _table_info(db)
        names = [row[1] for row in db.execute("PRAGMA table_info(provider_models)").fetchall()]
    assert len(names) == len(set(names)), "duplicate columns after repeated migration"
    assert len(info) == EXPECTED_TOTAL_COLUMNS


def test_migration_is_idempotent_on_new_database(tmp_path):
    path = tmp_path / "idempotent_new.db"
    _fresh_db(path)
    _fresh_db(path)
    with get_db() as db:
        info = _table_info(db)
    assert METADATA_COLUMNS <= set(info)
    assert len(info) == EXPECTED_TOTAL_COLUMNS


def test_partially_migrated_database_gets_remaining_columns(tmp_path):
    """A database where a previous run only added some columns must be completed.

    Guards the PRAGMA table_info(provider_models) per-column check: the migration
    must not bail out just because one column already exists.
    """
    path = _make_legacy_db(tmp_path / "partial.db")
    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE provider_models ADD COLUMN context_length INTEGER")
    conn.execute("ALTER TABLE provider_models ADD COLUMN input_modalities TEXT NOT NULL DEFAULT '[]'")
    conn.commit()
    conn.close()

    _fresh_db(path)
    with get_db() as db:
        info = _table_info(db)
    missing = METADATA_COLUMNS - set(info)
    assert not missing, f"partially migrated DB left columns missing: {sorted(missing)}"
    assert len(info) == EXPECTED_TOTAL_COLUMNS


# -- Constraints untouched --

def test_existing_constraints_survive_migration(tmp_path):
    """UNIQUE(provider_id, model_id), FK CASCADE and the model_id index must remain."""
    path = _make_legacy_db(tmp_path / "constraints.db")
    _fresh_db(path)

    with get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, source) "
            "VALUES ('legacy', 'dup-model', 'Dup', 'auto')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        with get_db() as db:
            db.execute(
                "INSERT INTO provider_models (provider_id, model_id, model_name, source) "
                "VALUES ('legacy', 'dup-model', 'Dup Again', 'auto')"
            )

    with get_db() as db:
        indexes = {row[1] for row in db.execute("PRAGMA index_list(provider_models)").fetchall()}
    assert "idx_provider_models_model_id" in indexes


def test_foreign_key_cascade_still_deletes_models(tmp_path):
    path = _make_legacy_db(tmp_path / "cascade.db")
    _fresh_db(path)
    with get_db() as db:
        db.execute("DELETE FROM providers WHERE id = 'legacy'")
    with get_db() as db:
        remaining = db.execute(
            "SELECT COUNT(*) AS n FROM provider_models WHERE provider_id = 'legacy'"
        ).fetchone()["n"]
    assert remaining == 0
