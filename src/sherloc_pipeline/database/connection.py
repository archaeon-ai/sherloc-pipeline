"""
Database connection utilities for PHASE.

This module provides functions for creating database connections,
sessions, and managing the database lifecycle.

Example:
    >>> from sherloc_pipeline.database.connection import get_engine, get_session
    >>>
    >>> engine = get_engine("./phase.db")
    >>> with get_session(engine) as session:
    ...     # Use session for queries
    ...     pass
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

def _default_db_url() -> str:
    """Resolve the default SQLite URL from $SHERLOC_DB or fall back to ./phase.db."""
    db = os.getenv("SHERLOC_DB", "./phase.db")
    if db == ":memory:":
        return "sqlite:///:memory:"
    return f"sqlite:///{db}" if not db.startswith("/") else f"sqlite:////{db.lstrip('/')}"


DATABASE_URL = _default_db_url()


def get_engine(
    database_path: Optional[Union[str, Path]] = None,
    echo: bool = False,
) -> Engine:
    """Create a SQLAlchemy engine for the PHASE database.

    Args:
        database_path: Path to SQLite database file. If None, uses the
            default path ./phase.db. Can also be ":memory:"
            for an in-memory database (useful for testing).
        echo: If True, log all SQL statements (for debugging).

    Returns:
        SQLAlchemy Engine instance.

    Example:
        >>> engine = get_engine("./phase.db")
        >>> engine = get_engine(":memory:")  # In-memory for testing
    """
    if database_path is None:
        url = DATABASE_URL
    elif str(database_path) == ":memory:":
        url = "sqlite:///:memory:"
    else:
        # Ensure path is absolute
        db_path = Path(database_path).absolute()
        url = f"sqlite:///{db_path}"

    engine = create_engine(
        url,
        echo=echo,
        # Enable foreign key enforcement for SQLite
        connect_args={"check_same_thread": False},
    )

    # Enable foreign key constraints and WAL mode for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def get_session_factory(engine: Engine) -> sessionmaker:
    """Create a session factory bound to an engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        Session factory (sessionmaker).
    """
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    """Create a database session with automatic cleanup.

    This context manager creates a new session, yields it for use,
    and ensures proper cleanup (commit on success, rollback on error).

    Args:
        engine: SQLAlchemy Engine instance.

    Yields:
        SQLAlchemy Session instance.

    Example:
        >>> with get_session(engine) as session:
        ...     sol = SolORM(sol_number=921)
        ...     session.add(sol)
        ...     # Commits automatically on exit
    """
    SessionLocal = get_session_factory(engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables(engine: Engine) -> None:
    """Create all database tables.

    Creates all tables defined in the ORM models. This is idempotent;
    existing tables are not modified.

    Args:
        engine: SQLAlchemy Engine instance.

    Example:
        >>> engine = get_engine("./phase.db")
        >>> create_all_tables(engine)
    """
    from sherloc_pipeline.database.models import Base
    Base.metadata.create_all(engine)


class SchemaNotCurrentError(RuntimeError):
    """An existing database is not at the code's Alembic head.

    Paths that read and write data never create or upgrade schema on an
    existing database: migrations are applied by an operator. The message
    names the revision pair and the command.
    """

    def __init__(self, database_path, current, heads):
        self.database_path = str(database_path)
        self.current = tuple(current)
        self.heads = tuple(heads)
        found = ", ".join(self.current) or "unstamped (tables but no alembic_version)"
        expected = ", ".join(self.heads)
        target = self.heads[0] if len(self.heads) == 1 else "heads"
        super().__init__(
            f"Database {self.database_path} schema is at {found}; this code expects {expected}. "
            f"Nothing was changed. Schema changes are applied by an operator, from the "
            f"sherloc-pipeline checkout: PHASE_DATABASE_PATH={self.database_path} "
            f"alembic upgrade {target}"
        )


def _alembic_config():
    from alembic.config import Config

    package_root = Path(__file__).parent.parent.parent.parent  # src/../..
    ini_path = package_root / "alembic.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"alembic.ini not found at {ini_path}")
    return Config(str(ini_path))


_HEADS_CACHE: dict = {}


def alembic_heads() -> tuple:
    """The Alembic head revision(s) of this code, sorted."""
    if "heads" not in _HEADS_CACHE:
        from alembic.script import ScriptDirectory

        _HEADS_CACHE["heads"] = tuple(sorted(ScriptDirectory.from_config(_alembic_config()).get_heads()))
    return _HEADS_CACHE["heads"]


def schema_revisions(engine: Engine) -> tuple:
    """The revision(s) stamped in the database's alembic_version, sorted (empty if none)."""
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as conn:
        return tuple(sorted(MigrationContext.configure(conn).get_current_heads()))


def _data_tables(engine: Engine) -> list:
    from sqlalchemy import inspect

    return [
        name for name in inspect(engine).get_table_names()
        if name != "alembic_version" and not name.startswith("sqlite_")
    ]


def _create_and_stamp(engine: Engine) -> None:
    """Create the ORM schema on an empty database and stamp it at the code's head."""
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_config())
    create_all_tables(engine)
    with engine.begin() as conn:
        MigrationContext.configure(conn).stamp(script, "heads")


def require_current_schema(engine: Engine, database_path) -> None:
    """Raise SchemaNotCurrentError unless the database is stamped at the code's head(s)."""
    current, heads = schema_revisions(engine), alembic_heads()
    if set(current) != set(heads):
        raise SchemaNotCurrentError(database_path, current, heads)


def open_database(database_path: Optional[Union[str, Path]] = None) -> Engine:
    """Open a database for a command that reads and writes data, without changing its schema.

    - ``:memory:``: the ORM schema is created (nothing to conflict with).
    - A missing file, or one with no tables: the ORM schema is created and
      stamped at the code's Alembic head, so a later ``alembic upgrade``
      starts from the right revision.
    - Any other database must already be at the code's head; otherwise
      ``SchemaNotCurrentError`` is raised and nothing is written.
    """
    if database_path is None:
        database_path = "./phase.db"
    if str(database_path) == ":memory:":
        engine = get_engine(database_path)
        create_all_tables(engine)
        return engine

    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(database_path)
    try:
        if _data_tables(engine):
            require_current_schema(engine, database_path)
        else:
            _create_and_stamp(engine)
    except Exception:
        engine.dispose()
        raise
    return engine


def open_database_readonly(database_path: Union[str, Path]) -> Engine:
    """Open an existing SQLite database read-only (no file creation, no schema change)."""
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    return create_engine(
        f"sqlite:///file:{path.resolve()}?mode=ro&uri=true",
        connect_args={"check_same_thread": False},
    )


def ensure_database_exists(database_path: Optional[Union[str, Path]] = None) -> Engine:
    """Ensure the database exists and has all tables.

    This is a convenience function that creates the database file
    (if needed) and all tables.

    Args:
        database_path: Path to SQLite database file.

    Returns:
        SQLAlchemy Engine instance for the database.

    Example:
        >>> engine = ensure_database_exists("./phase.db")
    """
    if database_path is None:
        database_path = "./phase.db"

    # Ensure parent directory exists
    if str(database_path) != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine(database_path)
    create_all_tables(engine)
    return engine


def init_database(
    db_path: Optional[Union[str, Path]] = None,
) -> Engine:
    """Initialize a SHERLOC database via Alembic migrations.

    For new databases, runs the full Alembic migration chain from scratch.
    For existing databases, applies any pending migrations to bring the
    schema to HEAD. Idempotent.

    Used by ``sherloc init`` for the primary ``phase.db``. The PDS-specific
    variant (``init_pds_database``) layers an additional unique constraint
    on top of this for ``phase_pds.db``.

    Args:
        db_path: Path to the database file. Defaults to ``./phase.db``.
            ``:memory:`` falls back to ``create_all_tables`` since
            Alembic needs a file URL.

    Returns:
        SQLAlchemy Engine instance for the initialized database.
    """
    if db_path is None:
        db_path = "./phase.db"

    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine(db_path)

    if str(db_path) != ":memory:":
        # `sherloc init` is the explicit operator command: it migrates an
        # existing database to head. A failed upgrade is an error; falling
        # back to create_all would add tables the alembic stamp does not
        # record (#135).
        _run_alembic_upgrade(str(db_path))
    else:
        create_all_tables(engine)

    logger.info("Initialized database at %s", db_path)
    return engine


def _default_pds_db_path() -> str:
    """Resolve the default PDS SQLite path.

    Reads $SHERLOC_PDS_DB first (canonical, mirrors $SHERLOC_DB for the
    main database), then $SHERLOC_PDS_DB_PATH (legacy — already used by
    config.resolve_paths() for the YAML database.pds_path field), and
    finally falls back to ./phase_pds.db. Returns a filesystem path,
    not a SQLAlchemy URL, because init_pds_database() needs the raw
    path for Alembic and Path() operations.
    """
    return (
        os.getenv("SHERLOC_PDS_DB")
        or os.getenv("SHERLOC_PDS_DB_PATH")
        or "./phase_pds.db"
    )


# Default PDS database path
PDS_DATABASE_PATH = _default_pds_db_path()


def init_pds_database(
    db_path: Optional[Union[str, Path]] = None,
) -> Engine:
    """Open phase_pds.db for PDS ingestion plus PDS-specific constraints.

    A new (missing or empty) database gets the ORM schema stamped at the
    code's Alembic head. An existing database must already be at head, or
    ``SchemaNotCurrentError`` is raised: this path never migrates. Then adds a PDS-specific unique constraint on ``scans.scan_id``
    to enforce idempotency via PDS LID (spec s9, s12).

    This function is idempotent on a current database: the unique index
    uses ``IF NOT EXISTS``.

    Args:
        db_path: Path to the PDS database file. Defaults to
            ``./phase_pds.db``. Can be ``:memory:`` for testing.

    Returns:
        SQLAlchemy Engine instance for the initialized database.
    """
    if db_path is None:
        db_path = PDS_DATABASE_PATH

    # Ensure parent directory exists
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # A new database gets the stamped ORM schema; an existing one must
    # already be at head. PDS ingestion never migrates (#135).
    engine = open_database(db_path)

    # Add PDS-specific unique constraint on scans.scan_id (spec s12).
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_pds_scans_scan_id "
            "ON scans (scan_id)"
        ))
        conn.commit()

    logger.info("Initialized PDS database at %s", db_path)
    return engine


def _run_alembic_upgrade(db_path: str) -> None:
    """Run alembic upgrade head against the given database path."""
    from alembic import command

    alembic_cfg = _alembic_config()

    # env.py reads PHASE_DATABASE_PATH env var — set it to target the right DB
    old_env = os.environ.get("PHASE_DATABASE_PATH")
    os.environ["PHASE_DATABASE_PATH"] = db_path
    try:
        command.upgrade(alembic_cfg, "head")
    finally:
        if old_env is not None:
            os.environ["PHASE_DATABASE_PATH"] = old_env
        else:
            os.environ.pop("PHASE_DATABASE_PATH", None)
