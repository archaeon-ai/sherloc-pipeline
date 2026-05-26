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
        try:
            _run_alembic_upgrade(str(db_path))
        except Exception as exc:
            logger.warning(
                "Alembic upgrade failed for %s, falling back to create_all: %s",
                db_path, exc,
            )
            create_all_tables(engine)
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
    """Initialize phase_pds.db via Alembic migrations plus PDS-specific constraints.

    For new databases, runs the full Alembic migration chain from scratch.
    For existing databases, runs any pending migrations to bring the schema
    to HEAD. Then adds a PDS-specific unique constraint on ``scans.scan_id``
    to enforce idempotency via PDS LID (spec s9, s12).

    This function is idempotent: calling it multiple times on an existing
    database is safe — Alembic skips already-applied migrations, and the
    unique index uses ``IF NOT EXISTS``.

    Args:
        db_path: Path to the PDS database file. Defaults to
            ``./phase_pds.db``. Can be ``:memory:`` for testing
            (falls back to create_all_tables since Alembic needs a file).

    Returns:
        SQLAlchemy Engine instance for the initialized database.
    """
    if db_path is None:
        db_path = PDS_DATABASE_PATH

    # Ensure parent directory exists
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine(db_path)

    # Use Alembic for schema management (file-based databases)
    if str(db_path) != ":memory:":
        try:
            _run_alembic_upgrade(str(db_path))
        except Exception as exc:
            logger.warning(
                "Alembic upgrade failed for %s, falling back to create_all: %s",
                db_path, exc,
            )
            create_all_tables(engine)
    else:
        # In-memory DBs can't use Alembic (no persistent URL)
        create_all_tables(engine)

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
    from alembic.config import Config
    from alembic import command

    # Find alembic.ini relative to this package
    package_root = Path(__file__).parent.parent.parent.parent  # src/../..
    ini_path = package_root / "alembic.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"alembic.ini not found at {ini_path}")

    # env.py reads PHASE_DATABASE_PATH env var — set it to target the right DB
    old_env = os.environ.get("PHASE_DATABASE_PATH")
    os.environ["PHASE_DATABASE_PATH"] = db_path
    try:
        alembic_cfg = Config(str(ini_path))
        command.upgrade(alembic_cfg, "head")
    finally:
        if old_env is not None:
            os.environ["PHASE_DATABASE_PATH"] = old_env
        else:
            os.environ.pop("PHASE_DATABASE_PATH", None)
