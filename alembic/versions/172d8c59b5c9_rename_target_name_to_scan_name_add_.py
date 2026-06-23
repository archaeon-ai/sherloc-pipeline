"""rename_target_name_to_scan_name_add_target

Revision ID: 172d8c59b5c9
Revises: 0385ab87eb83
Create Date: 2026-01-24 17:06:28.904408

Schema changes:
- Rename scans.target_name -> scans.scan_name (the scan sequence name like 'detail_1')
- Add scans.target (nullable) for the geological target name (like 'Amherst_Point')
- Update index from ix_scans_sol_target to ix_scans_sol_scan_name
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '172d8c59b5c9'
down_revision: Union[str, Sequence[str], None] = '0385ab87eb83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: rename target_name to scan_name, add target column.

    For SQLite we use raw SQL to be more explicit about what we're doing,
    since batch_alter_table can have unexpected index behavior.
    """
    # Step 1: Add the new target column
    op.execute("ALTER TABLE scans ADD COLUMN target VARCHAR(200)")

    # Step 2: Drop old indexes before rename (they reference target_name)
    op.execute("DROP INDEX IF EXISTS ix_scans_target_name")
    op.execute("DROP INDEX IF EXISTS ix_scans_sol_target")

    # Step 3: Rename target_name to scan_name
    # SQLite 3.25.0+ supports ALTER TABLE RENAME COLUMN
    op.execute("ALTER TABLE scans RENAME COLUMN target_name TO scan_name")

    # Step 4: Create new indexes with correct column names
    op.execute("CREATE INDEX ix_scans_scan_name ON scans (scan_name)")
    op.execute("CREATE INDEX ix_scans_sol_scan_name ON scans (sol_number, scan_name)")

    # Step 5: Create index on new target column
    op.execute("CREATE INDEX ix_scans_target ON scans (target)")


def downgrade() -> None:
    """Downgrade schema: rename scan_name back to target_name, drop target column."""
    # Drop new indexes
    op.execute("DROP INDEX IF EXISTS ix_scans_target")
    op.execute("DROP INDEX IF EXISTS ix_scans_sol_scan_name")
    op.execute("DROP INDEX IF EXISTS ix_scans_scan_name")

    # Rename scan_name back to target_name
    op.execute("ALTER TABLE scans RENAME COLUMN scan_name TO target_name")

    # SQLite doesn't support DROP COLUMN directly before 3.35.0
    # We need to recreate the table without the column
    # For simplicity, we'll create a new table, copy data, drop old, rename
    op.execute("""
        CREATE TABLE scans_new (
            id VARCHAR(36) PRIMARY KEY,
            sol_number INTEGER NOT NULL,
            target_name VARCHAR(200) NOT NULL,
            scan_id VARCHAR(200) NOT NULL,
            sclk_start INTEGER NOT NULL,
            sclk_stop INTEGER,
            n_points INTEGER NOT NULL,
            n_channels INTEGER NOT NULL,
            shots_per_point INTEGER NOT NULL,
            laser_wavelength_nm FLOAT NOT NULL,
            processing_applied VARCHAR(100),
            source_path TEXT,
            loupe_metadata JSON,
            pds4_metadata JSON,
            created_at DATETIME NOT NULL,
            updated_at DATETIME,
            FOREIGN KEY (sol_number) REFERENCES sols(sol_number) ON DELETE CASCADE
        )
    """)
    op.execute("""
        INSERT INTO scans_new (id, sol_number, target_name, scan_id, sclk_start, sclk_stop,
                              n_points, n_channels, shots_per_point, laser_wavelength_nm,
                              processing_applied, source_path, loupe_metadata, pds4_metadata,
                              created_at, updated_at)
        SELECT id, sol_number, target_name, scan_id, sclk_start, sclk_stop,
               n_points, n_channels, shots_per_point, laser_wavelength_nm,
               processing_applied, source_path, loupe_metadata, pds4_metadata,
               created_at, updated_at
        FROM scans
    """)
    op.execute("DROP TABLE scans")
    op.execute("ALTER TABLE scans_new RENAME TO scans")

    # Recreate indexes (must match what initial migration expects)
    op.execute("CREATE INDEX ix_scans_sol_number ON scans (sol_number)")
    op.execute("CREATE INDEX ix_scans_target_name ON scans (target_name)")
    op.execute("CREATE INDEX ix_scans_scan_id ON scans (scan_id)")
    op.execute("CREATE INDEX ix_scans_sclk ON scans (sclk_start)")
    op.execute("CREATE INDEX ix_scans_sol_target ON scans (sol_number, target_name)")
