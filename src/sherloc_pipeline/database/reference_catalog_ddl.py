"""SQLite catalog guards shared by the migration and metadata creation path."""

CATALOG_TABLES = ("reference_releases", "reference_files", "reference_traces", "reference_templates")
RELEASE_METADATA_COLUMNS = (
    "release_id", "source_snapshot_sha256", "metadata_sha256", "array_index_sha256",
    "serving_policy_sha256", "schema_version", "source_name", "source_base_url",
    "captured_at", "importer_version", "parser_version", "qa_version",
)


def catalog_triggers():
    for table in CATALOG_TABLES:
        condition = ""
        if table == "reference_releases":
            unchanged = " AND ".join(f"NEW.{column} IS OLD.{column}" for column in RELEASE_METADATA_COLUMNS)
            condition = (
                " WHEN NOT (((OLD.status = 'staged' AND NEW.status = 'ready') OR "
                "(OLD.status = 'ready' AND NEW.status = 'retired')) AND " + unchanged + ")"
            )
        for operation in ("UPDATE", "DELETE"):
            name = f"{table}_immutable_{operation.lower()}"
            guard = condition if operation == "UPDATE" else ""
            yield name, (
                f"CREATE TRIGGER {name} BEFORE {operation} ON {table}{guard} "
                "BEGIN SELECT RAISE(ABORT, 'reference catalog rows are immutable'); END"
            )
