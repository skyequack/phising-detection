# ADR 001: SQLite through SQLAlchemy for storage

Status: accepted (2026-10-11)

## Context
The system stores prediction history, per-prediction feature snapshots, a registry of model versions and an audit log. It
is a single-instance academic project run on one machine, with no multi-user write load and no operations team. The
project report allows PostgreSQL or SQLite.

## Decision
Use SQLite, accessed only through the SQLAlchemy ORM. The file path comes from `DATABASE_URL`. The schema has four tables
([database.md](../design/database.md)).

## Consequences
- No server to install or configure; the database is one file and tests use in-memory databases.
- Through the ORM, every value is a bound parameter, which is what the SQL-injection tests rely on.
- One writer at a time. That is enough here (prediction writes are small and run after the response), but it would be the
  first thing to fail under real traffic.
- Schema changes need manual migration (`create_all` does not alter tables), and nothing prunes old rows.
- Moving to PostgreSQL should only mean changing `DATABASE_URL`, but it has not been tried. JSON columns, timestamps and
  the in-memory test setup are the places a difference would show.

## Alternatives considered
PostgreSQL: better for concurrent writes and managed backups, but it adds an install step for no benefit at this scale.
Flat files (CSV or JSON lines): simpler still, but no constraints, no paging queries, and no safe concurrent appends.
