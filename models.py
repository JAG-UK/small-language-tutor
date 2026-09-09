"""Where conversations are kept.

The database is the record now: every turn is written through as it happens,
so a restart — or a tunnel dropping, or closing the laptop — loses nothing.
The in-memory dictionary in app.py is a cache in front of this, not the truth.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Column, DateTime, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def now():
    """Timezone-aware, unlike datetime.utcnow, which is also deprecated."""
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    language = Column(String(10))
    #: Kept, so reopening a flirty conversation does not resume it as friendly.
    tone = Column(String(20))
    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now, onupdate=now)
    #: These are JSON columns, so SQLAlchemy does the encoding. Passing them a
    #: string that has already been through json.dumps stores JSON inside JSON,
    #: which only ever round-tripped because the read did a matching json.loads.
    messages = Column(JSON, default=list)
    corrections = Column(JSON, default=list)
    hints = Column(JSON, default=list)


def database_path():
    """Beside this file unless told otherwise, never merely "wherever you
    happened to be standing" — a relative path silently gives you a different
    database when the app is started from a different directory."""
    return os.environ.get("SLT_DB") or str(Path(__file__).resolve().parent / "database.db")


def ensure_schema(engine):
    """Add any column the model has gained since this database was created.

    create_all() creates missing tables and stops there, so a column added to
    the model afterwards is simply absent and every query naming it fails with
    `no such column: conversations.hints` — which is what saving did, on every
    database that predated hints. Alembic is a great deal of machinery for one
    SQLite file; this is the part of it that was actually needed.
    """
    inspector = inspect(engine)
    if not inspector.has_table(Conversation.__tablename__):
        return []

    existing = {column["name"] for column in inspector.get_columns(Conversation.__tablename__)}
    added = []
    for column in Conversation.__table__.columns:
        if column.name in existing:
            continue
        declaration = column.type.compile(engine.dialect)
        with engine.begin() as connection:
            connection.execute(
                text(f"ALTER TABLE {Conversation.__tablename__} "
                     f"ADD COLUMN {column.name} {declaration}")
            )
        added.append(column.name)
    return added


engine = create_engine(f"sqlite:///{database_path()}", echo=False)
Base.metadata.create_all(engine)
ensure_schema(engine)
Session = sessionmaker(bind=engine)
