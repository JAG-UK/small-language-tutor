"""Reading and writing conversations, so app.py does not have to know SQL.

Every turn is written through, so what is on screen and what is on disk do not
drift. A conversation carries the id of its row once it has one; saving it again
updates that row rather than leaving a second copy behind.
"""

import json

from models import Conversation, Session, now

#: How much of the first thing the learner said becomes the title.
TITLE_LENGTH = 60


def new_conversation(language, tone):
    """The shape app.py keeps in memory. `id` is filled in by the first save."""
    return {
        'id': None,
        'messages': [],
        'language': language,
        'tone': tone,
        'corrections': [],
        'hints': [],
        # Turns already critiqued, so a repeated or reopened conversation does
        # not file the same learning point twice. Runtime only; not stored.
        'reviewed': set(),
    }


def title_for(conv):
    """The first thing the learner said, cut at a word rather than mid-word."""
    first = next((m['content'] for m in conv.get('messages', [])
                  if m.get('role') == 'user' and (m.get('content') or '').strip()), '')
    first = ' '.join(first.split())
    if not first:
        return "Untitled"
    if len(first) <= TITLE_LENGTH:
        return first
    return first[:TITLE_LENGTH].rsplit(' ', 1)[0] + '…'


def _decode(value):
    """Rows written before the double-encoding was fixed hold a JSON string
    inside the JSON column. Read either."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value or []


def save(conv):
    """Insert or update, returning the row id. Sets conv['id'] on the way."""
    session = Session()
    try:
        row = session.get(Conversation, conv['id']) if conv.get('id') else None
        if row is None:
            row = Conversation()
            session.add(row)

        row.title = title_for(conv)
        row.language = conv.get('language')
        row.tone = conv.get('tone')
        row.messages = conv.get('messages', [])
        row.corrections = conv.get('corrections', [])
        row.hints = conv.get('hints', [])
        row.updated_at = now()

        session.commit()
        conv['id'] = row.id
        return row.id
    finally:
        session.close()


def load(conv_id):
    """A stored conversation, in the shape app.py keeps in memory, or None.

    Everything already said is marked as reviewed: reopening a conversation
    should not re-critique its whole history, which would cost a model call per
    turn and file every learning point a second time.
    """
    session = Session()
    try:
        row = session.get(Conversation, conv_id)
        if row is None:
            return None
        messages = _decode(row.messages)
        return {
            'id': row.id,
            'messages': messages,
            'language': row.language or 'es',
            'tone': row.tone or 'friendly',
            'corrections': _decode(row.corrections),
            'hints': _decode(row.hints),
            'reviewed': {i for i, m in enumerate(messages) if m.get('role') == 'user'},
        }
    finally:
        session.close()


def recent(limit=50):
    """Summaries for the list, newest activity first."""
    session = Session()
    try:
        rows = (session.query(Conversation)
                .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                .limit(limit).all())
        return [{
            'id': row.id,
            'title': row.title or "Untitled",
            'language': row.language,
            'updated_at': row.updated_at or row.created_at,
            'message_count': len(_decode(row.messages)),
        } for row in rows]
    finally:
        session.close()


def delete(conv_id):
    """True if there was something to delete."""
    session = Session()
    try:
        row = session.get(Conversation, conv_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True
    finally:
        session.close()
