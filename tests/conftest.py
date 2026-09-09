"""Keep the suite from reading the shell it is run in.

Anyone hosting this has SLT_PASSWORD and SLT_MODEL exported, and app.py reads
both at import: the password would put a 401 in front of every endpoint test,
and the model name would change which prompts the tests see. Nineteen failures
that say nothing about the code.
"""

import os
import shutil
import tempfile

import pytest

#: Everything the app takes from the environment.
SETTINGS = ("SLT_MODEL", "SLT_CRITIC_MODEL", "SLT_PASSWORD", "SLT_HOST", "SLT_PORT", "SLT_DEBUG",
            "SLT_DB")

# At import, before pytest collects anything: app.py reads these at *its* import,
# which happens when a test module is collected, and that is earlier than any
# fixture runs.
_SAVED = {name: os.environ.pop(name, None) for name in SETTINGS}

# And a database of its own. Without this the suite opens the real one — the
# tests write conversations, so running them would put test data in your history.
_DATABASE = tempfile.mkdtemp(prefix="slt-tests-") + "/test.db"
os.environ["SLT_DB"] = _DATABASE


@pytest.fixture(autouse=True, scope="session")
def _restore_the_shell():
    """Put it all back afterwards, so a run leaves no trace on the process or disk."""
    yield
    for name, value in _SAVED.items():
        if value is not None:
            os.environ[name] = value
    shutil.rmtree(os.path.dirname(_DATABASE), ignore_errors=True)


@pytest.fixture(autouse=True)
def empty_database():
    """Each test starts with nothing saved, so none can see another's rows."""
    from models import Conversation, Session

    session = Session()
    session.query(Conversation).delete()
    session.commit()
    session.close()
