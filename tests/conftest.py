"""Keep the suite from reading the shell it is run in.

Anyone hosting this has SLT_PASSWORD and SLT_MODEL exported, and app.py reads
both at import: the password would put a 401 in front of every endpoint test,
and the model name would change which prompts the tests see. Nineteen failures
that say nothing about the code.
"""

import os

import pytest

#: Everything the app takes from the environment.
SETTINGS = ("SLT_MODEL", "SLT_CRITIC_MODEL", "SLT_PASSWORD", "SLT_HOST", "SLT_PORT", "SLT_DEBUG")

# At import, before pytest collects anything: app.py reads these at *its* import,
# which happens when a test module is collected, and that is earlier than any
# fixture runs.
_SAVED = {name: os.environ.pop(name, None) for name in SETTINGS}


@pytest.fixture(autouse=True, scope="session")
def _restore_the_shell():
    """Put them back afterwards, so a test run leaves no trace on the process."""
    yield
    for name, value in _SAVED.items():
        if value is not None:
            os.environ[name] = value
