"""Making the tutor safe to reach from somewhere other than the machine it runs on.

The reason this needs any thought: the app answers with a GPU and keeps a
database of conversations. An open port is someone else's free compute and your
transcripts, and the Flask debugger — which used to be on, on every interface —
executes whatever is sent to it.

So the defaults are closed. It binds to loopback, where nothing outside the
machine can reach it, and the debugger is off unless asked for and refuses to
come on anywhere it could be reached.

To use it from elsewhere, forward the port over SSH rather than opening one.
That is not belt-and-braces: HTTP Basic sends the password in a header that is
merely encoded, not encrypted, so on a bare HTTP port it is readable by anything
on the path. The tunnel supplies the encryption the password needs. The password
is then what stops other people *on the far machine* using the tunnel's exit.

    SLT_HOST      where to bind. Defaults to 127.0.0.1.
    SLT_PORT      defaults to 5001.
    SLT_PASSWORD  required before anything is served. Mandatory off loopback.
    SLT_DEBUG     the Werkzeug debugger. Loopback only, and off by default.
"""

import hmac
import os
import time
from collections import defaultdict

from flask import Response, request

#: Addresses that only this machine can reach.
LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}

REALM = "Language Tutor"

#: Wrong passwords tolerated from one address before it is made to wait. Slow
#: enough to make guessing pointless, loose enough to survive a few typos.
MAX_FAILURES = 10
LOCKOUT_SECONDS = 300


def bind_host():
    return os.environ.get("SLT_HOST", "127.0.0.1").strip()


def bind_port():
    return int(os.environ.get("SLT_PORT", "5001"))


def password():
    return os.environ.get("SLT_PASSWORD") or None


def debug_enabled():
    return os.environ.get("SLT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def is_loopback(host):
    return (host or "").strip().lower() in LOOPBACK


def startup_problems(host, secret, debug):
    """(refusals, warnings) for a given configuration.

    A refusal is a way of being reachable that this app should not offer at all;
    a warning is a real risk the operator may have accepted knowingly.
    """
    refusals, warnings = [], []

    if not is_loopback(host):
        if not secret:
            refusals.append(
                f"SLT_HOST={host!r} serves the app to the network, but SLT_PASSWORD is not set, "
                f"so anyone who can reach the port can use your GPU and read your conversations. "
                f"Set a password, or leave SLT_HOST unset and reach it over an SSH tunnel "
                f"(see the README)."
            )
        if debug:
            refusals.append(
                f"SLT_DEBUG with SLT_HOST={host!r} puts the Werkzeug debugger on the network. "
                f"It runs whatever it is sent, so this is a remote shell. Debug on loopback only."
            )
        if secret:
            warnings.append(
                "HTTP Basic sends the password encoded rather than encrypted, so on a plain "
                "HTTP port it can be read in transit. Prefer an SSH tunnel, or put a reverse "
                "proxy with TLS in front."
            )
    # No warning for loopback without a password: that is the ordinary way to
    # run this on your own laptop, and a warning printed on every normal start
    # is one people learn to scroll past. The README makes the case for setting
    # one where it matters, which is when the far end of a tunnel is shared.
    return refusals, warnings


class Gate:
    """Asks for the password, and makes repeated guessing not worth the trouble."""

    def __init__(self, secret, max_failures=MAX_FAILURES, lockout=LOCKOUT_SECONDS, clock=time.time):
        self.secret = secret
        self.max_failures = max_failures
        self.lockout = lockout
        self.clock = clock
        self._failures = defaultdict(list)

    def _recent_failures(self, who):
        now = self.clock()
        recent = [t for t in self._failures[who] if now - t < self.lockout]
        self._failures[who] = recent
        return recent

    def locked_out(self, who):
        return len(self._recent_failures(who)) >= self.max_failures

    def accepts(self, supplied):
        # Constant-time, so the answer does not leak the password one character
        # at a time to anyone timing the response.
        return hmac.compare_digest(supplied or "", self.secret or "")

    def check(self, who, supplied):
        """None if the request may proceed, otherwise the response to send."""
        if self.locked_out(who):
            return Response(
                "Too many attempts. Try again later.\n",
                429,
                {"Retry-After": str(self.lockout)},
            )
        if self.accepts(supplied):
            self._failures.pop(who, None)
            return None
        self._failures[who].append(self.clock())
        return Response(
            "This tutor is password-protected.\n",
            401,
            {"WWW-Authenticate": f'Basic realm="{REALM}"'},
        )


def install(app, secret, **kwargs):
    """Put a password in front of every route. Does nothing without a password.

    Every route, including the static files: an unauthenticated visitor should
    learn nothing about this app but that it wants a password.
    """
    if not secret:
        return None

    gate = Gate(secret, **kwargs)

    @app.before_request
    def _ask_for_the_password():
        supplied = request.authorization
        # The username is not checked. There is one user; a name adds a second
        # thing to remember and nothing to guess past.
        return gate.check(request.remote_addr or "?", supplied.password if supplied else None)

    return gate
