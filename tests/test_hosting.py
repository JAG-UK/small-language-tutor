"""Who gets in, from where, and what the app refuses to do at all."""

import pytest
from flask import Flask

import hosting


class TestWhereItBinds:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", ""])
    def test_recognises_the_addresses_only_this_machine_can_reach(self, host):
        assert hosting.is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com"])
    def test_recognises_the_ones_everyone_can(self, host):
        assert hosting.is_loopback(host) is False

    def test_binds_to_this_machine_unless_told_otherwise(self, monkeypatch):
        monkeypatch.delenv("SLT_HOST", raising=False)
        assert hosting.bind_host() == "127.0.0.1"

    def test_the_debugger_is_off_unless_asked_for(self, monkeypatch):
        monkeypatch.delenv("SLT_DEBUG", raising=False)
        assert hosting.debug_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_the_debugger_can_be_asked_for(self, monkeypatch, value):
        monkeypatch.setenv("SLT_DEBUG", value)
        assert hosting.debug_enabled() is True


class TestConfigurationsItWillNotStart:
    """The app used to run the Werkzeug debugger on every interface. That is a
    shell that executes what it is sent, offered to the whole network."""

    def test_the_network_without_a_password_is_refused(self):
        refusals, _ = hosting.startup_problems("0.0.0.0", None, debug=False)
        assert refusals
        assert "SLT_PASSWORD" in refusals[0]

    def test_the_debugger_on_the_network_is_refused_even_with_a_password(self):
        refusals, _ = hosting.startup_problems("0.0.0.0", "hunter2", debug=True)
        assert any("debugger" in r.lower() for r in refusals)

    def test_the_debugger_on_this_machine_is_allowed(self):
        refusals, _ = hosting.startup_problems("127.0.0.1", None, debug=True)
        assert refusals == []

    def test_loopback_needs_no_password(self):
        refusals, _ = hosting.startup_problems("127.0.0.1", None, debug=False)
        assert refusals == []

    def test_a_password_on_a_bare_http_port_is_still_worth_a_warning(self):
        # Basic auth encodes the password; it does not encrypt it.
        _, warnings = hosting.startup_problems("0.0.0.0", "hunter2", debug=False)
        assert any("encrypted" in w for w in warnings)

    def test_loopback_without_a_password_warns_about_the_far_end(self):
        _, warnings = hosting.startup_problems("127.0.0.1", None, debug=False)
        assert any("tunnel" in w for w in warnings)


class TestTheGate:
    def make(self, now=0.0, **kwargs):
        clock = {"t": now}
        gate = hosting.Gate("hunter2", clock=lambda: clock["t"], **kwargs)
        return gate, clock

    def test_the_right_password_gets_in(self):
        gate, _ = self.make()
        assert gate.check("1.2.3.4", "hunter2") is None

    def test_the_wrong_password_is_asked_again(self):
        gate, _ = self.make()
        response = gate.check("1.2.3.4", "wrong")
        assert response.status_code == 401
        assert "Basic" in response.headers["WWW-Authenticate"]

    def test_no_password_at_all_is_asked_for_one(self):
        gate, _ = self.make()
        assert gate.check("1.2.3.4", None).status_code == 401

    def test_guessing_repeatedly_stops_being_answered(self):
        gate, _ = self.make(max_failures=3, lockout=60)
        for _ in range(3):
            assert gate.check("1.2.3.4", "wrong").status_code == 401
        locked = gate.check("1.2.3.4", "wrong")
        assert locked.status_code == 429
        assert locked.headers["Retry-After"] == "60"

    def test_a_lockout_does_not_let_the_right_password_through_either(self):
        gate, _ = self.make(max_failures=2, lockout=60)
        gate.check("1.2.3.4", "wrong")
        gate.check("1.2.3.4", "wrong")
        assert gate.check("1.2.3.4", "hunter2").status_code == 429

    def test_the_lockout_lifts(self):
        gate, clock = self.make(max_failures=2, lockout=60)
        gate.check("1.2.3.4", "wrong")
        gate.check("1.2.3.4", "wrong")
        clock["t"] += 61
        assert gate.check("1.2.3.4", "hunter2") is None

    def test_one_address_guessing_does_not_lock_out_another(self):
        gate, _ = self.make(max_failures=2, lockout=60)
        gate.check("1.2.3.4", "wrong")
        gate.check("1.2.3.4", "wrong")
        assert gate.check("5.6.7.8", "hunter2") is None

    def test_getting_in_clears_the_count(self):
        gate, _ = self.make(max_failures=3, lockout=60)
        gate.check("1.2.3.4", "wrong")
        gate.check("1.2.3.4", "hunter2")
        for _ in range(3):
            assert gate.check("1.2.3.4", "wrong").status_code == 401


class TestWhatThePasswordCovers:
    def build(self, secret):
        app = Flask(__name__)

        @app.route("/")
        def index():
            return "the app"

        @app.route("/api/chat", methods=["POST"])
        def chat():
            return "expensive"

        hosting.install(app, secret)
        return app.test_client()

    def test_without_a_password_nothing_is_asked(self):
        assert self.build(None).get("/").status_code == 200

    @pytest.mark.parametrize("path", ["/", "/api/chat"])
    def test_every_route_is_behind_it(self, path):
        # Not just the pages: /api/chat is the one that costs GPU time.
        client = self.build("hunter2")
        assert client.open(path, method="POST" if "api" in path else "GET").status_code == 401

    def test_the_right_password_reaches_the_app(self):
        client = self.build("hunter2")
        response = client.get("/", auth=("anyone", "hunter2"))
        assert response.status_code == 200
        assert response.get_data(as_text=True) == "the app"

    def test_the_username_is_not_part_of_the_secret(self):
        # There is one user; a name is a second thing to remember and nothing
        # extra to guess past.
        client = self.build("hunter2")
        assert client.get("/", auth=("someone-else", "hunter2")).status_code == 200

    def test_a_visitor_without_it_learns_nothing_but_that_it_wants_one(self):
        body = self.build("hunter2").get("/").get_data(as_text=True)
        assert "the app" not in body
