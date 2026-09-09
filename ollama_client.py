import json

import requests


class OllamaError(RuntimeError):
    """Ollama could not answer. Distinct from the model answering badly."""


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="phi4-mini:3.8b", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout  # seconds; small models are quick, big ones are not

    def _post(self, messages, response_format=None):
        payload = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format

        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        if response.status_code == 404:
            # Ollama answers 404 for a model it has not pulled. Saying so beats
            # a bare status code, which is indistinguishable from a bad URL.
            raise OllamaError(
                f"Ollama has no model {self.model!r}. Pull it with "
                f"`ollama pull {self.model}`, or set a model it has."
            )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def chat(self, messages, language="en"):
        """Free-text reply.

        Errors come back as text because this feeds straight into the
        conversation, where an exception would lose the user's message.
        """
        try:
            return self._post(messages)
        except OllamaError as e:
            return f"Error: {e}"
        except requests.exceptions.Timeout:
            return (
                f"Error: the model took longer than {self.timeout}s to answer. "
                "A smaller model, or a longer OLLAMA timeout, would help."
            )
        except Exception as e:
            return f"Error: {e}"

    def chat_json(self, messages, schema):
        """A reply shaped by a JSON schema, parsed.

        Ollama constrains generation to the schema, so the model cannot wander
        off into prose, code fences or a half-finished object — which is what
        the pile of regex salvage this replaces was for.

        Returns None when the model or the server fails, so callers can fall
        back deliberately rather than parsing an error string as if it were an
        answer.
        """
        try:
            raw = self._post(messages, response_format=schema)
        except Exception:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def set_model(self, model):
        """Change the model being used"""
        self.model = model

    def available_models(self):
        """What this Ollama has pulled, for diagnostics and model pickers."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
