# small-language-tutor

Language learning reinforcement app based on local small language models.

## Features

- **Interactive Chat**: Practice conversations in your target language with an AI tutor
- **Grammar Corrections**: Real-time feedback on mistakes with explanations
- **Practice Area**: Test sentences before sending them to the conversation
- **Conversation History**: Save and review past conversations with corrections

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve

   # In another terminal, pull the two models this app is tuned for:
   ollama pull phi4-mini:3.8b
   ollama pull translategemma:12b
   ```

3. **Configure the models** (optional):
   ```bash
   export SLT_MODEL=phi4-mini:3.8b          # holds the conversation
   export SLT_CRITIC_MODEL=translategemma:12b   # marks the homework
   ```
   Both have defaults, and `SLT_CRITIC_MODEL` falls back to `SLT_MODEL`, so the
   app runs with neither set. See [Choosing models](#choosing-models) for why
   those two, and for what a machine with less memory should do instead.

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   Navigate to `http://localhost:5001`
   
   Note: Port 5001 is used instead of 5000 to avoid conflicts with macOS AirPlay Receiver

## Choosing models

The app asks a model to do three quite different jobs, and they do not want the
same thing from one:

| job | what it needs | who is waiting |
|---|---|---|
| **Conversation partner** | speed, and the discipline to stay in the target language and stay brief | the learner, mid-sentence |
| **Critic** | accuracy, and the judgement to leave a correct sentence alone | nobody — it runs after the reply |
| **Translator** | one phrase, rendered plainly, with no commentary | the learner, but they asked |

Because the critic runs after the reply is on screen (see Architecture below),
it can be a slower and more careful model than the one making conversation.
That is what `SLT_CRITIC_MODEL` is for.

### Recommendations

**One model, if you can spare ~8GB: `translategemma:12b`.** It is the only
model measured here that is good at all three jobs — 1.9s replies, and much the
best critic of the lot. Despite the name it holds a conversation perfectly well.

```bash
export SLT_MODEL=translategemma:12b
```

**Two models, for the snappiest conversation (~11GB resident):** let a small
fast model talk and the careful one mark:

```bash
export SLT_MODEL=phi4-mini:3.8b
export SLT_CRITIC_MODEL=translategemma:12b
```

Replies come back in 0.3s and the corrections catch up a few seconds later.
The one cost is that sending a message while a critique is still running takes
about 2.3s instead of 0.3s.

**Tight on memory: `translategemma:4b` (3.3GB)** as the single model. It catches
almost everything, but rewrites about half the sentences that were already fine.

**Avoid reasoning models** — `qwen3:*`, `deepseek-r1`, `phi4-mini-reasoning`.
They are accurate and hopeless here: 28s for a chat reply, and 119-137s for a
correction. `qwen3.5:9b` also wraps its corrections in markdown asterisks.

### The measurements

From `tools/compare_models.py`, on an M-series MacBook, Spanish, three trials of
each case. The correction set is 11 sentences: 7 containing a specific error
(accents, gender, number, ser/estar, tense) and 4 that are already correct.

| model | size | reply | catches errors | leaves good sentences alone |
|---|---|---|---|---|
| `phi4-mini:3.8b` | 2.5GB | **0.3s** | 11/21 | 10/12 |
| `translategemma:4b` | 3.3GB | 0.6s | 20/21 | 6/12 |
| `translategemma:12b` | 8.1GB | 1.9s | **20/21** | **12/12** |
| `gemma2:27b` | 15.6GB | 2.6s | **21/21** | 9/12 |
| `qwen3:4b` | 2.5GB | 28.5s | — | — |
| `qwen3.5:9b` | 6.6GB | — | 137s per correction | — |

All four of the usable models stayed in Spanish on every turn, and none needed
more than about 18 words to answer.

Two results worth knowing about, because they are not what the names suggest:

* **`phi4-mini` is a poor critic** — it caught half the errors, and repeatedly
  handed back sentences like `yo soy muy cansado hoy` untouched. It is an
  excellent conversation partner and it is still the default `SLT_MODEL`, but
  in an app whose point is catching mistakes, leaving it to mark its own
  homework is the weakest link. This is the single best reason to set
  `SLT_CRITIC_MODEL`.
* **`gemma2:27b` catches everything and cannot keep quiet.** It was the only
  model to score 21/21, but it rewrote `Sí, en Madrid` — a perfectly good
  answer to "where do you live?" — every single time, and it is three times
  slower than `translategemma:12b` for the privilege. Its median rose from 9s
  to 54s when other models were also resident, which is worth remembering
  before pairing anything this large with a second model.

### Re-running this yourself

These numbers are one laptop, one language, and a small test set. A different
machine or target language may well order the models differently, so the
comparison is a script rather than a claim:

```bash
python tools/compare_models.py critic phi4-mini:3.8b translategemma:12b gemma2:27b
python tools/compare_models.py chat phi4-mini:3.8b translategemma:12b
python tools/compare_models.py translate translategemma:4b phi4-mini:3.8b
```

Nothing here is deterministic — a single run genuinely flips results between
models, which is why it runs each case three times.

### Adding a family the app does not know

`model_profiles.py` holds what has to change per family: whether it has a system
turn at all (Gemma 1-3 do not), whether long prompts help or hurt, and how much
transcript to carry. Add a row to `FAMILIES` and the prompts adapt. An unknown
model gets a conservative default — short prompts, a system turn assumed.

## Architecture

- **Backend**: Flask (Python)
- **Frontend**: HTMX + vanilla CSS
- **Database**: SQLite
- **SLM**: Ollama API integration
- **Future**: Architecture supports voice conversations (to be implemented)

### A turn is two requests

`POST /api/chat` returns the conversational reply and nothing else, along with a
turn number. Once that reply is on screen the client calls `POST /api/review`
with the turn number, which produces the correction and the naturalness hints
and fills in the learning panel behind the conversation.

This is the difference between waiting 2.0s and 0.28s for a reply. The critique
costs the same either way; it just stops happening in front of the learner. The
critic is shown the conversation up to and including the turn it is judging —
`Sí, en Madrid` is a fragment on its own and a good answer to a question, and a
critic that cannot see the question misreads it.

Model output is asked for against a JSON schema, which Ollama enforces during
generation, so responses cannot arrive as prose or half-formed JSON. Whether a
correction happened is derived from whether the text actually changed rather
than from the model's own `has_errors` flag, because small models are markedly
better at producing a corrected sentence than at judging whether they produced
one.

## Project Structure

```
small-language-tutor/
├── app.py                 # Flask backend server
├── models.py              # Database models
├── ollama_client.py       # SLM integration wrapper
├── grammar_checker.py     # Corrections and hints, and the guards on them
├── prompts.py             # Everything the models are asked, in one place
├── model_profiles.py      # What each model family needs said to it
├── static/
│   └── css/
│       └── style.css      # Main stylesheet
├── templates/
│   └── index.html         # Main HTMX interface
├── tools/
│   └── compare_models.py  # Measure a model at each of the three jobs
├── tests/                 # pytest; no model or network needed
└── requirements.txt       # Python dependencies
```

## Running the tests

```bash
python -m pytest tests/
```

The suite stands in for the models rather than calling them, so it needs no
Ollama and runs in about a second.
