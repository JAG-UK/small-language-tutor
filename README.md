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

3. **Run it** — see the two recipes below.

## Two ways to run it

Both are entirely local: the models run on your own hardware and no text leaves
it. The difference is only whether "your own hardware" is the machine in front
of you.

### On this machine

The default. Small models, nothing listening but loopback, no password needed.

```bash
export SLT_MODEL=phi4-mini:3.8b              # holds the conversation
export SLT_CRITIC_MODEL=translategemma:12b   # marks the homework
python app.py
```

Open `http://localhost:5001`. Works on a plane; nothing else on the network can
reach it. Both variables have defaults, and `SLT_CRITIC_MODEL` falls back to
`SLT_MODEL`, so plain `python app.py` also works — see
[Choosing models](#choosing-models) for what you give up.

### On the box with the GPU in it

Bigger models there, tunnel in from wherever you are. Two terminals.

**On the GPU box:**

```bash
export OLLAMA_MAX_LOADED_MODELS=2            # keep both models resident
export OLLAMA_KEEP_ALIVE=30m
export SLT_MODEL=phi4-mini:3.8b              # small and instant
export SLT_CRITIC_MODEL=translategemma:27b   # large, and never in your way
export SLT_PASSWORD='something long'
python app.py
```

It still binds to that machine's loopback. Opening a port is not part of this.

**On your laptop:**

```bash
tools/tunnel.sh you@gpu-box
```

Open `http://localhost:5001` and give the password when asked. Any username.

Without the repo on the laptop, that script is only this:

```bash
ssh -N -L 5001:127.0.0.1:5001 you@gpu-box
```

Why those models and those Ollama variables:
[When there is a GPU to spend](#when-there-is-a-gpu-to-spend). Why a tunnel
rather than a port: [Reaching it from another
machine](#reaching-it-from-another-machine).

### Settings

| variable | default | what it does |
|---|---|---|
| `SLT_MODEL` | `phi4-mini:3.8b` | the conversation partner |
| `SLT_CRITIC_MODEL` | whatever `SLT_MODEL` is | marks each message, off the critical path |
| `SLT_PASSWORD` | unset | required before anything is served; mandatory off loopback |
| `SLT_HOST` | `127.0.0.1` | where to bind. Anything else needs a password |
| `SLT_PORT` | `5001` | 5000 collides with macOS AirPlay Receiver |
| `SLT_DEBUG` | off | the Werkzeug debugger. Loopback only — it runs what it is sent |

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
  handed back sentences like `yo soy muy cansado hoy` untouched. It also
  invents explanations: on 1 correction in 9 it described a change it had not
  made, once reporting `Corrected 'true' to 'true', 'horrors' to 'errors'` for
  a sentence containing none of those words, and once telling the learner a
  word had been removed that was still there. `translategemma:12b` did that in
  0 of 31. Fabricated grammar advice is the worst thing this app can produce,
  so `grammar_checker.py` drops an explanation the sentences disprove — but a
  guard is not a substitute for a model that does not need one. phi4-mini is an
  excellent conversation partner and is still the default `SLT_MODEL`; this is
  the single best reason to set `SLT_CRITIC_MODEL`.
* **`gemma2:27b` catches everything and cannot keep quiet.** It was the only
  model to score 21/21, but it rewrote `Sí, en Madrid` — a perfectly good
  answer to "where do you live?" — every single time, and it is three times
  slower than `translategemma:12b` for the privilege. Its median rose from 9s
  to 54s when other models were also resident, which is worth remembering
  before pairing anything this large with a second model.

### When there is a GPU to spend

The split between conversation and critique pays off most here: the critic is
off the critical path, so on a box with room for a second model it can be a much
better one without the conversation slowing down at all.

On a 32GB card, the pairing to try first is a small fast partner and a large
critic:

```bash
export SLT_MODEL=phi4-mini:3.8b            # 2.5GB, replies in a fraction of a second
export SLT_CRITIC_MODEL=translategemma:27b  # ~16GB, and never in the learner's way
```

That leaves plenty of headroom. If you would rather the conversation itself were
better, `gemma2:27b` chatting with `translategemma:12b` marking is about 24GB
together and still fits.

**Set Ollama to keep both resident**, or the win is eaten by swapping weights:

```bash
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=30m
```

This matters more than the model choice. Measured on the laptop, a reply asked
for while a critique was still running took 2.3s instead of 0.3s — that is the
two models taking turns. With both resident there is nothing to swap.

Two things worth knowing before picking something enormous:

* **Bigger is not automatically faster on your hardware, and not automatically
  better here.** The 27B dense models measured 0.35–0.38s on that box while
  `gemma4:31b` took 6.23s for the same work. And on the laptop the best critic
  was 12B, not the 27B: `gemma2:27b` caught marginally more errors and rewrote
  correct sentences that `translategemma:12b` left alone. Catching everything is
  not the same as being right.
* **Reasoning models remain the wrong shape for this**, however much card you
  have. The cost is tokens of chain-of-thought before the answer, which a faster
  card shortens but does not remove.

None of the above is measured on a 5090 — it is reasoning from the laptop
numbers and from what you measured on that box. `tools/compare_models.py` runs
there too, and its answer beats this paragraph:

```bash
python tools/compare_models.py critic translategemma:12b translategemma:27b gemma2:27b
```

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

## Reaching it from another machine

The app binds to `127.0.0.1`. Nothing outside the machine it runs on can reach
it, which is the right default for something that answers with a GPU and keeps
a database of your conversations.

To use it from your laptop while it runs on the box with the card in it, forward
the port over SSH rather than opening one:

```bash
tools/tunnel.sh you@gpu-box
```

Then open `http://localhost:5001`. Leave it running; Ctrl-C closes it. The far
end never puts a port on the network — SSH carries the traffic, and supplies the
encryption that HTTP Basic does not.

### The password

Set `SLT_PASSWORD` on the machine running the app:

```bash
SLT_PASSWORD='something long' python app.py
```

Every route is then behind it, including `/api/chat` — the one that costs GPU
time. Any username works; there is one user, and a name is a second thing to
remember rather than another thing to guess past. Ten wrong guesses from one
address and it stops answering that address for five minutes.

Over a tunnel the password is not what stops the internet: the closed port is.
The password is what stops **other people with accounts on the GPU box** using
the tunnel's exit. Set one anyway.

### What it refuses to do

`SLT_HOST` will bind somewhere other than loopback, but not into a state worth
regretting:

| configuration | what happens |
|---|---|
| `SLT_HOST=0.0.0.0`, no password | refuses to start |
| `SLT_HOST=0.0.0.0` + `SLT_DEBUG=1` | refuses to start |
| `SLT_HOST=0.0.0.0` + password | starts, warns the password is readable in transit |
| default loopback | starts |

`SLT_DEBUG=1` turns on the Werkzeug debugger, which is a shell that runs
whatever is sent to it. That is fine on loopback and catastrophic anywhere else,
so it is off by default and refused off-loopback. It used to be on, on every
interface.

If you do want this on the open internet rather than through a tunnel, put a
reverse proxy with TLS in front of it and point that at the loopback port. Basic
auth over plain HTTP sends the password merely encoded.

### One user

Conversations are keyed by an id the browser makes up, and the password does not
distinguish between people who know it. Two people sharing the password get
separate conversations only as long as neither goes looking for the other's.
This is a personal tool, not a service.

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
├── hosting.py             # Where it binds, and who gets past the password
├── static/
│   └── css/
│       └── style.css      # Main stylesheet
├── templates/
│   └── index.html         # Main HTMX interface
├── tools/
│   ├── compare_models.py  # Measure a model at each of the three jobs
│   └── tunnel.sh          # Reach a remote instance over SSH
├── tests/                 # pytest; no model or network needed
└── requirements.txt       # Python dependencies
```

## Running the tests

```bash
python -m pytest tests/
```

The suite stands in for the models rather than calling them, so it needs no
Ollama and runs in about a second.
