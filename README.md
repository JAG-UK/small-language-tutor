# small-language-tutor

Language learning reinforcement app based on local small language models.

## Features

- **Interactive Chat**: Practice conversations in your target language with an AI tutor
- **Grammar Corrections**: Real-time feedback on mistakes with explanations
- **Practice Area**: Test sentences before sending them to the conversation
- **Conversation History**: Every turn is saved as it happens; reopen any past
  conversation, with its corrections, and carry on from where you left off
- **Report Card**: What your recent mistakes have in common, and what to work on
- **Let me start**: The tutor opens, in a situation chosen for what you keep
  getting wrong — so having nothing to say is not a reason not to practise

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
python app.py
```

Open `http://localhost:5001`. Works on a plane; nothing else on the network can
reach it.

That runs `phi4-mini:3.8b` for the conversation and `translategemma:12b` for the
corrections, which is the pairing the measurements favour — see
[Choosing models](#choosing-models). On a machine that cannot hold both, one
will do:

```bash
export SLT_CRITIC_MODEL=phi4-mini:3.8b   # the same model for both jobs
```

It catches about half as many mistakes. If the critic has not been pulled the
app says so at startup and falls back to this on its own, rather than quietly
marking nothing.

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
| `SLT_CRITIC_MODEL` | `translategemma:12b` | marks each message, off the critical path. Set it to `$SLT_MODEL` to run one model |
| `SLT_PASSWORD` | unset | required before anything is served; mandatory off loopback |
| `SLT_HOST` | `127.0.0.1` | where to bind. Anything else needs a password |
| `SLT_PORT` | `5001` | 5000 collides with macOS AirPlay Receiver |
| `SLT_DB` | `database.db` beside `app.py` | where conversations are kept |
| `SLT_DEBUG` | off | the Werkzeug debugger, and reloading on edit. Loopback only — it runs what it is sent |

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

**The default (~11GB resident):** a small fast model talks, a careful one marks.

```bash
export SLT_MODEL=phi4-mini:3.8b
export SLT_CRITIC_MODEL=translategemma:12b
```

Replies come back in 0.3s and the corrections catch up a few seconds later. The
one cost is that sending a message while a critique is still running takes about
2.3s instead of 0.3s. This is what you get without setting anything.

**One model, if you can only spare ~8GB: `translategemma:12b`.** The only model
measured here that is good at all three jobs — 1.9s replies, and much the best
critic of the lot. Despite the name it holds a conversation perfectly well.

```bash
export SLT_MODEL=translategemma:12b
export SLT_CRITIC_MODEL=translategemma:12b
```

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
  excellent conversation partner and is still the default `SLT_MODEL`, but it is
  no longer the default critic — an app whose point is catching mistakes should
  not ship missing half of them to save a pull.
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

There are two good ways to reach it anyway, and which one depends on what you
are holding.

### From a laptop, over SSH

Forward the port rather than opening one:

```bash
tools/tunnel.sh you@gpu-box
```

Then open `http://localhost:5001`. Leave it running; Ctrl-C closes it. The far
end never puts a port on the network — SSH carries the traffic, and supplies the
encryption that HTTP Basic does not.

This is the right tool between two machines on the same network, or over a link
you already have. It is the wrong one for a phone: iOS and Android both suspend
the SSH app the moment you switch to the browser, which is exactly when the
forward needs to be alive, and SSH is TCP, so the session dies every time the
handset moves between wifi and cell.

### From a phone, over WireGuard

Put the phone on the network the box is on, and reach it by its LAN address like
anything else there. WireGuard is the right shape for this: it runs at the OS
level so the browser benefits without a second app in the foreground, and it is
UDP, so roaming between wifi and cell reconnects instead of breaking.

The app then binds to the LAN address rather than loopback, which means a
password — and it will not start without one:

```bash
export SLT_HOST=192.168.1.42        # the box's address on your VLAN
export SLT_PASSWORD='something long'
python app.py
```

One UDP port for WireGuard is the only thing your firewall needs to allow, and
the tutor's own port stays off the public internet entirely. The startup warning
about HTTP Basic being readable in transit is about a bare HTTP port; inside
WireGuard the traffic is already encrypted on the wire, so the password is not
travelling in the clear. It is still worth setting, because it is what stops
anything *else* on the VLAN from using your GPU.

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

It is also what makes the server pick up edits. With it off, Flask caches
templates for the life of the process, so a change to `index.html` does nothing
until a restart — set it while working on the app, and leave it off otherwise.

If you do want this on the open internet — rather than through a tunnel or a
VPN — put a reverse proxy with TLS in front of it and point that at the loopback
port. Basic auth over plain HTTP sends the password merely encoded, and the open
internet is the one place where that matters most.

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

### Conversations keep themselves

Every turn is written to SQLite as it happens, so nothing depends on remembering
to save: close the laptop, drop the tunnel, restart the server, and the
conversation is still there. **Conversations** lists them; opening one puts it
back on the server as well as on screen, so the partner still has the history
and you can carry on rather than only read it. Reopening does not re-mark
anything already marked.

The database is at `database.db` beside `app.py`, or wherever `SLT_DB` says.
Beside the file rather than in the working directory, because a relative path
silently gives you a different database when the app is started from somewhere
else.

Columns the model gains are added to an existing database at startup.
`create_all()` creates missing tables and stops there, so `hints` — added to the
model after the file existed — was simply absent, and every save failed with
`no such column: conversations.hints`.

### Letting the tutor go first

**…or let me start** has the tutor open the conversation. Two problems, one
answer. Asked simply to start talking, a small model says "¡Hola! ¿Cómo estás?"
and says it again tomorrow; a concrete role gets a concrete opening, and a deck
of them gets variety without hoping for it. And a learner who has to invent the
subject every time practises whatever they can already say, which is the
opposite of the point.

So `scenarios.py` holds a deck of situations, each tagged with what it drags out
of you, and the choice is made from what your corrections show you keep
dropping. Muddle `ser` and `estar` twice and you get the portera on the stairs
about the builders, or the neighbour who has decided the bags by the bins are
somebody's fault — situations that cannot be answered without choosing between
them.

The deck is set where the learner actually lives, which is the difference
between practising a language and practising a phrasebook: a resident does not
check into a hotel or ask where the station is, they argue about a deposit, take
the wrong paper to the gestoría, and get talked out of their usual order at the
market. And each situation gives the other person something they want — an
opinion, a complaint, a thing to sell, a suspicion. A partner with nothing at
stake asks "¿y tú?" until the conversation dies of it.

What you need to practise is counted, not asked for. `what_it_shows` maps a
corrected word to one of five things and refuses to guess past them: accents and
capitals work in any language the app offers, `ser`/`estar` and articles are a
short Spanish table, and agreement is an `o`/`a` swap or a suffix. A looser
version of that last rule called `comer` → `comí` an agreement slip, and
`hablo` → `hablé` too once folding away the accent left an ending that looked
like a gender one. Both are tense. A scenario chosen for the wrong reason is a
wasted conversation, so a swap it cannot explain contributes nothing and the
deck falls back to variety.

The settings each name both parts and end with something for the model to do.
"You are a pharmacist. The learner has come in feeling unwell" left phi4-mini
describing its own symptoms — which is the learner's job, and the entire point
of the exercise.

### The situation lasts the whole conversation

A scenario spent on the opening alone buys one good line. `POST /api/chat` used
to build the partner's instruction from the language and the tone and nothing
else, so the portera stopped being the portera on turn two and reverted to a
native speaker asking how you are — which is what the deck exists to avoid. The
situation now travels with every turn, and is stored on the conversation, so
reopening one from last week picks it up in character rather than out of it.

That is why each entry has a `setting` and an `opening` rather than one field
holding both. The setting — who you are, and what you want — is true for the
whole conversation. The opening is a first move, and "Greet them and ask what
the matter is" said on every turn had the partner saying hello again each time
it spoke.

### The report card

**Report card** looks over the corrections from recent conversations and says
what they have in common. It is in two halves, and they are not equally
trustworthy — the panel shows them as such.

The first half is counted. That you wrote `anos` for `años` four times is
arithmetic over the corrections, and arithmetic does not invent anything. Only
one-for-one word swaps are counted: a correction that rewrites half a sentence
says something about that sentence and nothing countable about a habit, and
counting it would bury the swaps that do repeat.

The second half is a model's reading of the same mistakes, grouped into a few
areas with something to practise. Every example it gives is checked against what
the learner actually wrote — loosely, since accents and capitals are exactly what
tends to differ between the quote and the original. An example that cannot be
found is dropped, and a theme left with none goes with it. Asked to find
patterns, a model will happily illustrate one with a plausible mistake nobody
made, and a report card that invents your mistakes is worse than no report card.

Two things learned while building it, both the same lesson in different clothes:

* The corrections go to the model as bare wrote/should-be pairs, without the
  explanations that came with them. Those are the least reliable thing the
  critic produces, and a wrong one repeated under a heading is a wrong lesson.
* `examples` is `minItems: 1` in the schema. Without it, a model that ran long
  in the prose fields closed the array empty, every theme was dropped for having
  nothing behind it, and the report came back saying nothing stood out — on
  about half of all runs. With that and a prompt asking for one sentence per
  field, four runs out of four produced four themes.

It costs one model call over the whole pile, so it takes 20 seconds or so and is
asked for rather than produced on every turn. Below four corrections it does not
ask at all: there is no pattern in three mistakes, only three mistakes.

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
├── models.py              # The conversations table, and keeping it up to date
├── store.py               # Saving, reopening, listing and deleting
├── report.py             # What the mistakes add up to
├── scenarios.py           # Something to talk about, chosen for what you get wrong
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
├── requirements-dev.txt   # the above, plus pytest
├── tools/
│   ├── compare_models.py  # Measure a model at each of the three jobs
│   └── tunnel.sh          # Reach a remote instance over SSH
├── tests/                 # pytest; no model or network needed
└── requirements.txt       # Python dependencies
```

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

The suite stands in for the models rather than calling them, so it needs no
Ollama and runs in about a second. GitHub Actions runs it on every push and pull
request against Python 3.12, 3.13 and 3.14 — the dependency pins that were here
before could not be imported at all on the newer two, which is exactly the sort
of thing that works perfectly on the machine it was written on.
