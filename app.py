import html
import json
import os
import re
import sys
from datetime import datetime

from flask import Flask, jsonify, render_template, request

import hosting
from grammar_checker import GrammarChecker
from model_profiles import profile_for
import store
from ollama_client import OllamaClient
from prompts import conversation_messages, translation_messages

app = Flask(__name__)

# Nothing is served without the password, when one is set. See hosting.py for
# what is and is not protected by it, and the README for reaching this from
# another machine.
hosting.install(app, hosting.password())

# Set SLT_MODEL to try another one — the prompts adapt to the family. See
# model_profiles.py for what the app knows about each, and the README for
# which ones are worth using.
MODEL = os.environ.get("SLT_MODEL", "phi4-mini:3.8b")

# The critic never speaks to the learner and never holds up the reply, so it can
# be a slower and more careful model than the one making conversation — two jobs
# that want different things from a model.
#
# It is a different model by default because measurement said so: over 11
# sentences x 3 trials, phi4-mini:3.8b caught 11 of 21 errors, while
# translategemma:12b caught 20 and left every correct sentence alone. An app
# whose point is catching mistakes should not ship missing half of them to save
# a pull. Set SLT_CRITIC_MODEL=$SLT_MODEL to go back to one model on a machine
# that cannot hold two. See "Choosing models" in the README, and
# tools/compare_models.py to check any of it on your own machine.
DEFAULT_CRITIC = "translategemma:12b"
CRITIC_MODEL = os.environ.get("SLT_CRITIC_MODEL", DEFAULT_CRITIC)

ollama = OllamaClient(model=MODEL)
PROFILE = profile_for(MODEL)

# A deliberately generous timeout when the critic is its own model: whatever it
# is, it was chosen for care rather than speed, and nobody is waiting on it.
critic = None
grammar_checker = None


def use_critic(model):
    """Point the critic at a model, and say which one it is.

    Separate from import so that importing the app — which the tests do —
    asks Ollama nothing.
    """
    global CRITIC_MODEL, critic, grammar_checker
    CRITIC_MODEL = model
    critic = ollama if model == MODEL else OllamaClient(model=model, timeout=300)
    grammar_checker = GrammarChecker(critic)
    return grammar_checker


use_critic(CRITIC_MODEL)


def check_models(available=None):
    """Warn about models that are configured but not pulled.

    A missing model is not an error the app can answer: the conversation would
    come back as an apology and the critic would silently stop marking. Better
    to say which one and what to type. If the critic in particular is missing,
    fall back to the conversation model — half a critic is worth more than none,
    and the warning says exactly what was given up.
    """
    if available is None:
        available = ollama.available_models()
    if not available:
        # Ollama not running, or not answering yet. Not this function's problem
        # to diagnose; the first real request will say so plainly.
        return []

    notes = []
    if MODEL not in available:
        notes.append(f"SLT_MODEL {MODEL!r} is not pulled. `ollama pull {MODEL}`.")
    if CRITIC_MODEL not in available:
        notes.append(
            f"SLT_CRITIC_MODEL {CRITIC_MODEL!r} is not pulled, so corrections fall back to "
            f"{MODEL!r} — which catches about half as many mistakes. "
            f"`ollama pull {CRITIC_MODEL}` to fix that."
        )
        use_critic(MODEL)
    return notes

# Current conversation state (in-memory, per session)
conversations = {}  # Simple dict for now, keyed by session_id


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """The reply, and only the reply.

    This used to correct the message, then look for hints, then answer it —
    three round trips to a model before the learner saw a word back. The two
    critical ones are worth having but not worth waiting for: the reply is the
    conversation, the critique is homework. So this returns as soon as there is
    something to say, and hands back a turn number the client uses to ask for
    the critique afterwards, at /api/review.

    Measured on phi4-mini:3.8b: 2.3s to a reply before, 0.4s after.
    """
    data = request.json
    session_id = data.get('session_id', 'default')
    user_message = data.get('message', '')
    language = data.get('language', 'es')  # Default to Spanish
    tone = data.get('tone', 'friendly')  # Default to friendly

    if session_id not in conversations:
        conversations[session_id] = store.new_conversation(language, tone)

    conv = conversations[session_id]
    # Both can be changed from the header mid-conversation, and the review that
    # follows this turn reads them from here rather than from its own request.
    conv['language'] = language
    conv['tone'] = tone

    conv['messages'].append(
        {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()}
    )
    turn = len(conv['messages']) - 1  # the message /api/review will critique

    # How much of the transcript travels, and whether the instruction goes in a
    # system turn at all, is the model family's call.
    messages_for_llm = conversation_messages(PROFILE, language, tone, conv["messages"])

    ai_response = ollama.chat(messages_for_llm, language)
    ai_msg = {"role": "assistant", "content": ai_response, "timestamp": datetime.now().isoformat()}
    conv['messages'].append(ai_msg)
    store.save(conv)  # every turn, so nothing is lost to a restart

    return jsonify({
        'response': ai_response,
        'turn': turn,
        'conversation_id': conv['id'],
        'messages': conv['messages'],
    })

@app.route('/api/review', methods=['POST'])
def review():
    """What was wrong with one turn, and how it could sound more like a native.

    Asked for after the reply is on screen, so the time it costs is spent while
    the learner is reading rather than while they are waiting for an answer.

    It is a separate request rather than a background thread because the work
    only matters if someone is still there to read it: a client that has gone
    away simply never asks. The two calls stay in series — running them at once
    measured no faster, since Ollama serialises requests to a single model.
    """
    data = request.json
    session_id = data.get('session_id', 'default')
    turn = data.get('turn')

    conv = conversations.get(session_id)
    if conv is None or not isinstance(turn, int) or not 0 <= turn < len(conv['messages']):
        # Conversations live in memory, so a restart loses them, and there is
        # nothing to say about a turn we no longer hold.
        return jsonify({'error': 'No such turn'}), 404

    message = conv['messages'][turn]
    if message['role'] != 'user':
        return jsonify({'error': 'Only the learner\'s own turns are reviewed'}), 400

    if turn in conv['reviewed']:
        return render_learning_points(conv)
    conv['reviewed'].add(turn)

    text = message['content']
    language = conv['language']
    tone = conv['tone']
    # Up to and including the turn being judged. The reply that came after it
    # is not evidence about it, and would push the sentence out of view.
    context = conv['messages'][: turn + 1]

    correction = grammar_checker.check_message(text, context, language, tone)
    if correction.get('has_errors'):
        conv['corrections'].append({
            'message': text,
            'corrected': correction.get('corrected'),
            'explanation': correction.get('explanation'),
            'timestamp': datetime.now().isoformat(),
        })

    hints_result = grammar_checker.get_hints(text, context, language, tone)
    if hints_result.get('has_hints') and hints_result.get('hints'):
        conv['hints'].append({
            'message': text,
            'hints': hints_result.get('hints', []),
            'timestamp': datetime.now().isoformat(),
        })

    store.save(conv)  # the learning points belong to the conversation
    return render_learning_points(conv)

@app.route('/api/practice', methods=['POST'])
def practice():
    """Check a practice sentence without adding to conversation.

    Unlike the chat critique this one is on the critical path — the learner
    pressed Check and is waiting for the answer — so a deliberately slow
    SLT_CRITIC_MODEL will be felt here.
    """
    data = request.json
    sentence = data.get('sentence', '')
    language = data.get('language', 'es')

    correction = grammar_checker.check_message(sentence, [], language)
    return jsonify(correction)

@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate between English and target language (bidirectional)"""
    data = request.json
    phrase = data.get('phrase', '')
    target_language = data.get('language', 'es')
    direction = data.get('direction', 'en-to-target')
    
    if not phrase:
        return jsonify({'error': 'No phrase provided'}), 400
    
    messages = translation_messages(PROFILE, phrase, target_language, direction)
    response_language = target_language if direction == "en-to-target" else "en"

    try:
        translation = ollama.chat(messages, response_language)
        return jsonify({'translation': translation.strip()})
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

#: The bullet a model puts at the start of a line: "- ", "*   ", "• ".
_LEADING_BULLET = re.compile(r"^[-*\u2022]+\s+")


def explanation_lines(text):
    """An explanation as the lines it was actually written as.

    Asked what changed, a model answers with a markdown list — a lead-in, then
    one point per line, each opening with a bullet. HTML collapses the newlines,
    so it all arrived as a single run-on paragraph with stray asterisks in it.
    The bullets go (the panel makes its own), the line breaks stay.
    """
    lines = []
    for line in (text or "").splitlines():
        line = _LEADING_BULLET.sub("", line).strip()
        # A bullet with nothing after it: a point the model opened and dropped.
        if line and line.strip("-*\u2022 "):
            lines.append(line)
    return lines


def render_learning_points(conv):
    """The learning panel, as HTML.

    There is one of these, on purpose. The panel used to be built twice — here,
    and again in JavaScript in the page — and the two had to agree about the
    shape of a correction and a hint. When that shape changed, every browser
    already holding the old page rendered "[object Object]" until it was
    reloaded, which is not a thing a server-rendered app should be able to do to
    itself. The page now displays what this returns and knows nothing about the
    shape.
    """
    corrections = (conv or {}).get('corrections', [])
    hints = (conv or {}).get('hints', [])

    if not corrections and not hints:
        return '<p class="empty-state">No learning points yet. Keep practicing!</p>'
    
    # Combine corrections and hints, sorted by timestamp (most recent first)
    all_items = []
    for corr in corrections:
        all_items.append({
            'type': 'correction',
            'timestamp': corr.get('timestamp', ''),
            'data': corr
        })
    for hint in hints:
        all_items.append({
            'type': 'hint',
            'timestamp': hint.get('timestamp', ''),
            'data': hint
        })
    
    # Sort by timestamp, most recent first
    all_items.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Render as HTML
    html_output = ''
    for item in all_items:
        if item['type'] == 'correction':
            corr = item['data']
            original = html.escape(str(corr.get('message', '')))
            corrected = html.escape(str(corr.get('corrected', '')))
            explanation_raw = corr.get('explanation', '')
            # Saved conversations may hold a list; models write one anyway,
            # flattened into a single string.
            if not isinstance(explanation_raw, list):
                explanation_raw = explanation_lines(explanation_raw)
            explanation = '<br>'.join(
                html.escape(str(line)) for line in explanation_raw if str(line).strip()
            )
            
            html_output += f'''
            <div class="correction-item">
                <div class="item-label">Correction</div>
                <div class="original">{original}</div>
                <div class="corrected">✓ {corrected}</div>
                <div class="explanation">{explanation}</div>
            </div>
            '''
        else:  # hint
            hint_data = item['data']
            message = html.escape(hint_data.get('message', ''))
            hints_list = hint_data.get('hints', [])
            
            hints_html = ''
            for hint in hints_list:
                # Conversations saved before hints were split carry plain strings.
                if isinstance(hint, dict):
                    suggestion = html.escape(str(hint.get('suggestion', '')))
                    why = html.escape(str(hint.get('why', '')))
                    hints_html += (f'<li><span class="hint-suggestion">{suggestion}</span>'
                                   f'<span class="hint-why">{why}</span></li>')
                else:
                    hints_html += f'<li>{html.escape(str(hint))}</li>'
            
            html_output += f'''
            <div class="hint-item">
                <div class="item-label">Hint</div>
                <div class="hint-message">{message}</div>
                <ul class="hints-list">{hints_html}</ul>
            </div>
            '''
    
    return html_output


@app.route('/api/corrections', methods=['GET'])
def get_corrections():
    """The learning panel for a conversation, for the page to show on load."""
    return render_learning_points(conversations.get(request.args.get('session_id', 'default')))

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """The saved conversations, as the list the page shows.

    Rendered here rather than in the page for the same reason the learning panel
    is: one renderer cannot disagree with itself, and a browser holding an old
    copy of the page cannot render the new shape wrongly.
    """
    saved = store.recent()
    if not saved:
        return '<p class="empty-state">Nothing saved yet. Start talking.</p>'

    items = ''
    for row in saved:
        title = html.escape(row['title'])
        when = row['updated_at'].strftime('%d %b %H:%M') if row['updated_at'] else ''
        turns = row['message_count']
        items += f'''
        <div class="conversation-item" data-id="{row['id']}">
            <button class="conversation-open" onclick="openConversation({row['id']})">
                <span class="conversation-title">{title}</span>
                <span class="conversation-meta">{html.escape(when)} · {turns} messages</span>
            </button>
            <button class="conversation-delete" title="Delete"
                    onclick="deleteConversation({row['id']})">×</button>
        </div>
        '''
    return items


@app.route('/api/conversations/<int:conv_id>/open', methods=['POST'])
def open_conversation(conv_id):
    """Pick a stored conversation up where it was left.

    The transcript alone would only be something to look at: the partner reads
    its history from the server, so without putting it back there the model
    would answer the next message knowing nothing of what came before.
    """
    data = request.json or {}
    session_id = data.get('session_id', 'default')

    conv = store.load(conv_id)
    if conv is None:
        return jsonify({'error': 'No such conversation'}), 404

    conversations[session_id] = conv
    return jsonify({
        'id': conv['id'],
        'language': conv['language'],
        'tone': conv['tone'],
        'messages': conv['messages'],
        'panel': render_learning_points(conv),
    })


@app.route('/api/conversations/<int:conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    """Forget one. Also from memory, so a page still holding its session id
    cannot write it back on the next turn."""
    if not store.delete(conv_id):
        return jsonify({'error': 'No such conversation'}), 404

    for session_id, conv in list(conversations.items()):
        if conv.get('id') == conv_id:
            del conversations[session_id]
    return jsonify({'deleted': conv_id})


def main():
    """Start the server, or explain why the configuration is not one to start.

    It used to run with debug=True on 0.0.0.0, which is the Werkzeug debugger —
    a shell that executes what it is sent — offered to the whole network.
    """
    host, port = hosting.bind_host(), hosting.bind_port()
    debug = hosting.debug_enabled()
    refusals, warnings = hosting.startup_problems(host, hosting.password(), debug)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if refusals:
        for refusal in refusals:
            print(f"refusing to start: {refusal}", file=sys.stderr)
        raise SystemExit(2)

    for note in check_models():
        print(f"warning: {note}", file=sys.stderr)

    where = "this machine only" if hosting.is_loopback(host) else "the network"
    locked = "password required" if hosting.password() else "no password"
    models = MODEL if CRITIC_MODEL == MODEL else f"{MODEL} + {CRITIC_MODEL}"
    print(f"Language Tutor on http://{host}:{port} — {where}, {locked}", file=sys.stderr)
    print(f"  talking with {models}", file=sys.stderr)
    app.run(debug=debug, host=host, port=port)


if __name__ == '__main__':
    main()

