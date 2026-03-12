from flask import Flask, render_template, request, jsonify
from models import Session, Conversation
from ollama_client import OllamaClient
from grammar_checker import GrammarChecker
from datetime import datetime
import json
import html

app = Flask(__name__)

# Initialize clients
ollama = OllamaClient(model="phi4-mini:3.8b")  # Configurable
grammar_checker = GrammarChecker(ollama)

# Current conversation state (in-memory, per session)
conversations = {}  # Simple dict for now, keyed by session_id

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    session_id = data.get('session_id', 'default')
    user_message = data.get('message', '')
    language = data.get('language', 'es')  # Default to Spanish
    tone = data.get('tone', 'friendly')  # Default to friendly
    
    # Get or create conversation
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'language': language,
            'tone': tone,
            'corrections': [],
            'hints': []
        }
    
    conv = conversations[session_id]
    # Update tone if changed
    conv['tone'] = tone
    
    # Add user message
    user_msg = {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()}
    conv['messages'].append(user_msg)
    
    # Check grammar (with tone awareness)
    correction = grammar_checker.check_message(user_message, conv['messages'], language, tone)
    
    # Debug: Log correction result
    print(f"DEBUG: Correction result: {correction}")
    print(f"DEBUG: has_errors value: {correction.get('has_errors')}, type: {type(correction.get('has_errors'))}")
    
    # Handle both boolean and string values for has_errors
    has_errors = correction.get('has_errors', False)
    if isinstance(has_errors, str):
        has_errors = has_errors.lower() in ('true', '1', 'yes')
    
    if has_errors:
        print(f"DEBUG: Adding correction to session {session_id}")
        conv['corrections'].append({
            'message': user_message,
            'corrected': correction.get('corrected'),
            'explanation': correction.get('explanation'),
            'timestamp': datetime.now().isoformat()
        })
    else:
        print(f"DEBUG: No errors detected or has_errors is False")
    
    # Get hints for naturalness improvement (with tone awareness)
    hints_result = grammar_checker.get_hints(user_message, conv['messages'], language, tone)
    if hints_result.get('has_hints') and hints_result.get('hints'):
        conv['hints'].append({
            'message': user_message,
            'hints': hints_result.get('hints', []),
            'timestamp': datetime.now().isoformat()
        })
    
    # Get AI response
    tone_instructions = {
        'friendly': 'warm, casual, and approachable. Use informal language, friendly expressions, and show interest in the conversation.',
        'professional': 'polite, formal, and business-like. Use formal language, proper titles if appropriate, and maintain a respectful distance.',
        'flirty': 'playful, charming, and slightly suggestive. Use teasing language, compliments, and create a light romantic or playful atmosphere.'
    }
    
    tone_desc = tone_instructions.get(tone, tone_instructions['friendly'])
    
    system_prompt = f"""You are a {language} native speaker having a natural conversation with the user. The tone of this conversation is {tone}: be {tone_desc}

CRITICAL RULES:
1. You MUST respond ENTIRELY in {language}. Do NOT use English or any other language.
2. Every word, phrase, and sentence must be in {language} only.
3. Maintain the {tone} tone throughout your responses - match the user's tone and style.
4. Keep responses natural, conversational, and appropriate for a language learner.
5. Keep responses brief (2-3 sentences max).
6. If you need to explain something, explain it in {language}, not in English.
7. If you don't understand what the user is saying, or it doesn't make sense in the context of the rest of the conversation, ask them for clarification.

Remember: This is a language practice conversation in a {tone} tone. The entire conversation must be in {language}."""
    
    messages_for_llm = [
        {"role": "system", "content": system_prompt}
    ] + [{"role": msg["role"], "content": msg["content"]} for msg in conv['messages'][-10:]]  # Last 10 messages for context
    
    ai_response = ollama.chat(messages_for_llm, language)
    ai_msg = {"role": "assistant", "content": ai_response, "timestamp": datetime.now().isoformat()}
    conv['messages'].append(ai_msg)
    
    # Prepare correction data for response (if errors found)
    correction_data = None
    if has_errors:
        correction_data = {
            'message': user_message,
            'corrected': correction.get('corrected'),
            'explanation': correction.get('explanation'),
            'timestamp': datetime.now().isoformat()
        }
    
    # Prepare hints data for response (if hints found)
    hints_data = None
    if hints_result.get('has_hints') and hints_result.get('hints'):
        hints_data = {
            'message': user_message,
            'hints': hints_result.get('hints', []),
            'timestamp': datetime.now().isoformat()
        }
    
    # Return everything in one response - include all corrections and hints for this conversation
    return jsonify({
        'response': ai_response,
        'correction': correction_data,  # New correction from this message (if any)
        'hints': hints_data,  # New hints from this message (if any)
        'all_corrections': conv.get('corrections', []),  # Full history
        'all_hints': conv.get('hints', []),  # Full history
        'messages': conv['messages']
    })

@app.route('/api/practice', methods=['POST'])
def practice():
    """Check a practice sentence without adding to conversation"""
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
    
    # Language names for prompts
    language_names = {
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian'
    }
    lang_name = language_names.get(target_language, target_language)
    
    # Determine translation direction
    if direction == 'en-to-target':
        # English to target language
        system_prompt = f"""You are a professional translator. Translate the following English phrase into {lang_name}. 
        Provide a natural, conversational translation that a native speaker would use. 
        Do not provide explanations, only the translation."""
        user_prompt = f"Translate this to {lang_name}: {phrase}"
        response_language = target_language
    else:
        # Target language to English
        system_prompt = f"""You are a professional translator. Translate the following {lang_name} phrase into English. 
        Provide a natural, conversational translation that a native English speaker would use. 
        Do not provide explanations, only the translation."""
        user_prompt = f"Translate this {lang_name} phrase to English: {phrase}"
        response_language = 'en'
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        translation = ollama.chat(messages, response_language)
        return jsonify({'translation': translation.strip()})
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@app.route('/api/corrections', methods=['GET'])
def get_corrections():
    """Get corrections and hints for current conversation"""
    session_id = request.args.get('session_id', 'default')
    corrections = []
    hints = []
    if session_id in conversations:
        corrections = conversations[session_id].get('corrections', [])
        hints = conversations[session_id].get('hints', [])
    
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
            
            # Handle explanation as string or list
            if isinstance(explanation_raw, list):
                explanation = '<br>'.join([html.escape(str(item)) for item in explanation_raw])
            else:
                explanation = html.escape(str(explanation_raw))
            
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
            for i, hint_text in enumerate(hints_list, 1):
                hints_html += f'<li>{html.escape(hint_text)}</li>'
            
            html_output += f'''
            <div class="hint-item">
                <div class="item-label">Hint</div>
                <div class="hint-message">{message}</div>
                <ul class="hints-list">{hints_html}</ul>
            </div>
            '''
    
    return html_output

@app.route('/api/save', methods=['POST'])
def save_conversation():
    """Save conversation to database"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id not in conversations:
        return '<span style="color: red;">No conversation to save</span>', 400
    
    conv = conversations[session_id]
    session = Session()
    
    db_conv = Conversation(
        title=conv['messages'][0]['content'][:50] if conv['messages'] else "Untitled",
        language=conv['language'],
        messages=json.dumps(conv['messages']),
        corrections=json.dumps(conv.get('corrections', [])),
        hints=json.dumps(conv.get('hints', []))
    )
    
    session.add(db_conv)
    session.commit()
    session.close()
    
    return '<span style="color: green;">✓ Saved!</span>'

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all saved conversations"""
    session = Session()
    convs = session.query(Conversation).order_by(Conversation.created_at.desc()).all()
    result = [{
        'id': c.id,
        'title': c.title,
        'language': c.language,
        'created_at': c.created_at.isoformat(),
        'message_count': len(json.loads(c.messages))
    } for c in convs]
    session.close()
    return jsonify({'conversations': result})

@app.route('/api/conversations/<int:conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """Get a specific conversation"""
    session = Session()
    conv = session.query(Conversation).filter_by(id=conv_id).first()
    if not conv:
        return jsonify({'error': 'Not found'}), 404
    
    result = {
        'id': conv.id,
        'title': conv.title,
        'language': conv.language,
        'created_at': conv.created_at.isoformat(),
        'messages': json.loads(conv.messages),
        'corrections': json.loads(conv.corrections) if conv.corrections else [],
        'hints': json.loads(conv.hints) if hasattr(conv, 'hints') and conv.hints else []
    }
    session.close()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

