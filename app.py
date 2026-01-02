from flask import Flask, render_template, request, jsonify
from models import Session, Conversation
from ollama_client import OllamaClient
from grammar_checker import GrammarChecker
from datetime import datetime
import json
import html

app = Flask(__name__)

# Initialize clients
ollama = OllamaClient(model="gemma2:27b")  # Configurable
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
    
    # Get or create conversation
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'language': language,
            'corrections': []
        }
    
    conv = conversations[session_id]
    
    # Add user message
    user_msg = {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()}
    conv['messages'].append(user_msg)
    
    # Check grammar
    correction = grammar_checker.check_message(user_message, conv['messages'], language)
    if correction.get('has_errors'):
        conv['corrections'].append({
            'message': user_message,
            'corrected': correction.get('corrected'),
            'explanation': correction.get('explanation'),
            'timestamp': datetime.now().isoformat()
        })
    
    # Get AI response
    system_prompt = f"""You are a friendly language tutor helping someone learn {language}. 
    Keep responses natural, conversational, and appropriate for a language learner. 
    Keep responses brief (2-3 sentences max)."""
    
    messages_for_llm = [
        {"role": "system", "content": system_prompt}
    ] + [{"role": msg["role"], "content": msg["content"]} for msg in conv['messages'][-10:]]  # Last 10 messages for context
    
    ai_response = ollama.chat(messages_for_llm, language)
    ai_msg = {"role": "assistant", "content": ai_response, "timestamp": datetime.now().isoformat()}
    conv['messages'].append(ai_msg)
    
    return jsonify({
        'response': ai_response,
        'correction': correction if correction.get('has_errors') else None,
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

@app.route('/api/corrections', methods=['GET'])
def get_corrections():
    """Get corrections for current conversation"""
    session_id = request.args.get('session_id', 'default')
    corrections = []
    if session_id in conversations:
        corrections = conversations[session_id]['corrections']
    
    if not corrections:
        return '<p class="empty-state">No corrections yet. Keep practicing!</p>'
    
    # Render corrections as HTML
    html_output = ''
    for correction in reversed(corrections):  # Show most recent first
        original = correction.get('message', '')
        corrected = correction.get('corrected', '')
        explanation = correction.get('explanation', '')
        
        # Escape HTML to prevent XSS
        original = html.escape(original)
        corrected = html.escape(corrected)
        explanation = html.escape(explanation)
        
        html_output += f'''
        <div class="correction-item">
            <div class="original">{original}</div>
            <div class="corrected">✓ {corrected}</div>
            <div class="explanation">{explanation}</div>
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
        corrections=json.dumps(conv['corrections'])
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
        'corrections': json.loads(conv.corrections)
    }
    session.close()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

