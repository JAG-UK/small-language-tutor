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
   
   # In another terminal, pull a model:
   ollama pull gemma2:27b
   # Or try other models: qwen2.5, mistral, llama3.2, etc.
   ```

3. **Configure the model** (optional):
   Edit `app.py` to change the default model:
   ```python
   ollama = OllamaClient(model="your-model-name")
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   Navigate to `http://localhost:5001`
   
   Note: Port 5001 is used instead of 5000 to avoid conflicts with macOS AirPlay Receiver

## Architecture

- **Backend**: Flask (Python)
- **Frontend**: HTMX + vanilla CSS
- **Database**: SQLite
- **SLM**: Ollama API integration
- **Future**: Architecture supports voice conversations (to be implemented)

## Project Structure

```
small-language-tutor/
├── app.py                 # Flask backend server
├── models.py              # Database models
├── ollama_client.py       # SLM integration wrapper
├── grammar_checker.py     # Grammar/correction logic
├── static/
│   └── css/
│       └── style.css     # Main stylesheet
├── templates/
│   └── index.html        # Main HTMX interface
└── requirements.txt       # Python dependencies
```
