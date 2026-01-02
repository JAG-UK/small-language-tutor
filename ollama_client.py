import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="gemma2:27b"):
        self.base_url = base_url
        self.model = model
    
    def chat(self, messages, language="en"):
        """Send messages to Ollama and get response"""
        url = f"{self.base_url}/api/chat"
        
        # Format messages for Ollama API
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_model(self, model):
        """Change the model being used"""
        self.model = model

