import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="translategemma:12b", timeout=120):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout  # Timeout in seconds (default 2 minutes for larger models)
    
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
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.Timeout:
            return f"Error: Request timed out after {self.timeout} seconds. The model may need more time to respond."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_model(self, model):
        """Change the model being used"""
        self.model = model

