import re
from ollama_client import OllamaClient

class GrammarChecker:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
    
    def check_message(self, user_message, conversation_context, target_language):
        """Check user message for errors and suggest corrections"""
        # Use the SLM to check grammar
        system_prompt = f"""You are a language tutor. Analyze the user's message for grammar, spelling, and naturalness in {target_language}. 
        If there are errors, provide:
        1. The corrected version
        2. Brief explanation of the mistakes
        
        Format as JSON: {{"has_errors": true/false, "corrected": "...", "explanation": "..."}}
        If no errors, return has_errors: false."""
        
        check_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Check this message: {user_message}"}
        ]
        
        try:
            response = self.ollama.chat(check_messages, target_language)
            # Try to parse JSON from response
            import json
            # Extract JSON if wrapped in markdown
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            return {"has_errors": False, "corrected": user_message, "explanation": "No errors detected"}
        except:
            return {"has_errors": False, "corrected": user_message, "explanation": "Could not check grammar"}

