import re
from ollama_client import OllamaClient

class GrammarChecker:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
    
    def check_message(self, user_message, conversation_context, target_language):
        """Check user message for errors and suggest corrections"""
        # Use the SLM to check grammar
        system_prompt = f"""You are a precise language tutor. Analyze the user's message for grammar, spelling, and naturalness in {target_language}. 

CRITICAL INSTRUCTIONS:
1. First, create the corrected version of the message
2. Then, CAREFULLY compare the ORIGINAL and CORRECTED versions side-by-side
3. In your explanation, ONLY describe the ACTUAL differences you made between original and corrected
4. Be precise and accurate. For example:
   - If "Si" becomes "Sí", say "missing accent on 'i'" NOT "should be capitalized" (it's already capitalized)
   - If a question is missing "¿" at the start, say "missing inverted question mark (¿) at the beginning" NOT "missing question mark at the end"
   - If "razónes" becomes "razones", say "remove accent from 'o' in 'razones'" NOT just "should be razones"
5. For Spanish: questions need inverted question marks (¿) at the BEGINNING and (?) at the end
6. Only mention changes that actually exist - verify each point by comparing original vs corrected
7. Be specific: name the exact letters, accents, punctuation marks, or words that changed

Format as JSON: {{"has_errors": true/false, "corrected": "...", "explanation": "..."}}
The explanation should list ONLY the actual differences, numbered if multiple. Double-check each explanation against the original and corrected versions.
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
    
    def get_hints(self, user_message, conversation_context, target_language):
        """Get hints and tips to make language more natural, even if grammatically correct"""
        system_prompt = f"""You are a helpful language tutor providing tips to make {target_language} more natural and idiomatic.

CRITICAL RULES:
1. Your explanations can be in English (so the learner understands), BUT
2. ALL suggested phrases, alternatives, and examples MUST be in {target_language} only
3. NEVER suggest English phrases as alternatives - always suggest {target_language} phrases
4. When giving examples, provide the {target_language} phrase, not an English translation

Analyze the user's message and provide helpful hints:
1. If the language is very good or natural, praise it specifically (in English)
2. If they use basic vocabulary, suggest more advanced or nuanced alternatives IN {target_language}
3. If they use formulaic or textbook phrases, suggest more idiomatic or natural expressions IN {target_language}
4. Focus on making the language sound more native-like, even if it's grammatically correct

Example of CORRECT format:
- "Instead of 'No todavia', try 'Aún no' or 'Todavía no'." (suggestions in {target_language})
- "Consider replacing 'Espero a la conferencia' with 'Estoy esperando la conferencia'." (suggestions in {target_language})

Example of WRONG format:
- "Instead of 'No todavia', try 'Not yet'." (NEVER suggest English!)

Be encouraging and constructive. Format as JSON: {{"has_hints": true/false, "hints": ["hint 1", "hint 2", ...]}}
If the language is already excellent and natural, return has_hints: false.
Keep hints brief and actionable."""
        
        check_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Provide hints for this message: {user_message}"}
        ]
        
        try:
            response = self.ollama.chat(check_messages, target_language)
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            return {"has_hints": False, "hints": []}
        except:
            return {"has_hints": False, "hints": []}

