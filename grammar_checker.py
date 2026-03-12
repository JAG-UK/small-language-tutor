import re
from ollama_client import OllamaClient

class GrammarChecker:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
    
    def _parse_json_response(self, response, manual_extractor=None):
        """
        Parse JSON from LLM response using multiple strategies.
        
        Args:
            response: The raw response string from the LLM
            manual_extractor: Optional function to manually extract fields when JSON is malformed.
                            Should return a dict or None.
        
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        import json
        
        # Strategy 1: Look for JSON code blocks
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except:
                pass
        
        # Strategy 2: Look for JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                return json.loads(json_str)
            except:
                # If that fails, try to find a better JSON match
                json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                matches = re.finditer(json_pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match.group(0))
                    except:
                        continue
        
        # Strategy 3: Manual field extraction (if provided)
        if manual_extractor:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            return manual_extractor(cleaned)
        
        return None
    
    def check_message(self, user_message, conversation_context, target_language, tone='friendly'):
        """Check user message for errors and suggest corrections"""
        # Use the SLM to check grammar
        tone_instructions = {
            'friendly': 'casual, informal, warm',
            'professional': 'formal, polite, business-like',
            'flirty': 'playful, charming, slightly suggestive'
        }
        tone_desc = tone_instructions.get(tone, 'casual, informal')
        
        system_prompt = f"""You are a precise language tutor. Analyze the user's message for grammar, spelling, naturalness, and tone appropriateness in {target_language}. The conversation tone is {tone} ({tone_desc}). PRESERVE THE USER'S INTENT: Do not change statements to questions, questions to statements, or alter the sentence structure. Only fix grammar, spelling, punctuation, and word errors.

CRITICAL INSTRUCTIONS:
1. First, create the corrected version of the message
2. Then, CAREFULLY compare the ORIGINAL and CORRECTED versions side-by-side
3. In your explanation, ONLY describe the ACTUAL differences you made between original and corrected
4. Do NOT convert statements to questions or vice versa unless there is a clear grammatical error requiring it. For example:
    - If the user writes 'Estoy buscando...' (statement), do NOT change it to '¿Estás buscando...?' (question) - preserve the statement form.
5. Only correct actual errors. Do not try to 'improve' or restructure the message.
6. Be precise and accurate when describing the differences. For example:
   - If the change was a simple typo, say "typo in '...' should be '...'" NOT "should be '...'"
   - If the change was a missing accent, say "missing accent on '...' should be '...'" NOT "should be '...'"
   - If the change was a mis-placed or extra accent, say "mis-placed accent on '...' should be '...'" NOT "should be '...'"
7. Only mention changes that actually exist - verify each point by comparing original vs corrected
8. Be specific: name the exact letters, accents, punctuation marks, or words that changed
9. Check if the tone matches the conversation style ({tone}): if the user uses overly formal language in a flirty conversation, or vice versa, suggest appropriate tone adjustments

Format as JSON: {{"has_errors": true/false, "corrected": "...", "explanation": "..."}}
The explanation should list ONLY the actual differences, numbered if multiple. Double-check each explanation against the original and corrected versions.
If no errors, return has_errors: false."""
        
        check_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Correct this message: {user_message}"}
        ]
        
        def extract_correction_fields(cleaned):
            """Manual extraction for correction fields"""
            has_errors_match = re.search(r'"has_errors"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
            corrected_match = re.search(r'"corrected"\s*:\s*"([^"]*)"', cleaned)
            
            # Explanation might be unquoted or have formatting issues
            explanation_match = None
            
            # Try pattern 1: Quoted explanation on same line
            explanation_match = re.search(
                r'"explanation"\s*:\s*"([^"]*)"',
                cleaned,
                re.DOTALL
            )
            
            # Try pattern 2: Unquoted explanation (could be multi-line)
            if not explanation_match:
                explanation_match = re.search(
                    r'"explanation"\s*:\s*(.+?)(?=\s*[},]\s*$)',
                    cleaned,
                    re.DOTALL | re.MULTILINE
                )
                
                if not explanation_match:
                    explanation_match = re.search(
                        r'"explanation"\s*:\s*(.+?)(?=\s*\n\s*})',
                        cleaned,
                        re.DOTALL
                    )
            
            if has_errors_match and corrected_match:
                has_errors = has_errors_match.group(1).lower() == 'true'
                corrected = corrected_match.group(1)
                
                # Extract explanation
                explanation = "No explanation provided"
                if explanation_match:
                    if explanation_match.lastindex >= 1 and explanation_match.group(1):
                        explanation = explanation_match.group(1)
                    elif explanation_match.lastindex >= 1:
                        explanation = explanation_match.group(1).strip()
                        explanation = re.sub(r'^\d+\.\s*', '', explanation, flags=re.MULTILINE)
                        explanation = re.sub(r'\n\s*\d+\.\s*', '\n', explanation)
                        explanation = explanation.strip().strip('"').strip("'")
                        explanation = re.sub(r'[ \t]+', ' ', explanation)
                        explanation = re.sub(r'\n\s*\n', '\n', explanation)
                
                result = {
                    "has_errors": has_errors,
                    "corrected": corrected,
                    "explanation": explanation
                }
                print(f"DEBUG Grammar Checker: Manually extracted result: {result}")
                return result
            return None
        
        try:
            response = self.ollama.chat(check_messages, target_language)
            result = self._parse_json_response(response, extract_correction_fields)
            
            if result:
                # Ensure has_errors is a boolean
                if 'has_errors' in result:
                    if isinstance(result['has_errors'], str):
                        result['has_errors'] = result['has_errors'].lower() in ('true', '1', 'yes')
                    else:
                        result['has_errors'] = bool(result['has_errors'])
                print(f"DEBUG Grammar Checker: Parsed result: {result}")
                return result
            
            print(f"DEBUG Grammar Checker: No valid JSON found in response: {response[:200]}")
            return {"has_errors": False, "corrected": user_message, "explanation": "No errors detected"}
        except Exception as e:
            print(f"DEBUG Grammar Checker: Exception occurred: {e}")
            print(f"DEBUG Grammar Checker: Response was: {response[:500] if 'response' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
            return {"has_errors": False, "corrected": user_message, "explanation": "Could not check grammar"}
    
    def get_hints(self, user_message, conversation_context, target_language, tone='friendly'):
        """Get hints and tips to make language more natural, even if grammatically correct"""
        tone_instructions = {
            'friendly': 'casual, informal, warm',
            'professional': 'formal, polite, business-like',
            'flirty': 'playful, charming, slightly suggestive'
        }
        tone_desc = tone_instructions.get(tone, 'casual, informal')
        
        system_prompt = f"""You are a helpful language tutor providing tips to make {target_language} more natural and idiomatic. You are watching a conversation the user is having with a native {target_language} speaker and providing real-time feedback to help the user improve next time. The conversation tone is {tone} ({tone_desc}).

CRITICAL RULES:
1. Your explanations must be in English (so the learner understands), BUT
2. ALL suggested phrases, alternatives, and examples MUST be in {target_language} only
3. NEVER suggest English phrases as alternatives - always suggest {target_language} phrases
4. Keep explanations brief and concise, but long enough to be helpful.
5. Consider the {tone} tone: suggest phrases that match the conversation style

Analyze the user's message and provide helpful hints:
1. If the language is very good or natural, praise it specifically (in English)
2. If they use basic vocabulary, suggest more advanced or nuanced alternatives IN {target_language} that fit the {tone} tone
3. If they use formulaic or textbook phrases, suggest more idiomatic or natural expressions IN {target_language} appropriate for a {tone} conversation
4. If the tone doesn't match (e.g., too formal for friendly, too casual for professional), suggest tone-appropriate alternatives IN {target_language}
5. Focus on making the language sound more native-like and tone-appropriate, even if it's grammatically correct

Example of CORRECT format:
- "Instead of 'No todavia', it would be more natural to say 'Aún no' or 'Todavía no'." (suggestions in {target_language})
- "Consider replacing 'Espero a la conferencia' with 'Estoy esperando la conferencia'." (suggestions in {target_language})

Example of WRONG format:
- "Instead of 'No todavia', try 'Not yet'." (NEVER suggest English!)
- "Aún no" (ALWAYS provide a short explanation for the hint!)

Be encouraging and constructive. Format as JSON: {{"has_hints": true/false, "hints": ["hint 1", "hint 2", ...]}}
If the language is already excellent and natural, return has_hints: false.
Keep hints brief, clear, and actionable."""
        
        check_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Provide hints for this message: {user_message}"}
        ]
        
        def extract_hints_fields(cleaned):
            """Manual extraction for hints fields"""
            has_hints_match = re.search(r'"has_hints"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
            
            # Try to extract hints array - could be formatted in various ways
            hints_list = []
            
            # Pattern 1: Standard JSON array format
            hints_array_match = re.search(r'"hints"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
            if hints_array_match:
                hints_str = hints_array_match.group(1)
                hint_matches = re.findall(r'"([^"]*)"', hints_str)
                hints_list = hint_matches
            
            # Pattern 2: If hints are on separate lines or formatted differently
            if not hints_list:
                hints_section = re.search(r'"hints"\s*:\s*\[?(.*?)(?:\]|})', cleaned, re.DOTALL)
                if hints_section:
                    hints_content = hints_section.group(1)
                    hint_matches = re.findall(r'"([^"]*)"', hints_content)
                    if hint_matches:
                        hints_list = hint_matches
                    else:
                        hints_list = [h.strip().strip('"') for h in re.split(r'[,\n]', hints_content) if h.strip()]
            
            if has_hints_match:
                has_hints = has_hints_match.group(1).lower() == 'true'
                result = {
                    "has_hints": has_hints,
                    "hints": hints_list if hints_list else []
                }
                print(f"DEBUG Hints Checker: Manually extracted result: {result}")
                return result
            return None
        
        try:
            response = self.ollama.chat(check_messages, target_language)
            result = self._parse_json_response(response, extract_hints_fields)
            
            if result:
                return result
            return {"has_hints": False, "hints": []}
        except Exception as e:
            print(f"DEBUG Hints Checker: Exception occurred: {e}")
            return {"has_hints": False, "hints": []}

