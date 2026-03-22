import sys
import os
import asyncio
import ollama
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL = os.getenv("OLLAMA_MODEL")

# --- GLOBAL CACHE ---
TOOL_RESULT_CACHE = {}

async def get_ai_response(query):
    try:
        from server import query_insurance_benefits # Import the tool directly

        # 1. DEFINE THE TOOL
        tools = [{
            'type': 'function',
            'function': {
                'name': 'query_insurance_benefits',
                'description': 'Retrieves specific benefit details from local insurance PDFs.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'year': {'type': 'integer', 'description': 'The plan year (e.g. 2024)'},
                        'plan_type': {'type': 'string', 'description': 'Medical, Dental, or Vision'},
                        'plan_tier': {'type': 'string', 'description': 'Gold, Silver, or Bronze'},
                        'topic': {'type': 'string', 'description': 'Specific benefit (e.g. deductible)'}
                    },
                    'required': ['year', 'plan_type', 'plan_tier', 'topic']
                }
            }
        }]

        # 2. SYSTEM PROMPT (Ensures AI uses the provided data)
        system_prompt = {
            'role': 'system',
            'content': (
                "You are a specialized Corporate Insurance Assistant. "
                "IMPORTANT: When a user asks to compare plans or asks about multiple years, "
                "you MUST generate separate tool calls for EACH year and EACH plan mentioned. "
                "Do not skip any years. For example, if asked about 2024 vs 2025, you must "
                "call 'query_insurance_benefits' for 2024 AND for 2025."
            )
        }

        # 3. FIRST TURN: Ask Ollama to plan the search
        resp = ollama.chat(model=LOCAL_MODEL, messages=[system_prompt, {'role': 'user', 'content': query}], tools=tools)
        
        if resp.get('message', {}).get('tool_calls'):
            print(f"[*] SUCCESS: LLM is planning {len(resp['message']['tool_calls'])} tool calls.")
            
            # Start tracking message history for the second turn
            msgs = [system_prompt, {'role': 'user', 'content': query}, resp['message']]
            
            for call in resp['message']['tool_calls']:
                args = call['function']['arguments']
                
                # --- STEP 1: WIDE CACHE KEY (Year + Plan + Tier ONLY) ---
                # We ignore 'topic' because the PDF page usually covers the whole plan.
                c_year = str(args.get('year', '')).strip()
                c_type = str(args.get('plan_type', '')).lower().strip()
                c_tier = str(args.get('plan_tier', '')).lower().strip()
                plan_cache_key = f"{c_year}_{c_type}_{c_tier}".replace(" ", "")
                
                if plan_cache_key in TOOL_RESULT_CACHE:
                    print(f"[*] CACHE HIT: Reusing cached data for {plan_cache_key}")
                    result = TOOL_RESULT_CACHE[plan_cache_key]
                else:
                    print(f"[*] CACHE MISS: Reading PDF for {plan_cache_key}")
                    valid_keys = ['year', 'plan_type', 'plan_tier', 'topic']
                    filtered_args = {k: v for k, v in args.items() if k in valid_keys}
                    
                    # CALL THE TOOL
                    result = query_insurance_benefits(**filtered_args)
                    
                    # ONLY CACHE IF SUCCESSFUL (Must contain 'SOURCE:')
                    if "SOURCE:" in str(result):
                        TOOL_RESULT_CACHE[plan_cache_key] = result
                    else:
                        print(f"[!] REJECTION: Tool returned an error for {plan_cache_key}")

                # --- STEP 2: ADD TO HISTORY ---
                msgs.append({
                    'role': 'tool', 
                    'content': str(result), 
                    'name': 'query_insurance_benefits'
                })
            
            # --- DEBUG: Verify what we are sending back to the AI ---
            print(f"[*] FINAL DATA SENT TO AI: {len(msgs)} messages in history.")
            for m in msgs:
                if m['role'] == 'tool':
                    print(f"    -> Tool Result: {m['content'][:50]}...") 

            # 4. THE SECOND TURN (Synthesis)
            final_response = ollama.chat(model=LOCAL_MODEL, messages=msgs)
            return final_response['message']['content']
        
        else:
            print("[!] LLM provided a direct answer without tools.")
            return resp['message']['content']
            
    except Exception as e:
        return f"DIRECT CALL ERROR: {str(e)}"