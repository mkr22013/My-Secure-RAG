import sys, os, asyncio, ollama, json, re
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL = os.getenv("OLLAMA_MODEL")

# --- GLOBAL CACHE ---
TOOL_RESULT_CACHE = {}

async def get_ai_response(query):
    try:
        from server import query_insurance_benefits 

        # --- STEP 0: THE FAST-PATH (PRE-LLM CACHE CHECK) ---
        q_lower = query.lower()
        
        # 1. Extract potential parameters directly from the user's text
        found_years = re.findall(r'202\d', q_lower) # Finds 2024, 2025
        found_tier = re.search(r'(gold|silver|bronze)', q_lower)
        found_type = re.search(r'(medical|dental|vision)', q_lower)

        # 2. Check if we have enough info to try a Cache Lookup
        if found_years and found_tier and found_type:
            cache_results = []
            
            for y in found_years:
                key = f"{y}_{found_type.group()}_{found_tier.group()}".replace(" ", "")
                if key in TOOL_RESULT_CACHE:
                    cache_results.append(TOOL_RESULT_CACHE[key])
            
            # 3. If ALL requested years are in cache, skip the LLM Planning Turn!
            if len(cache_results) == len(found_years):
                print(f"[*] FAST-PATH HIT: Bypassing LLM planning for {found_years}")
                
                # Define a minimal system prompt for the synthesis
                fast_system_prompt = {'role': 'system', 'content': "You are an insurance expert. Use the provided tool results to answer the user."}
                
                msgs = [fast_system_prompt, {'role': 'user', 'content': query}]
                for res in cache_results:
                    msgs.append({'role': 'tool', 'content': str(res), 'name': 'query_insurance_benefits'})
                
                # Jump straight to final synthesis (Saves ~20 seconds)
                final = ollama.chat(model=LOCAL_MODEL, messages=msgs)
                return final['message']['content']

        # --- STEP 1: THE SLOW-PATH (Standard Agent Logic) ---
        # (This only runs if the Fast-Path didn't have all the data)

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

        system_prompt = {
            'role': 'system',
            'content': (
                "You are a specialized Corporate Insurance Assistant. "
                "INSTRUCTION: When asked to compare different years (e.g., 2024 and 2025), "
                "you MUST generate a SEPARATE tool call for EACH year mentioned. "
                "Example: If comparing 2024 vs 2025 Gold Medical, you must call "
                "'query_insurance_benefits' twice: once for 2024 and once for 2025. "
                "Never combine them into one call. Never answer from memory."
            )
        }   

        # FIRST TURN: Planning
        resp = ollama.chat(
            model=LOCAL_MODEL, 
            messages=[system_prompt, {'role': 'user', 'content': query}], 
            tools=tools,
            options={"num_ctx": 4096, "temperature": 0, "num_predict": 100}
        )
        
        if resp.get('message', {}).get('tool_calls'):
            print(f"[*] SUCCESS: LLM is planning {len(resp['message']['tool_calls'])} tool calls.")
            msgs = [system_prompt, {'role': 'user', 'content': query}, resp['message']]
            
            for call in resp['message']['tool_calls']:
                args = call['function']['arguments']
                c_year, c_type, c_tier = str(args.get('year')).strip(), str(args.get('plan_type')).lower().strip(), str(args.get('plan_tier')).lower().strip()
                plan_cache_key = f"{c_year}_{c_type}_{c_tier}".replace(" ", "")
                
                if plan_cache_key in TOOL_RESULT_CACHE:
                    print(f"[*] CACHE HIT: Reusing data for {plan_cache_key}")
                    result = TOOL_RESULT_CACHE[plan_cache_key]
                else:
                    print(f"[*] CACHE MISS: Reading PDF for {plan_cache_key}")
                    result = query_insurance_benefits(**{k: v for k, v in args.items() if k in ['year', 'plan_type', 'plan_tier', 'topic']})
                    if "SOURCE:" in str(result):
                        TOOL_RESULT_CACHE[plan_cache_key] = result

                msgs.append({'role': 'tool', 'content': str(result), 'name': 'query_insurance_benefits'})
            
            # SECOND TURN: Synthesis
            final_response = ollama.chat(model=LOCAL_MODEL, messages=msgs)
            return final_response['message']['content']
        
        else:
            return resp['message']['content']
            
    except Exception as e:
        return f"DIRECT CALL ERROR: {str(e)}"
