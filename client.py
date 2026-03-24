import os, ollama, json, re
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- GLOBAL RAM CACHE (Persistent for the session) ---
TOOL_RESULT_CACHE = {}

def flatten_message_content(content):
    """
    NUCLEAR NORMALIZER: Forces any Ollama response (List, Dict, or None) 
    into a plain string to prevent Gradio/Streamlit/Pydantic validation errors.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get('text', str(item)))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()

async def get_ai_response(query, history):
    try:
        from server import query_insurance_benefits, get_available_plans 
        
        # --- 1. CONTEXT MERGING (MEMORY) ---
        recent_history = " ".join([flatten_message_content(m['content']) for m in history[-3:]])
        full_context_query = f"{recent_history} {query}".lower()

        # --- 2. EXTRACTION FROM MERGED CONTEXT ---
        found_years = re.findall(r'202\d', full_context_query)
        p_tier_match = re.search(r'(gold|silver|bronze)', full_context_query)
        p_tier_fast = p_tier_match.group() if p_tier_match else "gold"
        
        p_type_fast = "medical"
        if "dental" in full_context_query: p_type_fast = "dental"
        elif "vision" in full_context_query: p_type_fast = "vision"

        # --- 3. TURBO CACHE HIT CHECK ---
        cached_data_fragments = []
        if found_years:
            for y in found_years:
                key = f"{y}_{p_type_fast}_{p_tier_fast}".replace(" ", "").lower()
                if key in TOOL_RESULT_CACHE:
                    cached_data_fragments.append(TOOL_RESULT_CACHE[key])

        # If we have ALL requested years in cache, skip the LLM loop!
        if len(cached_data_fragments) == len(found_years) and len(found_years) > 0:
            print(f"[*] TURBO CACHE HIT: Bypassing reasoning for {found_years}")
            fast_msgs = [
                {'role': 'system', 'content': 'You are an insurance expert. Synthesize this cached data accurately.'},
                {'role': 'user', 'content': f"Synthesize this data: {str(cached_data_fragments)} for query: {query}"}
            ]
            final_synth = ollama.chat(model=LOCAL_MODEL, messages=fast_msgs)
            return flatten_message_content(final_synth['message'].get('content', ''))

        # --- 4. INITIAL DISCOVERY CHECK ---
        called_discovery = True if not found_years else False
            
        # --- 5. THE REASONING LOOP ---
        messages = []
        system_prompt = {
            'role': 'system',
            'content': (
                "You are a specialized Insurance Assistant. Goal: 100% Accuracy."
                "\n\nSTRICT TOOL RULES:"
                "\n1. VAGUE: If Year, Type, or Tier is missing, call 'get_available_plans' first."
                "\n2. MULTI-YEAR: If 2024 and 2025 are mentioned, you MUST call 'query_insurance_benefits' for BOTH."
                "\n3. NO QUESTIONS: If you have a Year and Tier, DO NOT ask permission. CALL THE TOOL."
                "\n4. SILENCE: Provide ONLY JSON tool blocks during planning. No conversational filler."
            )
        }
        messages.append(system_prompt)

        if history:
            for turn in history[-2:]:
                messages.append({'role': turn.get('role', 'user'), 'content': flatten_message_content(turn.get('content', ''))})

        messages.append({'role': 'user', 'content': query})

        tools = [
		{'type': 'function', 'function': {'name': 'get_available_plans', 'description': 'PROBE DB index for available years/tiers'}},
		{
			'type': 'function', 
			'function': {
				'name': 'query_insurance_benefits',
				'description': 'Retrieve benefit text from indexed PDFs.',
				'parameters': {
					'type': 'object',
					'properties': {
						'year': {'type': 'integer'},
						'plan_type': {'type': 'string'},
						'plan_tier': {'type': 'string'},
						'topic': {'type': 'string'}
					},
					"required": ["year", "plan_type", "plan_tier", "topic"]
				}
			}
		}
	]

        turn_count = 0
        final_raw_context = "" # To store data for the cache later

        while turn_count < 3:
            turn_count += 1
            resp = ollama.chat(model=LOCAL_MODEL, messages=messages, tools=tools, options={"temperature": 0})
            
            raw_msg = resp['message']
            clean_content = flatten_message_content(raw_msg.get('content', ''))
            tool_calls = raw_msg.get('tool_calls', [])

            # Update history with the assistant's action (JSON or Text)
            messages.append({'role': 'assistant', 'content': clean_content, 'tool_calls': tool_calls if tool_calls else None})
            
            # --- EXIT CASE: AI finished its search ---
            if not tool_calls:
                # If we already have a long text answer, return it immediately
                if clean_content and len(clean_content.strip()) > 10:
                    return clean_content
                break # Exit the loop to trigger the "Final Safety Net" below

            print(f"[*] TURN {turn_count}: Processing {len(tool_calls)} tool calls...")
            for call in tool_calls:
                func_name = call['function']['name']
                args = call['function']['arguments']
                
                if func_name == "get_available_plans":
                    called_discovery = True
                    result = get_available_plans()
                    messages.append({'role': 'tool', 'content': str(result), 'name': func_name})
                else:
                    # SCAVENGER logic for years
                    all_args_blob = str(args).lower()
                    years_to_process = re.findall(r'202\d', all_args_blob) or ["2024"]
                    
                    results_list = []
                    for p_year in years_to_process:
                        p_type = next((t for t in ["medical", "dental", "vision"] if t in all_args_blob), "medical")
                        p_tier = next((t for t in ["gold", "silver", "bronze"] if t in all_args_blob), "gold")
                        raw_topic = str(args.get('topic', 'deductible')).lower()
                        p_topic = "deductible" if any(w in raw_topic for w in ["change", "diff", "benefit"]) else raw_topic

                        data = query_insurance_benefits(year=int(p_year), plan_type=p_type, plan_tier=p_tier.capitalize(), topic=p_topic)
                        results_list.append(data)
                        
                        # --- CACHE STORAGE ---
                        cache_key = f"{p_year}_{p_type}_{p_tier}".replace(" ", "").lower()
                        TOOL_RESULT_CACHE[cache_key] = data

                    final_raw_context = "\n\n".join(results_list)
                    messages.append({'role': 'tool', 'content': final_raw_context, 'name': func_name})

        # --- THE FINAL SAFETY NET (DYNAMIC SYNTHESIS) ---
        print("[*] TRIGGERING FINAL DYNAMIC SYNTHESIS...")
        
        # 1. Strip technical system instructions to allow natural speech
        final_messages = [m for m in messages if m.get('role') != 'system']
        
        # 2. Build a dynamic instruction based on the actual request
        target_label = f"{'/'.join(found_years)} {p_tier_fast} {p_type_fast}"
        
        # --- THE FINAL SAFETY NET (DYNAMIC SYNTHESIS) ---
        if final_raw_context:
            instruction = (
                f"You have retrieved insurance details for {target_label}. "
                "Synthesize this into a plain-English response. "
                "1. Create a Markdown Table. "
                "2. CONTEXT ISOLATION: ONLY use data from the MOST RECENT tool results. "
                "Do NOT carry over numbers (like $25 or $35) from previous Medical conversations into a Dental response. "
                "3. STRICT TRUTH: If a value is not EXPLICITLY written in the dental text, put 'Not Listed'. "
                "4. YOU ARE THE DOCUMENT READER: Provide the specific Ortho or Dental values found. "
                "5. Do NOT use JSON."
            )
        else:
            # UPDATED DISCOVERY INSTRUCTION: Prioritize the Plan Type over History
            current_schema = get_available_plans()
            instruction = (
                f"CONTEXT: The database contains: {current_schema}. "
                f"The user is asking about: '{query}'. "
                "1. INTENT RESET: If the user mentions a new Plan Type (e.g., 'Dental'), "
                "IGNORE previous Tiers (like 'Gold') or Years from the history. "
                "2. Look for ANY plan matching the new Plan Type in the schema. "
                "3. If a match is found (e.g., Silver Dental 2025), tell them: "
                "'I found a [Year] [Tier] [Type] plan. Would you like me to pull those benefits?' "
                "4. Do NOT say a plan doesn't exist just because the Tier doesn't match history. "
                "5. Speak naturally. No JSON."
            )

        final_messages.insert(0, {'role': 'system', 'content': instruction})

        # 3. Final call to get text for the UI
        final_resp = ollama.chat(model=LOCAL_MODEL, messages=final_messages, options={"temperature": 0.7})
        return flatten_message_content(final_resp['message'].get('content', ''))

    except Exception as e:
        return f"⚠️ System Error: {str(e)}"
