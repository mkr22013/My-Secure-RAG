import sys
import os
import asyncio
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL = os.getenv("OLLAMA_MODEL")

async def get_ai_response(query):
    # This simplified version assumes you have 'server.py' 
    # already running in its own separate terminal
    try:
        # Instead of 'stdio_client', we just call the local logic 
        # (This is a temporary workaround to bypass the Windows pipe error)
        import ollama
        from server import query_insurance_benefits # Import the tool directly

        # Define the tool for Ollama
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
                    'required': ['year', 'plan_type', 'plan_tier', 'topic'] # CRITICAL
                }
            }
        }]

        # Create a strong System Prompt to override the default refusal
        system_prompt = {
            'role': 'system',
            'content': (
                "You are a specialized Corporate Insurance Assistant. "
                "You HAVE PERMISSION to provide specific plan details using the search tools provided. "
                "Always use the 'query_insurance_benefits' tool to look up facts from the company PDFs "
                "before answering. Do not give generic advice; give facts from the documents."
            )
        }
        # Ask Ollama to plan the search
        resp = ollama.chat(model=LOCAL_MODEL, messages=[system_prompt, {'role': 'user', 'content': query}], tools=tools)
        
        if resp.get('message', {}).get('tool_calls'):
            print("[*] SUCCESS: LLM is calling the tool!")
        else:
            print("[!] FAIL: LLM is ignoring the tool and giving a generic answer.")
            
        # If the AI wants to use the tool, we just call the Python function directly!
        if resp.get('message', {}).get('tool_calls'):
            msgs = [{'role': 'user', 'content': query}, resp['message']]
            for call in resp['message']['tool_calls']:
                # The arguments coming from the LLM
                args = call['function']['arguments']
                
                # CLEANING STEP: Remove any accidental keys the LLM might have added
                valid_keys = ['year', 'plan_type', 'plan_tier', 'topic']
                filtered_args = {k: v for k, v in args.items() if k in valid_keys}

                # CALL THE FUNCTION with only the 4 correct arguments
                result = query_insurance_benefits(**filtered_args)
                
                msgs.append({'role': 'tool', 'content': result, 'name': 'query_insurance_benefits'})
            
            final = ollama.chat(model=LOCAL_MODEL, messages=msgs)
            return final['message']['content']
        else:
            return resp['message']['content']
            
    except Exception as e:
        return f"DIRECT CALL ERROR: {str(e)}"