import os
import json
import requests
import textwrap
from typing import List, Dict, Any
from openai import OpenAI

# 1. Configuration from Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN") # NO DEFAULT for token per requirement
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70b-instruct") # Default model
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860") # Local or Space URL

def run_agent():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is missing.")
        return

    # OpenAI compatible client initialization
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    print(f"[START] Initializing Episode for Multi-Stock Environment")
    print(f"Agent: {MODEL_NAME} | Endpoint: {API_BASE_URL}")

    # Reset Environment
    try:
        resp = requests.post(f"{ENV_URL}/reset") # Using POST per validator script
        obs = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to connect to environment at {ENV_URL}: {e}")
        return

    done = False
    step_count = 0
    max_steps = 30 # Matching environment config

    while not done and step_count < max_steps:
        step_count += 1
        
        # Build prompt from observation
        market_summary = ""
        if "market_data" in obs:
            market_summary = "\n".join([f"- {t}: ${p['price']} (Trend: {p['trend_signal']})" 
                                       for t, p in obs['market_data'].items()])
        
        prompt = textwrap.dedent(f"""
            Your goal is to manage a stock portfolio. Current portfolio value: ${obs.get('portfolio_value', 'unknown')}.
            Market Data:
            {market_summary}
            
            Current Holdings: {obs.get('holdings', {{}})}
            
            Task: Provide allocation percentages for AAPL, MSFT, TSLA, GME, AMC. 
            The sum of absolute allocations must be <= 1.0. 
            Format your response as a JSON object: {{"allocations": {{"TICKER": 0.X, ...}}}}
        """)

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={ "type": "json_object" }
            )
            action_raw = completion.choices[0].message.content
            action = json.loads(action_raw)
        except Exception as e:
            # Fallback per bootcamp recommendation to ensure inference doesn't crash
            print(f"  [FALLBACK] LLM Error: {e}. Using safety allocation.")
            action = {"allocations": {"AAPL": 0.1, "MSFT": 0.1}}

        # Execute Step
        print(f"[STEP {step_count}] Action: {action.get('allocations', {})}")
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action)
            step_data = step_resp.json()
            
            obs = step_data["observation"]
            reward = step_data["reward"]["value"]
            done = step_data["done"]
            
            # Print state details to stdout for grader visibility
            print(f"  Result: Value=${obs['portfolio_value']} | Reward={reward:.4f}")
        except Exception as e:
            print(f"  [ERROR] Connection lost during step: {e}")
            break

    print(f"[END] Final Portfolio Value: ${obs['portfolio_value']}")

if __name__ == "__main__":
    run_agent()
