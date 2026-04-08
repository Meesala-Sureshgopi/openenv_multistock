import os
import json
import requests
import textwrap
from typing import List, Dict, Any
from openai import OpenAI

# 1. Configuration from Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70b-instruct")

# Environment vars per specification
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "single_stock_profit")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "openenv_multistock")

def run_agent():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is missing.")
        return

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # EXACT MANDATORY START FORMAT
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        resp = requests.post(f"{ENV_URL}/reset")
        obs = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to connect to env: {e}")
        return

    done = False
    step_count = 0
    max_steps = 30
    rewards = []
    error_msg = "null"
    
    trajectory = []

    while not done and step_count < max_steps:
        step_count += 1
        
        market_summary = ""
        if "market_data" in obs:
            market_summary = "\n".join([f"- {t}: ${p['price']} (Trend: {p['trend_signal']})" 
                                       for t, p in obs['market_data'].items()])
        current_holdings = obs.get('holdings', {})
        portfolio_val = obs.get('portfolio_value', 'unknown')

        prompt = textwrap.dedent(f"""
            Your goal is to manage a stock portfolio. Current portfolio value: ${portfolio_val}.
            Market Data:
            {market_summary}
            
            Current Holdings: {current_holdings}
            
            Task: Provide allocation percentages for AAPL, MSFT, TSLA, GME, AMC. 
            The sum of absolute allocations must be <= 1.0. 
            Format your response as a JSON object: {{"allocations": {{"TICKER": 0.X, ...}}}}
        """)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={ "type": "json_object" }
            )
            action_raw = completion.choices[0].message.content
            action = json.loads(action_raw)
            error_msg = "null"
        except Exception as e:
            action = {"allocations": {"AAPL": 0.1, "MSFT": 0.1}}
            action_raw = json.dumps(action)
            error_msg = str(e).replace('\n', ' ')

        # Execute Step
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action)
            step_data = step_resp.json()
            
            obs = step_data["observation"]
            reward = float(step_data["reward"]) # Changed to flattened format to match server
            done = step_data["done"]
            
            rewards.append(reward)
            
            trajectory.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": done
            })
            
            # EXACT MANDATORY STEP FORMAT
            act_str = action_raw.replace('\n', '')
            print(f"[STEP] step={step_count} action={act_str} reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}")
            
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            print(f"[STEP] step={step_count} action={action_raw.replace(chr(10), '')} reward=0.00 done=true error={error_msg}")
            break

    # Get final score from Grader
    try:
        grader_payload = {
            "task_id": TASK_NAME,
            "trajectory": trajectory
        }
        score_resp = requests.post(f"{ENV_URL}/grader", json=grader_payload)
        final_score = float(score_resp.json().get("score", 0.01))
    except Exception:
        final_score = 0.01

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success_str = "true" if final_score > 0.5 else "false"
    
    # EXACT MANDATORY END FORMAT
    print(f"[END] success={success_str} steps={step_count} score={final_score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    run_agent()
