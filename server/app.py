from fastapi import FastAPI, Body
import uvicorn
import yaml
import os
from typing import List, Dict, Any
from env.environment import MultiStockEnv
from env.models import Action
from env.tasks import (
    grade_single_stock_profit,
    grade_multi_stock_diversification,
    grade_risk_adjusted_returns
)

app = FastAPI()
env = MultiStockEnv()

# Load tasks once from openenv.yaml
YAML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
with open(YAML_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)
    TASKS = CONFIG.get("tasks", [])

@app.get("/tasks")
def get_tasks():
    # Compliant task discovery endpoint
    return TASKS

@app.post("/reset")
def reset():
    return env.reset()

@app.get("/state")
def state():
    return env.state()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.post("/evaluate")
def evaluate(payload: Dict[str, Any] = Body(...)):
    # Compliant evaluation endpoint
    task_id = payload.get("task_id")
    trajectory = payload.get("trajectory", [])
    
    if task_id == "single_stock_profit":
        score = grade_single_stock_profit(trajectory)
    elif task_id == "multi_stock_diversification":
        score = grade_multi_stock_diversification(trajectory)
    elif task_id == "risk_adjusted_returns":
        score = grade_risk_adjusted_returns(trajectory)
    else:
        # Default or unknown task
        score = 0.001
        
    return {"score": score}

@app.get("/")
def health_check():
    return {"message": "OpenEnv Multi-Stock Running", "status": 200}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
