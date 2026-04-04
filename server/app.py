from fastapi import FastAPI
import uvicorn
from env.environment import MultiStockEnv
from env.models import Action

app = FastAPI()
env = MultiStockEnv()

@app.get("/reset")
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

@app.get("/")
def health_check():
    return {"message": "OpenEnv Multi-Stock Running", "status": 200}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
