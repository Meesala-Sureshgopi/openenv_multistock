---
title: OpenEnv Multi-Stock Portfolio
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 💹 OpenEnv Multi-Stock Portfolio Manager

An advanced, real-world Financial RL Environment built for the **ET AI Hackathon**. This environment simulates a multi-stock market where an AI agent acts as a portfolio manager, making capital allocation decisions across several volatile assets.

## 🚀 Why this Environment?
Unlike "toy" games, **Portfolio Management** is a high-stakes, real-world task. This environment evaluates an agent's ability to:
1.  **Identify Trends**: Allocate capital to high-growth assets (Trend Following).
2.  **Manage Risk**: Protect capital during market shocks (Risk Management).
3.  **Diversify**: Intelligently spread risk across multiple tickers (Portfolio Diversification).

---

## 🛠️ Environment Architecture

### 📊 Assets Under Management (AUM)
The environment simulates 5 distinct stock behaviors (deterministic for fair evaluation):
- **AAPL**: Steady, slow growth.
- **MSFT**: Growth with cyclic volatility.
- **TSLA**: Highly volatile with parabolic recovery.
- **GME**: Flat with massive, unpredictable price spikes.
- **AMC**: Continuous decay (The "Short" target).

### 🕹️ Action Space
A dictionary of portfolio weights for each ticker. 
- **Type**: `Dict[str, float]`
- **Range**: `0.0` to `1.0` (Long allocations).
- **Constraint**: Total allocation sum must be `<= 1.0`. Any excess is automatically normalized.

### 👁️ Observation Space
- `portfolio_value`: Current total value (Cash + Holdings).
- `cash`: Available liquidity.
- `holdings`: Number of shares held per ticker.
- `market_data`: Current price and trend signals for all assets.
- `max_drawdown`: The peak-to-trough decline experienced so far.

---

## 🏆 Assessment Tasks (Easy → Hard)

| Task ID | Name | Objective | Success Criteria (Score 1.0) |
| :--- | :--- | :--- | :--- |
| `single_stock_profit` | **Easy** | Profitability on a single asset. | Achieve > 10% ROI on AAPL. |
| `multi_stock_div` | **Medium** | Diversified Trading. | Hold >= 2 stocks and achieve > 15% ROI. |
| `risk_adjusted` | **Hard** | Risk-Aware Growth. | Achieve > 20% ROI while keeping **Max Drawdown < 10%**. |

---

## 💻 Setup & Deployment

### Run via Docker (Recommended)
This is the command judges will use to spin up your environment:
```bash
docker build -t openenv-multistock .
docker run -p 7860:7860 openenv-multistock
```

### Local Development
1. Install dependencies: `pip install -r requirements.txt`
2. Run the FastAPI server: `python server/app.py`
3. The environment will be available at `http://127.0.0.1:7860`.

---

## 🤖 Running Inference (Baseline Agent)
The provided `inference.py` script benchmarks the environment using an LLM (OpenAI/Llama-3).

**MANDATORY: Set Environment Variables**
```powershell
$env:HF_TOKEN="your_huggingface_token"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Llama-3-70b-instruct"
```

**Run Inference:**
```bash
python inference.py
```
The script will log structured progress: `[START]`, `[STEP X]`, and `[END]`.

---

## 📄 License & Compliance
This submission is fully compliant with the **OpenEnv Spec v1.0.0**. 
- **Author**: Meesala Suresh Gopi
- **Framework**: FastAPI + Pydantic + Docker
