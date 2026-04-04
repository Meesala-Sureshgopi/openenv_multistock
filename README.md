---
title: OpenEnv Multi-Stock
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv Multi-Stock Portfolio

An OpenEnv submission for the ET AI Hackathon.

## Description
This environment models a deterministic multi-stock market where an AI agent must allocate its capital across 5 different stocks to maximize return while minimizing drawdown. It solves the real-world utility of creating risk-aware automated trading agents.

## Action & Observation Space
- **Action Space**: A dictionary mapping ticker symbols to portfolio allocation percentages (values between 0.0 and 1.0).
- **Observation Space**: Step, Cash, Portfolio Value, Holdings, Multi-Stock price data, and Max Drawdown.

## Usage Overview
1. Start the Environment mapping `app.py` via FastAPI.
2. Endpoints: `/reset` to get initial observation, `/state` to view current state, and `/step` (POST Action) to get observation, reward, and done flags.

## Baseline Run
Run `python inference.py` (ensure `API_BASE_URL` and `OPENAI_API_KEY`/`HF_TOKEN` are valid).
