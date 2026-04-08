from typing import Dict, Any, List

def clamp_score(score: float) -> float:
    # Hackathon requirement: Each task's score must be strictly between 0 and 1 (not 0.0 and not 1.0)
    return max(0.001, min(0.999, float(score)))

def grade_single_stock_profit(trajectory: List[Dict[str, Any]]) -> float:
    # Easy Task: agent traded a single stock and made profit
    if not trajectory:
        return clamp_score(0.0)
    final_obs = trajectory[-1]["observation"]
    start_value = trajectory[0]["observation"]["portfolio_value"] if isinstance(trajectory[0]["observation"], dict) else trajectory[0]["observation"].portfolio_value
    final_value = final_obs["portfolio_value"] if isinstance(final_obs, dict) else final_obs.portfolio_value
    
    # Needs to be >0 profit. More profit = higher score up to 1.0 (Say, 10% profit is full score)
    roi = (final_value - start_value) / start_value
    if roi <= 0:
        return clamp_score(0.0)
    return clamp_score(min(1.0, roi / 0.10))

def grade_multi_stock_diversification(trajectory: List[Dict[str, Any]]) -> float:
    # Medium Task: Maximize profit but MUST hold at least 2 stocks in most steps
    if not trajectory:
        return clamp_score(0.0)
    final_obs = trajectory[-1]["observation"]
    start_value = trajectory[0]["observation"]["portfolio_value"] if isinstance(trajectory[0]["observation"], dict) else trajectory[0]["observation"].portfolio_value
    final_value = final_obs["portfolio_value"] if isinstance(final_obs, dict) else final_obs.portfolio_value
    
    roi = (final_value - start_value) / start_value
    if roi <= 0:
        return clamp_score(0.0)
        
    diversification_score = 0
    for step in trajectory:
        obs = step["observation"]
        holdings = obs["holdings"] if isinstance(obs, dict) else obs.holdings
        held = sum(1 for v in holdings.values() if v > 0)
        if held >= 2:
            diversification_score += 1
            
    # Need to be diversified at least 50% of the time
    div_ratio = diversification_score / len(trajectory)
    
    roi_score = min(1.0, roi / 0.15)
    
    if div_ratio < 0.5:
        return clamp_score(roi_score * 0.5)
    return clamp_score(roi_score)

def grade_risk_adjusted_returns(trajectory: List[Dict[str, Any]]) -> float:
    # Hard Task: maximize profit while max_drawdown < 10%
    if not trajectory:
        return clamp_score(0.0)
    final_obs = trajectory[-1]["observation"]
    start_value = trajectory[0]["observation"]["portfolio_value"] if isinstance(trajectory[0]["observation"], dict) else trajectory[0]["observation"].portfolio_value
    final_value = final_obs["portfolio_value"] if isinstance(final_obs, dict) else final_obs.portfolio_value
    max_drawdown = final_obs["max_drawdown"] if isinstance(final_obs, dict) else final_obs.max_drawdown
    
    roi = (final_value - start_value) / start_value
    if roi <= 0:
        return clamp_score(0.0)
        
    roi_score = min(1.0, roi / 0.20)
    
    # Drawdown penalty
    if max_drawdown > 0.10:
        return clamp_score(roi_score * 0.2) # heavy penalty for blowing risk limits
    elif max_drawdown > 0.05:
        return clamp_score(roi_score * 0.8)
        
    return clamp_score(roi_score)
