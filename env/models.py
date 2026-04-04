from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Action(BaseModel):
    allocations: Dict[str, float] = Field(
        default_factory=dict,
        description="Target portfolio weights for each ticker. Max sum = 1.0. Values should be between 0.0 and 1.0."
    )

class Observation(BaseModel):
    step: int
    portfolio_value: float
    cash: float
    holdings: Dict[str, float]
    market_data: Dict[str, Dict[str, float]]
    max_drawdown: float

class Reward(BaseModel):
    value: float
