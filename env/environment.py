from typing import Tuple, Dict, Any, Optional
import math
from .models import Action, Observation, Reward

class MultiStockEnv:
    def __init__(self):
        self.tickers = ["AAPL", "MSFT", "TSLA", "GME", "AMC"]
        self.initial_balance = 100000.0
        self.max_steps = 30
        self.reset()
        
    def reset(self) -> Observation:
        self.current_step = 0
        self.cash = self.initial_balance
        self.holdings = {ticker: 0.0 for ticker in self.tickers} # Number of shares
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0
        self.history = []
        return self.state()
        
    def _generate_market_data(self, step: int) -> Dict[str, Dict[str, float]]:
        # Deterministic synthetic market generator based on step
        data = {}
        # AAPL: slow steady growth
        aapl_price = 150.0 * (1.0 + 0.002 * step)
        
        # MSFT: steady growth with slight cycles
        msft_price = 300.0 * (1.0 + 0.001 * step + 0.02 * math.sin(step / 3.0))
        
        # TSLA: volatile, huge dip then recovery
        tsla_price = 200.0 * (1.0 - 0.05 * step if step < 10 else 0.5 + 0.08 * (step - 10))
        
        # GME: Flat, then massive spike at step 15
        gme_price = 20.0 * (1.0 if step < 15 else (5.0 if step == 15 else 2.0))
        
        # AMC: continuous decay
        amc_price = 50.0 * (0.95 ** step)
        
        prices = {"AAPL": aapl_price, "MSFT": msft_price, "TSLA": tsla_price, "GME": gme_price, "AMC": amc_price}
        
        for t in self.tickers:
            data[t] = {
                "price": round(prices[t], 2),
                "trend_signal": round(prices[t] / prices[t] * 1.0, 2) # simplified mock
            }
        return data
        
    def state(self) -> Observation:
        market = self._generate_market_data(self.current_step)
        
        # Calculate current portfolio value based on latest prices
        held_value = sum(self.holdings[t] * market[t]["price"] for t in self.tickers)
        total_value = self.cash + held_value
        
        return Observation(
            step=self.current_step,
            portfolio_value=round(total_value, 2),
            cash=round(self.cash, 2),
            holdings=self.holdings,
            market_data=market,
            max_drawdown=round(self.max_drawdown, 4)
        )
        
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        market = self._generate_market_data(self.current_step)
        
        # Calculate portfolio value pre-execution
        held_value = sum(self.holdings[t] * market[t]["price"] for t in self.tickers)
        total_value = self.cash + held_value
        
        # Sell everything to reallocate cleanly
        self.cash = total_value
        self.holdings = {t: 0.0 for t in self.tickers}
        
        # Process new allocations
        total_alloc = sum(abs(v) for v in action.allocations.values())
        if total_alloc > 1.0:
            # normalize if exceeding 1.0
            allocs = {k: v/total_alloc for k, v in action.allocations.items()}
        else:
            allocs = action.allocations
            
        for ticker, alloc in allocs.items():
            if ticker in self.tickers and alloc > 0:
                spend = total_value * alloc
                shares = spend / market[ticker]["price"]
                self.holdings[ticker] = shares
                self.cash -= spend
                
        # Advance time
        self.current_step += 1
        
        # Calculate new value
        new_market = self._generate_market_data(self.current_step)
        new_held = sum(self.holdings[t] * new_market[t]["price"] for t in self.tickers)
        new_total = self.cash + new_held
        
        self.portfolio_value = new_total
        
        if new_total > self.max_portfolio_value:
            self.max_portfolio_value = new_total
        else:
            drawdown = (self.max_portfolio_value - new_total) / self.max_portfolio_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                
        # Meaningful reward: normalized return * (1 - penalty for drawdown)
        step_return = (new_total - total_value) / total_value
        reward_val = step_return - (self.max_drawdown * 0.5)
        
        done = self.current_step >= self.max_steps
        
        obs = self.state()
        reward = Reward(value=reward_val)
        
        return obs, reward, done, {}
