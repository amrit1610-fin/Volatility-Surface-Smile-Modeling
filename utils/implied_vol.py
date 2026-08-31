import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class ImpliedVolatility:

    def _black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def implement_implied_vol(self, row):
        S, K, T, r, price, opt_type = row['underlying_price'], row['strike'], row['T'], row['rate'], row['mid_price'], row['option_type']
        if T <= 0 or price <= 0:
            return np.nan
        
        # Check intrinsic value
        intrinsic = max(0, S - K) if opt_type == 'call' else max(0, K - S)
        if price < intrinsic:
            return np.nan  # Invalid arbitrage price
        
        def obj(sigma):
            return self._black_scholes_price(S, K, T, r, sigma, opt_type) - price
        
        try:
            # Brent's method needs a bracket. 
            # f(0) = intrinsic - price <= 0. f(5) is definitely > 0 for reasonable prices.
            iv = brentq(obj, 1e-6, 5.0, xtol=1e-8)
            return iv
        except (ValueError, RuntimeError):
            return np.nan

    @staticmethod
    def compute_iv(S: float, K: float, T: float, r: float, 
                   market_price: float, option_type: str = 'call') -> float:
        """
        Universal static method to compute implied volatility from individual arguments.
        This is used by HestonPricer and any other module that needs IV without a DataFrame row.
        """
        if T <= 0 or market_price <= 0:
            return np.nan
        
        # Intrinsic value check
        intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
        if market_price < intrinsic:
            return np.nan
        
        # Temporary helper to price BS (since we don't have self here, we instantiate a local helper)
        # But we can reuse the same logic using a nested function.
        def bs_price(sigma):
            if T <= 0 or sigma <= 0:
                return intrinsic
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        def obj(sigma):
            return bs_price(sigma) - market_price
        
        try:
            iv = brentq(obj, 1e-6, 5.0, xtol=1e-8)
            return iv
        except (ValueError, RuntimeError):
            return np.nan