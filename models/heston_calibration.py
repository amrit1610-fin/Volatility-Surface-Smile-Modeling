import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

from utils.implied_vol import ImpliedVolatility


class HestonPricer:
    """
    Heston model pricer using the COS (Fourier-Cosine) method.
    """
    def __init__(self, r: float, q: float = 0.0):
        self.r = r
        self.q = q
    
    @staticmethod
    def _char_func(u: np.ndarray, S0: float, T: float, r: float, q: float,
                kappa: float, theta: float, sigma: float, rho: float, v0: float) -> np.ndarray:
        """
        Heston characteristic function using the "Little Heston Trap" 
        """
        i = 1j
        tau = T
    
        beta = kappa - rho * sigma * i * u
        gamma = np.sqrt(sigma**2 * (u**2 + i * u) + beta**2)
        gamma = np.sqrt(np.maximum(gamma.real, 0) + 1j * gamma.imag)
        
        exp_m = np.exp(-gamma * tau)
        denom_g = beta + gamma
        denom_g = np.where(np.abs(denom_g) < 1e-12, 1e-12, denom_g)
        g = (beta - gamma) / denom_g

        numerator = (beta - gamma) / sigma**2 * (1 - exp_m)
        denominator = 1 - g * exp_m
        D = numerator / denominator

        log_term = np.log(denominator / (1 - g))
        term1 = (beta - gamma) * tau
        term2 = -2 * log_term
        C = (r - q) * i * u * tau + (kappa * theta / sigma**2) * (term1 + term2)
        
        # --- Characteristic function ---
        phi = np.exp(C + D * v0 + i * u * np.log(S0))
        
        return phi
    
    def price_call_cos(self, S0: float, K: float, T: float,
                   kappa: float, theta: float, sigma: float, rho: float, v0: float,
                   N: int = 128, L: float = 12.0) -> float:
        """
        Price a European call option using the COS method.
        """
        r = self.r
        q = self.q
        
        # 1. Integration domain
        x0 = np.log(S0)
        a = x0 - L * np.sqrt(T)
        b = x0 + L * np.sqrt(T)
        A = b - a  
        
        # 2. Payoff truncation
        payoff_a = np.log(K / S0)
        payoff_a = np.clip(payoff_a, a, b)
        
        # 3. Compute coefficients V_k
        V = np.zeros(N)
        for k in range(N):
            k_pi = k * np.pi / A
            # The correct multiplier for Fourier coefficients:
            # k=0 -> 1/A, k>0 -> 2/A
            coeff = 1.0 / A if k == 0 else 2.0 / A
            
            # Integral of S0 * e^y * cos(...)
            term_e = (np.cos(k_pi * A) * np.exp(b) - np.cos(k_pi * (payoff_a - a)) * np.exp(payoff_a)) / (1 + k_pi**2)
            term_e += k_pi * (np.sin(k_pi * A) * np.exp(b) - np.sin(k_pi * (payoff_a - a)) * np.exp(payoff_a)) / (1 + k_pi**2)
            integral_S0 = term_e
            
            # Integral of K * cos(...)
            if k == 0:
                integral_K = K * (b - payoff_a)
            else:
                integral_K = K * (np.sin(k_pi * A) - np.sin(k_pi * (payoff_a - a))) / k_pi
            
            # V_k = (Correct scaling) * ( S0*integral_S0 - integral_K )
            V[k] = coeff * (S0 * integral_S0 - integral_K)
        
        # 4. Characteristic function values at grid points
        # Note: u_k = k * pi / A
        u_k = np.array([k * np.pi / A for k in range(N)])
        phi_k = self._char_func(u_k, S0, T, r, q, kappa, theta, sigma, rho, v0)
        
        # 5. COS sum (Euler exponential term)
        F = np.zeros(N, dtype=complex)
        for k in range(N):
            F[k] = np.exp(1j * k * np.pi * (-a) / A) * V[k] * phi_k
        
        # 6. Discount and return
        price = np.exp(-r * T) * np.sum(F).real
        return max(price, 1e-8)  # Prevent negative prices                                      
    
    def implied_vol(self, price: float, S0: float, K: float, T: float,
                    option_type: str = 'call') -> float:
        """
        Compute implied volatility from a Heston price using Brent's method.
        """
        iv = ImpliedVolatility.compute_iv(
            S=S0,
            K=K,
            T=T,
            r=self.r,
            market_price=price,
            option_type=option_type
        )
        return iv


class HestonCalibrator:
    """
    Calibrates Heston model parameters to a market implied volatility surface.
    """
    
    def __init__(self, interpolator, risk_free_rate: float, dividend_yield: float = 0.0):
        """
        Initialize the calibrator.
        
        Args:
            interpolator: TenorInterpolator object (contains the SVI surface).
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield (default: 0.0).
        """
        self.interpolator = interpolator
        self.r = risk_free_rate
        self.q = dividend_yield
        self.pricer = HestonPricer(r=risk_free_rate, q=dividend_yield)
        self.results = None
    
    def build_calibration_grid(self, strikes: np.ndarray, T_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the grid of (strikes, T) and fetch market IVs from the SVI surface.
        
        Returns:
            strikes_mesh: 2D array of strikes.
            T_mesh: 2D array of T values.
            market_ivs: 2D array of market implied volatilities.
        """
        K_mesh, T_mesh = np.meshgrid(strikes, T_grid, indexing='ij')
        market_ivs = np.zeros_like(K_mesh)
        
        for i in range(K_mesh.shape[0]):
            for j in range(K_mesh.shape[1]):
                market_ivs[i, j] = self.interpolator.get_vol(K_mesh[i, j], T_mesh[i, j])
        
        return K_mesh, T_mesh, market_ivs
    
    def objective(self, params: np.ndarray, K_mesh: np.ndarray, T_mesh: np.ndarray,
                  market_ivs: np.ndarray, verbose: bool = False) -> float:
        """
        Objective function for calibration: RMSE between Heston and market implied volatilities.
        """
        kappa, theta, sigma, rho, v0 = params
        
        # Enforce Feller condition via penalty
        if 2 * kappa * theta <= sigma**2:
            return 1e6
        
        # Enforce parameter bounds
        if any([x < 0 for x in [kappa, theta, sigma, v0]]) or abs(rho) >= 1:
            return 1e6
        
        total_error = 0.0
        n_points = 0
        
        S0 = self.interpolator.underlyings[0]  # Take the first underlying as reference
        
        for i in range(K_mesh.shape[0]):
            for j in range(K_mesh.shape[1]):
                K = K_mesh[i, j]
                T = T_mesh[i, j]
                market_iv = market_ivs[i, j]
                
                if T <= 0 or market_iv <= 0 or np.isnan(market_iv):
                    continue
                
                # Price the option under Heston
                price = self.pricer.price_call_cos(
                    S0, K, T, kappa, theta, sigma, rho, v0,
                    N=64, L=12.0
                )
                
                # Convert price to implied volatility
                model_iv = self.pricer.implied_vol(price, S0, K, T, option_type='call')
                
                if not np.isnan(model_iv) and model_iv > 0:
                    error = (market_iv - model_iv) ** 2
                    total_error += error
                    n_points += 1
        
        if n_points == 0:
            return 1e6
        
        rmse = np.sqrt(total_error / n_points)
        return rmse
    
    def calibrate(self, strikes: np.ndarray, T_grid: np.ndarray,
                  initial_guess: Optional[np.ndarray] = None,
                  verbose: bool = True) -> Dict:
        """
        Calibrate Heston parameters to the SVI surface.
        
        Args:
            strikes: 1D array of strike prices.
            T_grid: 1D array of time-to-expiry points.
            initial_guess: [kappa, theta, sigma, rho, v0]. If None, auto-generate.
            verbose: Print progress.
        
        Returns:
            Dictionary with calibrated parameters and RMSE.
        """
        # Build the calibration grid
        K_mesh, T_mesh, market_ivs = self.build_calibration_grid(strikes, T_grid)
        
        # Auto-generate initial guess if not provided
        if initial_guess is None:
            # v0: ATM variance at the shortest expiry
            T_min = self.interpolator.T_values.min()
            S0 = self.interpolator.underlyings[0]
            atm_iv = self.interpolator.get_vol(S0, T_min)
            v0_guess = atm_iv**2
            
            # sigma (vol-of-vol): take from average SABR alpha
            # We need to extract SABR parameters from the results (if available)
            # If not, use a reasonable default
            sigma_guess = 0.5
            
            # rho: typical for equity is -0.7 to -0.9
            rho_guess = -0.7
            
            # kappa: mean-reversion speed (2-5 is common)
            kappa_guess = 2.0
            
            # theta: long-term variance (slightly higher than v0)
            theta_guess = v0_guess * 1.2
            
            initial_guess = [kappa_guess, theta_guess, sigma_guess, rho_guess, v0_guess]
        
        # Bounds
        bounds = [
            (0.01, 15.0),   # kappa (mean-reversion speed)
            (0.001, 1.0),   # theta (long-term variance)
            (0.01, 3.0),    # sigma (vol-of-vol)
            (-0.999, 0.999),# rho (correlation)
            (0.001, 1.0)    # v0 (initial variance)
        ]
        
        # Objective wrapper
        def obj(params):
            return self.objective(params, K_mesh, T_mesh, market_ivs, verbose=False)
        
        # Calibrate using SLSQP
        result = minimize(obj, initial_guess, method='SLSQP', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-8})
        
        if verbose:
            print("\n" + "="*50)
            print("HESTON CALIBRATION RESULTS")
            print("="*50)
            print(f"Success: {result.success}")
            print(f"Final RMSE: {result.fun:.6f}")
            print(f"Parameter:")
            print(f"  kappa (κ): {result.x[0]:.4f}")
            print(f"  theta (θ): {result.x[1]:.4f}")
            print(f"  sigma (σ): {result.x[2]:.4f}")
            print(f"  rho (ρ):   {result.x[3]:.4f}")
            print(f"  v0:        {result.x[4]:.4f}")
            
            # Feller condition check
            feller = 2 * result.x[0] * result.x[1] / (result.x[2]**2 + 1e-8)
            print(f"\nFeller Condition (2κθ/σ²): {feller:.2f} (Should be > 1)")
            if feller > 1:
                print("✅ Feller condition satisfied.")
            else:
                print("⚠️ Feller condition NOT satisfied (variance may hit zero).")
        
        self.results = {
            'params': result.x,
            'rmse': result.fun,
            'success': result.success,
            'feller': 2 * result.x[0] * result.x[1] / (result.x[2]**2 + 1e-8),
        }
        
        return self.results
    
    def plot_fit(self, strikes: np.ndarray, T_grid: np.ndarray, expiry_to_plot: float = None) -> None:
        """
        Plot Heston fit vs market surface for a given expiry.
        """
        if self.results is None:
            raise ValueError("Calibrate first using calibrate()")
        
        kappa, theta, sigma, rho, v0 = self.results['params']
        S0 = self.interpolator.underlyings[0]
        
        if expiry_to_plot is None:
            expiry_to_plot = self.interpolator.T_values[len(self.interpolator.T_values) // 2]
        
        # Get market IVs for this expiry
        market_ivs = []
        model_ivs = []
        
        for K in strikes:
            # Market IV from SVI surface
            market_iv = self.interpolator.get_vol(K, expiry_to_plot)
            market_ivs.append(market_iv)
            
            # Model IV from Heston
            price = self.pricer.price_call_cos(S0, K, expiry_to_plot, kappa, theta, sigma, rho, v0)
            model_iv = self.pricer.implied_vol(price, S0, K, expiry_to_plot)
            model_ivs.append(model_iv)
        
        # Plot
        plt.figure(figsize=(12, 7))
        plt.plot(strikes, market_ivs, 'bo-', label='Market (SVI Surface)', linewidth=2)
        plt.plot(strikes, model_ivs, 'r--', label='Heston Fit', linewidth=2)
        plt.axvline(x=S0, color='black', linestyle=':', alpha=0.7, label='ATM Spot')
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title(f'Heston Fit vs SVI Market Surface (T = {expiry_to_plot:.4f} years)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()