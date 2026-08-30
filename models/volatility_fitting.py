import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Union


class VolatilityFitter:
    """
    Calibrate SVI and SABR models to implied volatility slices.
    """
    
    def __init__(self, final_df: pd.DataFrame, beta_sabr: float = 0.5):
        self.final_df = final_df.copy()
        self.beta = beta_sabr
        self.results: Dict = {}
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Ensure required columns exist in the input DataFrame."""
        required_cols = ['expiry', 'T', 'strike', 'forward', 'log_moneyness', 
                         'iv_calc', 'bid', 'ask', 'underlying_price', 'option_type']
        missing = [col for col in required_cols if col not in self.final_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # =========================================================================
    # 1. Static Model Calculations (SVI & SABR)
    # =========================================================================
    
    @staticmethod
    def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, 
                           m: float, sigma: float) -> np.ndarray:
        """
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        """
        wk = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        return wk
    
    @staticmethod
    def svi_implied_vol(k: np.ndarray, T: float, params: List[float]) -> np.ndarray:
        """Convert SVI total variance to implied volatility."""
        a, b, rho, m, sigma = params
        w = VolatilityFitter.svi_total_variance(k, a, b, rho, m, sigma)
        w = np.maximum(w, 1e-8)                                            # prevent negative variance
        return np.sqrt(w / T)
    
    @staticmethod
    def sabr_implied_vol(F: float, K: float, T: float, alpha: float, 
                         beta: float, rho: float, nu: float) -> float:
        """Hagan's SABR implied volatility approximation."""

        if F == K:
            # ATM special case
            term1 = alpha / (F ** (1 - beta))
            term2 = (1 + ((1 - beta)**2 / 24 * alpha**2 / F**(2 - 2*beta) +
                         1/4 * rho * beta * nu * alpha / F**(1 - beta) +
                         (2 - 3*rho**2) / 24 * nu**2) * T)
            return term1 * term2
        
        FK = F * K
        fk_beta = (FK) ** ((1 - beta) / 2)
        log_FK = np.log(F / K)
        z = (nu / alpha) * fk_beta * log_FK
        
        # Avoiding division by zero near z=0
        if abs(z) < 1e-12:
            x_z = 1.0
        else:
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        if abs(x_z) < 1e-12:
            term1 = alpha / fk_beta
        else:
            term1 = alpha / fk_beta * (z / x_z)
        
        adj = (1 + ((1 - beta)**2 / 24 * alpha**2 / fk_beta**2 +
                    1/4 * rho * beta * nu * alpha / fk_beta +
                    (2 - 3*rho**2) / 24 * nu**2) * T)
        
        return term1 * adj
    
    # =========================================================================
    # 2. Weight Calculation
    # =========================================================================
    
    @staticmethod
    def compute_vega_weights(df_slice: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized Vega weights for Weighted Least Squares.
        Weight = Vega / (Ask - Bid + 1e-6), normalized to sum to 1.
        """
        S = df_slice['underlying_price'].iloc[0]
        T = df_slice['T'].iloc[0]
        F = df_slice['forward'].iloc[0]
        K = df_slice['strike'].values
        iv = df_slice['iv_calc'].values
        spread = df_slice['ask'] - df_slice['bid']
        
        d1 = (np.log(F / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T) + 1e-8)
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        weights = vega / (spread + 1e-6)
        weights = weights / (weights.sum() + 1e-8)
        return weights
    
    # =========================================================================
    # 3. Fitting Methods (Per Expiry)
    # =========================================================================
    
    def _fit_svi_slice(self, df_slice: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """
        Fit SVI to a single expiry slice using SLSQP.
        Enforces no-arbitrage constraint: b * (1 + |rho|) <= 2.
        """
        k = df_slice['log_moneyness'].values
        T = df_slice['T'].iloc[0]
        market_iv = df_slice['iv_calc'].values
        weights = self.compute_vega_weights(df_slice)
        
        # --- Initial Guess ---
        atm_idx = np.argmin(np.abs(df_slice['strike'] - df_slice['underlying_price'].iloc[0]))
        atm_k = k[atm_idx]
        atm_iv = market_iv[atm_idx]
        
        a_guess = np.clip(0.5 * (atm_iv**2 * T), 0.01, 0.5)
        x0 = [a_guess, 0.5, -0.5, atm_k, 0.3]                   # [a, b, rho, m, sigma]
        
        # --- Objective ---
        def objective(params):
            a, b, rho, m, sigma = params
            w = VolatilityFitter.svi_total_variance(k, a, b, rho, m, sigma)
            model_iv = np.sqrt(np.maximum(w, 1e-8) / T)
            return np.sum(weights * (market_iv - model_iv)**2)
        
        # --- Constraint: b * (1 + |rho|) <= 2 ---
        def constraint(params):
            a, b, rho, m, sigma = params
            return 2 - b * (1 + abs(rho))
        
        cons = [{'type': 'ineq', 'fun': constraint}]
        bounds = [(0.001, 1.0), (0.001, 2.0), (-0.999, 0.999), (-2.0, 2.0), (0.001, 1.0)]
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 1000, 'ftol': 1e-8})
        
        if verbose:
            rmse = np.sqrt(result.fun / len(k))
            print(f"  SVI fit success: {result.success}, RMSE: {rmse:.6f}")
        return result.x if result.success else x0
    
    def _fit_sabr_slice(self, df_slice: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """
        Fit SABR to a single expiry slice using SLSQP.
        Beta is fixed to self.beta.
        """
        F = df_slice['forward'].iloc[0]
        T = df_slice['T'].iloc[0]
        K = df_slice['strike'].values
        market_iv = df_slice['iv_calc'].values
        weights = self.compute_vega_weights(df_slice)
        
        # --- Initial Guess ---
        atm_idx = np.argmin(np.abs(df_slice['strike'] - df_slice['underlying_price'].iloc[0]))
        atm_iv = market_iv[atm_idx]
        alpha_guess = np.clip(0.4 * atm_iv, 0.05, 1.0)
        x0 = [alpha_guess, -0.5, 0.4]  # [alpha, rho, nu]
        
        # --- Objective ---
        def objective(params):
            alpha, rho, nu = params
            model_iv = np.array([VolatilityFitter.sabr_implied_vol(F, k, T, alpha, self.beta, rho, nu) 
                                 for k in K])
            return np.sum(weights * (market_iv - model_iv)**2)
        
        bounds = bounds = [(0.001, 5.0), (-0.999, 0.999), (0.001, 5.0)]
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-8})
        
        if verbose:
            rmse = np.sqrt(result.fun / len(K))
            print(f"  SABR fit success: {result.success}, RMSE: {rmse:.6f}")
        return result.x if result.success else x0
    
    # =========================================================================
    # 4. Master Fitting Loop
    # =========================================================================
    
    def fit_all(self, verbose: bool = False) -> Dict:
        """
        Fit SVI and SABR for all unique expiries in the dataset.
        """
        self.results = {}
        
        for expiry in self.final_df['expiry'].unique():
            slice_df = self.final_df[self.final_df['expiry'] == expiry].copy().sort_values('strike')
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"Fitting expiry: {expiry} (N={len(slice_df)})")
            
            # Fit SVI
            svi_params = self._fit_svi_slice(slice_df, verbose=verbose)
            
            # Fit SABR
            sabr_params = self._fit_sabr_slice(slice_df, verbose=verbose)
            
            # Store results
            self.results[expiry] = {
                'expiry': expiry,
                'T': slice_df['T'].iloc[0],
                'forward': slice_df['forward'].iloc[0],
                'underlying': slice_df['underlying_price'].iloc[0],
                'svi_params': svi_params,
                'sabr_params': sabr_params,
                'svi_a': svi_params[0],
                'svi_b': svi_params[1],
                'svi_rho': svi_params[2],
                'svi_m': svi_params[3],
                'svi_sigma': svi_params[4],
                'sabr_alpha': sabr_params[0],
                'sabr_rho': sabr_params[1],
                'sabr_nu': sabr_params[2],
                'sabr_beta': self.beta,
            }
        
        return self.results
    
    # =========================================================================
    # 5. Utility Methods
    # =========================================================================
    
    def get_results(self) -> Dict:
        """Return the fitted results dictionary."""
        if not self.results:
            raise ValueError("No results available. Run fit_all() first.")
        return self.results
    
    def get_params_dataframe(self) -> pd.DataFrame:
        """
        Convert the results dictionary into a pandas DataFrame for easy analysis.
        """
        if not self.results:
            raise ValueError("No results available. Run fit_all() first.")
        return pd.DataFrame.from_dict(self.results, orient='index')
    
    # =========================================================================
    # 6. Plotting
    # =========================================================================
    
    def plot_fit(self, expiry_to_plot: Union[str, pd.Timestamp]) -> None:
        """
        Plot market IV along with SVI and SABR fitted curves for a given expiry.
        """
        if not self.results:
            raise ValueError("No results available. Run fit_all() first.")
        
        if expiry_to_plot not in self.results:
            raise ValueError(f"Expiry {expiry_to_plot} not found in results. "
                             f"Available: {list(self.results.keys())}")
        
        slice_df = self.final_df[self.final_df['expiry'] == expiry_to_plot].sort_values('strike')
        res = self.results[expiry_to_plot]
        
        T = res['T']
        F = res['forward']
        svi_params = res['svi_params']
        sabr_params = res['sabr_params']
        beta = res['sabr_beta']
        
        # Generate smooth curves
        strikes_plot = np.linspace(slice_df['strike'].min(), slice_df['strike'].max(), 200)
        k_plot = np.log(strikes_plot / F)
        
        # SVI
        iv_svi = VolatilityFitter.svi_implied_vol(k_plot, T, svi_params)
        
        # SABR
        iv_sabr = np.array([VolatilityFitter.sabr_implied_vol(F, K, T, sabr_params[0], 
                                                              beta, sabr_params[1], sabr_params[2]) 
                            for K in strikes_plot])
        
        # Plot
        plt.figure(figsize=(12, 7))
        
        calls = slice_df[slice_df['option_type'] == 'call']
        puts = slice_df[slice_df['option_type'] == 'put']
        
        plt.scatter(calls['strike'], calls['iv_calc'], label='Calls (Market)', 
                    color='blue', alpha=0.6, s=50)
        plt.scatter(puts['strike'], puts['iv_calc'], label='Puts (Market)', 
                    color='red', alpha=0.6, s=50)
        plt.plot(strikes_plot, iv_svi, 'g-', label='SVI Fit (SLSQP)', linewidth=2)
        plt.plot(strikes_plot, iv_sabr, 'orange', linestyle='--', label='SABR Fit (SLSQP)', linewidth=2)
        plt.axvline(x=F, color='black', linestyle=':', alpha=0.7, label='Forward (ATM)')
        
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title(f'SVI & SABR Fit - {expiry_to_plot} (SLSQP Optimizer)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_surface(self) -> None:
        """
        (Placeholder) Will be implemented in Phase 4: Tenor Interpolation.
        This is a reminder to build the 3D surface in the next phase.
        """
        print("3D Surface plot coming in Phase 4 (Tenor Interpolation).")