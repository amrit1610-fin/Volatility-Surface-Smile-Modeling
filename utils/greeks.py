import numpy as np
import pandas as pd
from typing import Dict
from models.heston_calibration import HestonPricer

class HestonGreeks:
    """
    Calculates 1st and 2nd order sensitivities (Greeks) for the calibrated Heston model 
    using finite differences.
    """
    def __init__(self, pricer: HestonPricer, params: np.ndarray, S0: float):
        self.pricer = pricer
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = params
        self.S0 = S0
        
        # Bump sizes
        self.dS = S0 * 0.01          # 1% bump in spot
        self.dv = max(self.v0 * 0.05, 0.001)  # 5% bump in initial variance

    def compute_risk_ladder(self, strikes: np.ndarray, T: float) -> pd.DataFrame:
        """
        Generates a risk ladder containing Delta, Vega, Vanna, and Volga 
        across a strip of strikes for a given tenor.
        """
        results = []
        
        # Pre-calculate bumped states
        S_up = self.S0 + self.dS
        S_dn = self.S0 - self.dS
        
        v_up = self.v0 + self.dv
        v_dn = self.v0 - self.dv
        
        for K in strikes:
            # Base Price
            P_base = self.pricer.price_call_cos(
                self.S0, K, T, self.kappa, self.theta, self.sigma, self.rho, self.v0)
            
            # Spot Bumps (Delta)
            P_S_up = self.pricer.price_call_cos(
                S_up, K, T, self.kappa, self.theta, self.sigma, self.rho, self.v0)
            P_S_dn = self.pricer.price_call_cos(
                S_dn, K, T, self.kappa, self.theta, self.sigma, self.rho, self.v0)
            
            # Vol Bumps (Vega & Volga)
            P_v_up = self.pricer.price_call_cos(
                self.S0, K, T, self.kappa, self.theta, self.sigma, self.rho, v_up)
            P_v_dn = self.pricer.price_call_cos(
                self.S0, K, T, self.kappa, self.theta, self.sigma, self.rho, v_dn)
            
            # Cross Bump (Vanna: bump spot AND vol up)
            P_cross = self.pricer.price_call_cos(
                S_up, K, T, self.kappa, self.theta, self.sigma, self.rho, v_up)
            
            # --- Greek Calculations ---
            # Delta: dV / dS
            delta = (P_S_up - P_S_dn) / (2 * self.dS)
            
            # Vega: dV / d(sqrt(v)) -> Approximated by variance bump
            vega = (P_v_up - P_v_dn) / (2 * self.dv)
            
            # Volga: d^2V / dv^2
            volga = (P_v_up - 2 * P_base + P_v_dn) / (self.dv ** 2)
            
            # Vanna: d^2V / dS dv
            # Using standard cross-derivative finite difference approximation
            vanna = (P_cross - P_S_up - P_v_up + P_base) / (self.dS * self.dv)
            
            results.append({
                'Strike': K,
                'Price': P_base,
                'Delta': delta,
                'Vega': vega,
                'Vanna': vanna,
                'Volga': volga
            })
            
        return pd.DataFrame(results)