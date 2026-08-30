import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, Optional


class TenorInterpolator:
    """
    Interpolates SVI parameters across time to build a continuous 3D volatility surface.
    """
    
    def __init__(self, results: Dict):
        self.results = results
        self._prepare_data()
        self._fit_interpolators()
    
    def _prepare_data(self) -> None:
        """Extract SVI parameters and sort by time-to-expiry."""
        # Sort by T to ensure monotonicity
        sorted_expiries = sorted(self.results.keys(), 
                                 key=lambda x: self.results[x]['T'])
        
        self.T_values = np.array([self.results[e]['T'] for e in sorted_expiries])
        
        # Stack SVI parameters: [a, b, rho, m, sigma]
        self.svi_params_array = np.array([
            [self.results[e]['svi_a'],
             self.results[e]['svi_b'],
             self.results[e]['svi_rho'],
             self.results[e]['svi_m'],
             self.results[e]['svi_sigma']]
            for e in sorted_expiries
        ])
        
        # Store forward and underlying for reference
        self.forwards = np.array([self.results[e]['forward'] for e in sorted_expiries])
        self.underlyings = np.array([self.results[e]['underlying'] for e in sorted_expiries])
        self.expiry_labels = sorted_expiries
    
    def _fit_interpolators(self) -> None:
        self.interpolators = {}
        
        # Param names and their preferred interpolation method
        param_config = {
            'a': PchipInterpolator,   # Total variance level must be increasing -> monotonic
            'b': CubicSpline,         # Wing slope can bend
            'rho': CubicSpline,       # Correlation can bend
            'm': PchipInterpolator,   # ATM shift should be smooth and monotonic
            'sigma': CubicSpline      # Curvature can bend
        }
        
        for idx, (name, interp_class) in enumerate(param_config.items()):
            y_vals = self.svi_params_array[:, idx]
            # PchipInterpolator requires strictly increasing x. Our T_values are already sorted.
            if interp_class == PchipInterpolator:
                self.interpolators[name] = interp_class(self.T_values, y_vals, 
                                                         extrapolate=True)
            else:
                # CubicSpline with natural boundary conditions
                self.interpolators[name] = interp_class(self.T_values, y_vals, 
                                                         bc_type='natural', 
                                                         extrapolate=True)
    
    def get_svi_params(self, T: float) -> np.ndarray:
        """
        Get interpolated SVI parameters at a specific time-to-expiry.
        """
        # --- 1. Clamp T strictly to the fitted range ---
        # Do NOT allow extrapolation outside the min/max T
        T_clamped = np.clip(T, self.T_values.min(), self.T_values.max())
        
        # --- 2. Get interpolated values ---
        a = self.interpolators['a'](T_clamped)
        b = self.interpolators['b'](T_clamped)
        rho = self.interpolators['rho'](T_clamped)
        m = self.interpolators['m'](T_clamped)
        sigma = self.interpolators['sigma'](T_clamped)
        
        # --- 3. HARD CLAMP parameters to valid ranges ---
        a = np.clip(a, 0.001, 1.0)       # Must be positive
        b = np.clip(b, 0.001, 2.0)       # Must be positive
        rho = np.clip(rho, -0.999, 0.999)  # Must be between -1 and 1
        # m can be any reasonable value, but clamp to prevent extreme shifts
        m = np.clip(m, -2.0, 2.0)
        sigma = np.clip(sigma, 0.001, 1.0)  # Must be positive
        
        return np.array([a, b, rho, m, sigma])
    
    def get_forward(self, T: float) -> float:
        """
        Interpolate forward price at time T.
        """
        return np.interp(T, self.T_values, self.forwards)
    
    def get_vol(self, strike: float, T: float, model: str = 'svi') -> float:
        """
        Get implied volatility at a specific (strike, T) point.
        """
        if model == 'svi':
            F = self.get_forward(T)
            k = np.log(strike / F)
            params = self.get_svi_params(T)
            
            # Compute total variance
            a, b, rho, m, sigma = params
            w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            w = np.maximum(w, 1e-8)
            
            return np.sqrt(w / T)
        else:
            raise NotImplementedError("SABR interpolation not yet implemented.")
    
    def get_surface(self, strikes: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
        """
        Compute the entire volatility surface over a grid of strikes and T.
        """
        surface = np.zeros((len(strikes), len(T_grid)))
        
        for i, T in enumerate(T_grid):
            F = self.get_forward(T)
            params = self.get_svi_params(T)
            k = np.log(strikes / F)
            
            a, b, rho, m, sigma = params
            w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            w = np.maximum(w, 1e-8)
            surface[:, i] = np.sqrt(w / T)
        
        return surface
    
    # =========================================================================
    # 3. Visualization
    # =========================================================================
    
    def plot_parameter_evolution(self) -> None:
        """
        Plot how each SVI parameter evolves with time-to-expiry.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_names = ['a (Level)', 'b (Wing Slope)', 'rho (Skew)', 
                       'm (ATM Shift)', 'sigma (Curvature)']
        
        for idx, name in enumerate(param_names):
            ax = axes[idx]
            ax.scatter(self.T_values, self.svi_params_array[:, idx], 
                      color='red', s=50, label='Fitted Points')
            
            # Plot smooth curve
            T_smooth = np.linspace(self.T_values.min(), self.T_values.max(), 100)
            y_smooth = self.interpolators[list(self.interpolators.keys())[idx]](T_smooth)
            ax.plot(T_smooth, y_smooth, 'b-', label='Interpolated')
            
            ax.set_xlabel('Time to Expiry (T)')
            ax.set_ylabel(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove the last empty subplot (if any)
        if len(param_names) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('SVI Parameter Evolution Across Tenors', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_surface_3d(self, strikes: np.ndarray, T_grid: np.ndarray) -> None:
        """
        Plot a 3D surface of implied volatility vs Strike and T.
        """
        surface = self.get_surface(strikes, T_grid)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        T_mesh, K_mesh = np.meshgrid(T_grid, strikes)
        
        # Plot surface
        surf = ax.plot_surface(K_mesh, T_mesh, surface, cmap='viridis', 
                               edgecolor='none', alpha=0.8)
        
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('3D Implied Volatility Surface (SVI Interpolation)')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Implied Volatility')
        plt.show()
    
    def plot_heatmap(self, strikes: np.ndarray, T_grid: np.ndarray) -> None:
        """
        Plot a heatmap of the volatility surface.
        """
        surface = self.get_surface(strikes, T_grid)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(surface.T, origin='lower', aspect='auto', 
                       cmap='viridis', extent=[strikes.min(), strikes.max(), 
                                               T_grid.min(), T_grid.max()])
        
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry (T)')
        ax.set_title('Volatility Surface Heatmap (SVI Interpolation)')
        fig.colorbar(im, ax=ax, label='Implied Volatility')
        
        # Mark the ATM line
        ax.axvline(x=self.underlyings[0], color='red', linestyle='--', alpha=0.5, label='ATM Spot')
        ax.legend()
        plt.show()