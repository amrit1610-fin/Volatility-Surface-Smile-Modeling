import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Your existing modules ---
from data.data_cleaning import CleanedOptionChain
from utils.implied_vol import ImpliedVolatility
from utils.arbitrage_checks import ArbitrageChecks
from utils.tenor_interpolation import TenorInterpolator
from models.volatility_fitting import VolatilityFitter
from models.heston_calibration import HestonCalibrator

# ============================= CONFIGURATION ===========================================
data_folder_path = "data/option_chains"
spot = 24281.30
rate = 0.07
dividend_yield = 0.013

# ============================= DATA & PREPROCESSING ====================================
print("="*60)
print("PHASE 1-2: DATA CLEANING & ARBITRAGE CHECKS")
print("="*60)

data_engine = CleanedOptionChain(data_folder_path, spot, rate)
cleaned_dfs = data_engine.process_all()

# IV Calculation
print("\nIV CALCULATIONS")
iv_engine = ImpliedVolatility()
for df in cleaned_dfs:
    df['iv_calc'] = df.apply(iv_engine.implement_implied_vol, axis=1)
    df.dropna(subset=['iv_calc'], inplace=True)
    print("  IV implemented for one expiry.")

# Butterfly Arbitrage
print("\nCHECKING BUTTERFLY ARBITRAGES")
arb_engine = ArbitrageChecks()
for df in cleaned_dfs:
    df, violations = arb_engine.check_butterfly_arb(df)
    if violations:
        print(f"   Butterfly violations found: {len(violations)}. Corrected.")
    else:
        print(f"   ✅ No butterfly violations.")
    # Recompute IV after corrections
    df['iv_calc'] = df.apply(iv_engine.implement_implied_vol, axis=1)
    df.dropna(subset=['iv_calc'], inplace=True)

# Concatenate
main_df = data_engine.concatenate_data(cleaned_dfs)

# Calendar Arbitrage
final_df, violations = arb_engine.check_calendar_arb(main_df)
if violations:
    print(f"   Calendar violations found: {len(violations)}. Corrected.")
else:
    print(f"   ✅ No Calendar violations.")

# Add forward and log-moneyness
final_df['forward'] = final_df['underlying_price'] * np.exp((final_df['rate'] - dividend_yield) * final_df['T'])
final_df['log_moneyness'] = np.log(final_df['strike'] / final_df['forward'])

print(f"\nData ready: {len(final_df)} rows across {final_df['expiry'].nunique()} expiries.")

# ============================= VOLATILITY FITTING (SVI & SABR) ==========================
print("\n" + "="*60)
print("PHASE 3: SVI & SABR FITTING (PER EXPIRY)")
print("="*60)

vol_fitter = VolatilityFitter(final_df, beta_sabr=0.5)
results = vol_fitter.fit_all(verbose=True)

params_df = vol_fitter.get_params_dataframe()
print("\nFitted Parameters (first 5 rows):")
print(params_df.head())

# Plot fit for a specific expiry (optional)
vol_fitter.plot_fit(expiry_to_plot='15-Sep-2026')

# ============================= TENOR INTERPOLATION (3D SVI SURFACE) ====================
print("\n" + "="*60)
print("PHASE 4: TENOR INTERPOLATION & 3D SVI SURFACE")
print("="*60)

interpolator = TenorInterpolator(results)

# Define grid for surface
strikes_grid = np.linspace(20000, 27000, 50)
T_min = interpolator.T_values.min()
T_max = interpolator.T_values.max()
T_grid = np.linspace(T_min, T_max, 30)  # Slightly coarser for speed

# Compute the SVI surface
svi_surface = interpolator.get_surface(strikes_grid, T_grid)

# Query a specific point
T_42days = 42 / 365.0
iv_at_point = interpolator.get_vol(strike=24500.0, T=T_42days)
print(f"IV for Strike 24500, 42 days (SVI): {iv_at_point:.4f}")

# ============================= HESTON CALIBRATION ======================================
print("\n" + "="*60)
print("PHASE 5: GLOBAL HESTON CALIBRATION")
print("="*60)

# Instantiate calibrator
calibrator = HestonCalibrator(
    interpolator=interpolator,
    risk_free_rate=rate,
    dividend_yield=dividend_yield
)

# ===== SINGLE-POINT DEBUG TEST =====
print("\n" + "="*60)
print("DEBUG: TESTING HESTON PRICER ON A SINGLE POINT")
print("="*60)

S0 = spot
K = 24500.0
T = 0.0548  # ~20 days (e.g., 15-Sep)

# 1. Get market IV from SVI surface
market_iv = interpolator.get_vol(K, T)
print(f"Market IV (SVI) at K={K}, T={T:.4f}: {market_iv:.6f}")

# 2. Compute Heston price using initial guess (same as calibrator)
kappa_guess = 2.0
theta_guess = 0.155
sigma_guess = 0.5
rho_guess = -0.7
v0_guess = 0.1292

price = calibrator.pricer.price_call_cos(
    S0, K, T, 
    kappa_guess, theta_guess, sigma_guess, rho_guess, v0_guess,
    N=128, L=12.0
)
print(f"Heston Price: {price:.6f}")

# 3. Compute intrinsic value
intrinsic = max(S0 * np.exp(-dividend_yield * T) - K * np.exp(-rate * T), 0.0)
print(f"Intrinsic Value: {intrinsic:.6f}")
print(f"Price - Intrinsic: {price - intrinsic:.6f}")

# 4. Compute implied volatility from the Heston price
model_iv = calibrator.pricer.implied_vol(price, S0, K, T, option_type='call')
print(f"Heston IV: {model_iv:.6f}")

# 5. Check if the price is reasonable
if price < intrinsic:
    print("❌ ERROR: Heston price is below intrinsic! This will cause IV = NaN.")
else:
    print("✅ Price is above intrinsic.")
    
# Calibrate (use a sparser grid for speed, but enough for accuracy)
strikes_calib = np.linspace(20000, 27000, 25)
T_calib = np.linspace(T_min, T_max, 15)
heston_results = calibrator.calibrate(strikes_calib, T_calib, verbose=True)

# Extract calibrated parameters
params = heston_results['params']
kappa, theta, sigma, rho, v0 = params
print(f"\nCalibrated Heston Parameters:")
print(f"  κ (kappa) = {kappa:.4f}")
print(f"  θ (theta) = {theta:.4f}")
print(f"  σ (sigma) = {sigma:.4f}")
print(f"  ρ (rho)   = {rho:.4f}")
print(f"  v0        = {v0:.4f}")
print(f"  RMSE      = {heston_results['rmse']:.6f}")
print(f"  Feller    = {heston_results['feller']:.2f}")

# ============================= COMPARISON SUBPLOTS ====================================
print("\n" + "="*60)
print("PHASE 6: COMPARISON PLOTS (SVI vs HESTON SURFACE)")
print("="*60)

# 1. Compute the Heston surface on the same grid
heston_surface = np.zeros_like(svi_surface)
S0 = spot  # Use the spot from the data (or first underlying)

for i, K in enumerate(strikes_grid):
    for j, T in enumerate(T_grid):
        # Price the call under Heston
        price = calibrator.pricer.price_call_cos(
            S0, K, T, kappa, theta, sigma, rho, v0,
            N=64, L=12.0
        )
        # Convert to implied vol
        iv = calibrator.pricer.implied_vol(price, S0, K, T, option_type='call')
        heston_surface[i, j] = iv if not np.isnan(iv) else 0.0

# 2. Create side-by-side 3D subplots
fig = plt.figure(figsize=(16, 7))

# --- Subplot 1: SVI Surface ---
ax1 = fig.add_subplot(121, projection='3d')
T_mesh, K_mesh = np.meshgrid(T_grid, strikes_grid)
surf1 = ax1.plot_surface(K_mesh, T_mesh, svi_surface, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title('SVI Interpolated Surface (Base)')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Time to Expiry (T)')
ax1.set_zlabel('Implied Volatility')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# --- Subplot 2: Heston Surface ---
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(K_mesh, T_mesh, heston_surface, cmap='plasma', edgecolor='none', alpha=0.8)
ax2.set_title('Heston Calibrated Surface')
ax2.set_xlabel('Strike')
ax2.set_ylabel('Time to Expiry (T)')
ax2.set_zlabel('Implied Volatility')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.suptitle('Volatility Surface Comparison: SVI (Base) vs Heston (Calibrated)', fontsize=14)
plt.tight_layout()
plt.show()

# 3. Optional: Overlay plot for a specific expiry
expiry_to_plot = T_grid[len(T_grid)//2]  # middle expiry
calibrator.plot_fit(strikes_calib, T_calib, expiry_to_plot=expiry_to_plot)

print("\n✅ All phases completed successfully!")