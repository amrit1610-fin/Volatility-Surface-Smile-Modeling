import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data.data_cleaning import CleanedOptionChain
from utils.implied_vol import ImpliedVolatility
from utils.arbitrage_checks import ArbitrageChecks
from utils.tenor_interpolation import TenorInterpolator
from models.volatility_fitting import VolatilityFitter
from models.heston_calibration import HestonCalibrator
from utils.greeks import HestonGreeks

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

# =======================================================================
# THE OTM-ONLY FILTER
# =======================================================================
# 1. Standard OTM Filter
is_put = final_df['option_type'].str.lower().isin(['put', 'pe'])
is_call = final_df['option_type'].str.lower().isin(['call', 'ce'])
otm_puts = is_put & (final_df['strike'] < final_df['forward'])
otm_calls = is_call & (final_df['strike'] >= final_df['forward'])
final_df = final_df[otm_puts | otm_calls]

# 2. Strict Liquidity & Moneyness Filter
# Cap IV at a realistic maximum for NIFTY (e.g., 40% or 0.40)
final_df = final_df[(final_df['iv_calc'] > 0.01) & (final_df['iv_calc'] < 0.40)]

# Restrict strikes to a tight, highly liquid window (e.g., +/- 8% from spot)
final_df = final_df[(final_df['log_moneyness'] > -0.08) & (final_df['log_moneyness'] < 0.08)]

# 3. Data Starvation Filter
# Keep only expiries with at least 5 valid options to prevent SVI optimizer failure
final_df = final_df.groupby('expiry').filter(lambda x: len(x) >= 5)

print(f"Post-filter (OTM-Only & Liquid) rows: {len(final_df)}")

# ============================= VOLATILITY FITTING (SVI & SABR) ==========================
print("\n" + "="*60)
print("PHASE 3: SVI & SABR FITTING (PER EXPIRY)")
print("="*60)

vol_fitter = VolatilityFitter(final_df, beta_sabr=0.5)
results = vol_fitter.fit_all(verbose=True)

params_df = vol_fitter.get_params_dataframe()
print("\nFitted Parameters (first 5 rows):")
print(params_df.head())

# Plot fit for a specific expiry
vol_fitter.plot_fit(expiry_to_plot='15-Sep-2026')

# ============================= TENOR INTERPOLATION (3D SVI SURFACE) ====================
print("\n" + "="*60)
print("PHASE 4: TENOR INTERPOLATION & 3D SVI SURFACE")
print("="*60)

interpolator = TenorInterpolator(results)

# Define grid for surface restricted to the liquid trained bounds
strikes_grid = np.linspace(22300, 26200, 50) 
T_min = interpolator.T_values.min()
T_max = interpolator.T_values.max()
T_grid = np.linspace(T_min, T_max, 30)

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
    
# Calibrate (use a sparser grid for speed, but enough for accuracy)
strikes_calib = np.linspace(22300, 26200, 15) 
T_calib = np.linspace(T_min, T_max, 8)
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
S0 = spot  # Use the spot from the data

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

# 3. Overlay plot for a specific expiry
expiry_to_plot = T_grid[len(T_grid)//2]  # middle expiry
calibrator.plot_fit(strikes_calib, T_calib, expiry_to_plot=expiry_to_plot)

# ============================= VANNA/VOLGA RISK LADDER =================================
print("\n" + "="*60)
print("PHASE 7: HESTON RISK LADDER (VANNA/VOLGA)")
print("="*60)

# Initialize the Greeks engine with your calibrated parameters
greeks_engine = HestonGreeks(pricer=calibrator.pricer, params=params, S0=spot)

# Generate ladder for a 42-day tenor across the liquid strike range
ladder_T = 42 / 365.0
ladder_strikes = np.linspace(23500, 25500, 9)

risk_ladder = greeks_engine.compute_risk_ladder(strikes=ladder_strikes, T=ladder_T)

print(f"\nRisk Ladder (T = {ladder_T:.4f} years):")
print(risk_ladder.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n✅ All phases completed successfully!")