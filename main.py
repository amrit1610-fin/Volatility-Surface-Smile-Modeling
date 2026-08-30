import pandas as pd
import numpy as np
from data.data_cleaning import CleanedOptionChain
from utils.implied_vol import ImpliedVolatility
from utils.arbitrage_checks import ArbitrageChecks
from utils.tenor_interpolation import TenorInterpolator
from models.volatility_fitting import VolatilityFitter

# =============================DATA & PREPROCESSING===========================================
# VARIABLES
data_folder_path = "data/option_chains"
spot = 24281.30
rate = 0.07
dividend_yield = 0.013

# OPTIONS CHAIN DATA
data_engine = CleanedOptionChain(data_folder_path, spot, rate)
cleaned_dfs = []
cleaned_dfs = data_engine.process_all()

# ADDING CALCULATED IMPLIED VOLATILITY
print("\nIV CALCULATIONS")
iv_engine = ImpliedVolatility()
for df in cleaned_dfs:  
    df['iv_calc'] = df.apply(iv_engine.implement_implied_vol, axis=1)
    df.dropna(subset=['iv_calc'], inplace=True)                                # Dropping rows where IV calculation failed (deep OTM or stale quotes)
    print(f"IV implemented !!")

# CHECK FOR BUTTERFLY ARBITRAGE
print("\nCHECKING BUTTERFLY ARBITRAGES")
arb_engine = ArbitrageChecks()
for df in cleaned_dfs:
    df, violations = arb_engine.check_butterfly_arb(df)
    if violations:
        print(f"   Butterfly violations found: {len(violations)}. Corrected.")
    else:
        print(f"   ✅ No butterfly violations.")

    # RE-COMPUTE IV FOR THIS EXPIRY
    df['iv_calc'] = df.apply(iv_engine.implement_implied_vol, axis=1)
    df.dropna(subset=['iv_calc'], inplace=True)

# CONCATENATE DATAFRAMES
main_df = data_engine.concatenate_data(cleaned_dfs)

# CALENDAR ARBITRAGE CHECK
final_df, violations = arb_engine.check_calendar_arb(main_df)
if violations:
    print(f"   Calendar violations found: {len(violations)}. Corrected.")
else:
    print(f"   ✅ No Calendar violations.")


# IMPLEMENT LOG-MONEYNESS
final_df['forward'] = final_df['underlying_price'] * np.exp((final_df['rate'] - dividend_yield) * final_df['T'])
final_df['log_moneyness'] = np.log(final_df['strike'] / final_df['forward'])

#print(final_df.head())


#============================VOLATILITY FITTING==========================

vol_fitter = VolatilityFitter(final_df, beta_sabr=0.5)

# FIT ALL EXPIRIES
results = vol_fitter.fit_all(verbose=True)

params_df = vol_fitter.get_params_dataframe()
print(params_df.head())

# PLOTTING
vol_fitter.plot_fit(expiry_to_plot='15-Sep-2026')  

# ACCESS PARAMETERS
svi_params_15sep = results['15-Sep-2026']['svi_params']
sabr_params_15sep = results['15-Sep-2026']['sabr_params']

print("SVI Parameters by Expiry:")
for expiry, res in results.items():
    print(f"{expiry}: a={res['svi_a']:.4f}, b={res['svi_b']:.4f}, rho={res['svi_rho']:.4f}, sigma={res['svi_sigma']:.4f}")


# TENOR INTERPOLATION
interpolator = TenorInterpolator(results)

interpolator.plot_parameter_evolution()                            # Visualize how parameters evolve with time
strikes_grid = np.linspace(20000, 27000, 50)                       # Define your grid

T_min = interpolator.T_values.min()  
T_max = interpolator.T_values.max()  
T_grid = np.linspace(T_min, T_max, 50)                             # Only within your fitted range

interpolator.plot_surface_3d(strikes_grid, T_grid)                 # Plot the 3D Surface
interpolator.plot_heatmap(strikes_grid, T_grid)                    # Plot the Heatmap

T_42days = 42 / 365.0                                              # Query a specific point (e.g., 42 days, strike 24500)
iv_at_point = interpolator.get_vol(strike=24500.0, T=T_42days)
print(f"IV for Strike 24500, 42 days: {iv_at_point:.4f}")