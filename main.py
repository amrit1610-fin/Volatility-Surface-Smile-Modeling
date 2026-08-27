import pandas as pd
from data.data_cleaning import CleanedOptionChain
from utils.implied_vol import ImpliedVolatility
from utils.arbitrage_checks import ArbitrageChecks

# VARIABLES
data_folder_path = "data/option_chains"
spot = 24281.30
rate = 0.07

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

print(final_df.head())
