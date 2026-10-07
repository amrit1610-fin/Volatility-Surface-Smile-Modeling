# ==========================================
# ASSET SPECIFIC PARAMETERS
# ==========================================
ASSET_NAME = "NIFTY"
SPOT_PRICE = 24281.30
RISK_FREE_RATE = 0.07       # 7% INR risk-free rate
DIVIDEND_YIELD = 0.013      # 1.3% Nifty dividend yield

# ==========================================
# DATA CLEANING & FILTERING
# ==========================================
MONEYNESS_WINDOW = 0.08     # Filter options within +/- 8% of ATM
MIN_LIQUIDITY_IV = 0.01     # Drop IVs below 1%
MAX_LIQUIDITY_IV = 1.50     # Drop IVs above 150%

# ==========================================
# GRID & CALIBRATION SETTINGS
# ==========================================
# Strike grid boundaries for interpolation and Heston training
STRIKE_MIN = 23000
STRIKE_MAX = 25500
STRIKE_POINTS = 15

# Time grid boundaries (in years)
TIME_MIN = 0.05
TIME_MAX = 2.0
TIME_POINTS = 8