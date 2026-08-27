from data.data_cleaning import CleanedOptionChain
from utils.implied_vol import ImpliedVolatility

data_folder_path = "data/option_chains"
spot = 24281.30
rate = 0.07

data_engine = CleanedOptionChain(data_folder_path, spot, rate)
data = data_engine.process_all()

iv_engine = ImpliedVolatility()

data['iv_calc'] = data.apply(iv_engine.implement_implied_vol, axis=1)

print(data.head())
print(data.shape)

