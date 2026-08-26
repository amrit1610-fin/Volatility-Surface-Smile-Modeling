from data.data_cleaning import CleanedOptionChain

data_folder_path = "data/option_chains"
spot = 24281.30
rate = 0.07

engine = CleanedOptionChain(data_folder_path, spot, rate)
data = engine.process_all()
print(data.tail())
print(data.shape)