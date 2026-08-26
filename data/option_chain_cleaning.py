import numpy as np
import pandas as pd

class OptionChainCleaner:

    def _strip_data(self, df):
        strike_idx = df.columns.get_loc('STRIKE')

        df_calls = pd.DataFrame(df.iloc[:, :strike_idx].copy())                     # calls dataframe
        df_calls['strike'] = df['STRIKE']
        df_calls['option_type'] = 'Call'
        
        df_puts = pd.DataFrame(df.iloc[:, strike_idx+1:].copy())                    # puts dataframe
        df_puts['strike'] = df['STRIKE']
        df_puts.columns = [col.replace('.1', '') for col in df_puts.columns]
        df_puts['option_type'] = 'Put'
        
        df_long = pd.concat([df_calls, df_puts], ignore_index=True)                  # Concatenate the two dataframes

        return df_long

    def _clean_data(self, df_long):
        # =================================
        # Drop Garbage Columns
        cols_to_drop = ['Unnamed: 0', 'OI', 'CHNG IN OI', 'CHNG', 'BID QTY', 'ASK QTY', 'Unnamed: 22', 'LTP']
        df_long = df_long.drop(cols_to_drop, axis=1)

        # =================================
        # Renaming features
        df_long = df_long.rename(columns={'VOLUME': 'volume', 
                                            'IV': 'iv_market',
                                            'BID': 'bid',
                                            'ASK': 'ask'})

        # ==================================
        # Convert to Numeric / Float
        for col in ['bid', 'ask', 'strike']:
            df_long[col] = pd.to_numeric(df_long[col].astype(str).str.replace(",", ""), errors='coerce')


        # ==================================
        # Replacing blank spaces with 0
        df_long = df_long.replace('-', np.nan)
        for col in ['volume', 'iv_market']:
            df_long[col] = df_long[col].fillna(0)

        return df_long


    def _add_new_columns(self, df_long, file_name, spot, rate):
        # 1. expiry 
        parts = file_name.replace(".csv", "").split("-")                             # Remove the extension and split by dash
        expiry = "-".join(parts[4:7])          
        df_long['expiry'] = expiry

        # 2. T
        def compute_time_to_expiry(expiry_series, valuation_date='26-Aug-2026'):
            expiry = pd.to_datetime(expiry_series).dt.normalize()                    # Convert to datetime and normalize to midnight
            valuation = pd.to_datetime(valuation_date).normalize()                   # Convert valuation date to datetime and normalize
            delta_days = (expiry - valuation).dt.days                                # Calculate days difference (element-wise vectorized)
            T = delta_days / 365.0                                                   # Convert to years (ACT/365)
            T = T.clip(lower=1e-6)                                                   # Safety: Replace T <= 0 with a tiny epsilon (1e-6) to avoid division by zero
            return T
        
        df_long['T'] = compute_time_to_expiry(df_long['expiry'], valuation_date='26-Aug-2026')

        # 3. mid price
        df_long['mid_price'] = (df_long['bid'] + df_long['ask']) / 2

        # 4. underlying spot
        df_long['underlying_price'] = spot

        # 5. rate
        df_long['rate'] = rate

        return df_long


    def _add_filters(self, df_long):
        # Apply FILTERS
        df_long = df_long[
            (df_long['bid'] > 0) &
            (df_long['ask'] > 0) &
            (df_long['mid_price'] > 0) &
            (df_long['strike'] > 0) &
            (df_long['T'] > 0)
        ].copy()

        # ===========================================
        # Remove strikes with absurdly wide spreads (illiquid garbage)
        df_long = df_long[
            (df_long['ask'] / df_long['bid'] < 5.0)
        ].copy()

        return df_long

    
    def clean_option_chain(self, file_path, spot, rate):

        df = pd.read_csv(file_path, header=1)

        df_stripped = self._strip_data(df)
        df_cleaned = self._clean_data(df_stripped)
        df_added = self._add_new_columns(df_cleaned, file_path, spot, rate)
        df_filtered = self._add_filters(df_added)

        final_df = df_filtered
        return final_df
