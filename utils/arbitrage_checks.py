import numpy as np

class ArbitrageChecks:

    def check_butterfly_arb(self, df_slice, tolerance=1e-6):
        df = df_slice.copy()
    
        # --- Step 1: Create a temporary 'call_price' column ---
        # For calls: use the existing mid_price
        # For puts: convert to synthetic call using Put-Call parity:
        # C = P + S - K * exp(-r*T)
        
        S = df['underlying_price'].iloc[0]  # Assumes same spot for all rows in slice
        r = df['rate'].iloc[0]              # Assumes same rate for all rows in slice
        T = df['T'].iloc[0]                 # Assumes same T for all rows in slice
        
        def get_synthetic_call(row):
            if row['option_type'] == 'call':
                return row['mid_price']
            else:
                # Put-Call parity
                return row['mid_price'] + S - row['strike'] * np.exp(-r * T)
        
        df['call_price'] = df.apply(get_synthetic_call, axis=1)
        
        # --- Step 2: Sort by strike and check convexity ---
        df = df.sort_values('strike').reset_index(drop=True)
        strikes = df['strike'].values
        call_prices = df['call_price'].values
        violations = []
        
        for i in range(len(strikes) - 2):
            k1, k2, k3 = strikes[i], strikes[i+1], strikes[i+2]
            c1, c2, c3 = call_prices[i], call_prices[i+1], call_prices[i+2]
            
            butterfly = c1 - 2*c2 + c3
            
            if butterfly < -tolerance:
                violations.append({
                    'K1': k1, 'K2': k2, 'K3': k3,
                    'butterfly_value': butterfly
                })
                
                # --- Step 3: Correct the violation ---
                # Raise the middle price to restore convexity
                corrected_call_price = (c1 + c3) / 2 + tolerance
                
                # --- Step 4: Map correction back to original mid_price ---
                # Find the original row for this strike
                idx = df[df['strike'] == k2].index[0]
                
                if df.loc[idx, 'option_type'] == 'call':
                    # If it was a call, simply update the mid_price
                    df.loc[idx, 'mid_price'] = corrected_call_price
                else:
                    # If it was a put, convert back via reverse Put-Call parity:
                    # P = C - S + K * exp(-r*T)
                    corrected_put_price = corrected_call_price - S + k2 * np.exp(-r * T)
                    # Ensure the put price doesn't go negative (clip at 0.01)
                    corrected_put_price = max(corrected_put_price, 0.01)
                    df.loc[idx, 'mid_price'] = corrected_put_price
                
                # Update the 'call_price' column with the corrected value for next iterations
                df.loc[idx, 'call_price'] = corrected_call_price
        
        # --- Step 5: Drop the temporary column and return ---
        df.drop(columns=['call_price'], inplace=True)
        
        return df, violations


    def check_calendar_arb(self, df):
        df = df.copy()

        df['total_var'] = df['iv_calc']**2 * df['T']
        violations = []

        for strike in df['strike'].unique():
            slice_df = df[df['strike'] == strike].sort_values('T')
            if len(slice_df) < 2:
                continue

            T_vals = slice_df['T'].values
            w_vals = slice_df['total_var'].values

            for i in range(len(T_vals) - 1):
                if w_vals[i+1] < w_vals[i] - 1e-6:
                    violations.append({'strike': strike, 'T1': T_vals[i], 'T2': T_vals[i+1]})
                    # Correction: Set the later total variance equal to the earlier + small epsilon
                    new_w = w_vals[i] + 1e-6
                    new_iv = np.sqrt(new_w / T_vals[i+1])
                    # Update the main DataFrame
                    mask = (df['strike'] == strike) & (df['T'] == T_vals[i+1])
                    df.loc[mask, 'iv_calc'] = new_iv
                    df.loc[mask, 'total_var'] = new_w

        return df, violations


