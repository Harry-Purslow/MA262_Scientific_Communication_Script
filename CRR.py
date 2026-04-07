#!/usr/bin/env python3

import yfinance as yf
import numpy as np
import pandas as pd
import math

# Forms the binomial tree and then calculates the value of X(0) as discussed in the paper. 
def crr_binomial_tree(S_0, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    X_0_holding = np.zeros(N + 1)
    for i in range(N + 1):
        X_0_holding[i] = (math.comb(N,i)) * (p**i) * ((1-p)**(N-i))*np.maximum((S_0 * (u ** (i)) * (d ** (N-i)))-K,0)
    X_0 = np.exp(-r*T)*(sum(X_0_holding))
    return X_0

# Retrieves data about the NVIDIA stock. 
nvda = yf.Ticker("NVDA")
hist = nvda.history(period="1y")
S0 = hist['Close'].iloc[-1]
options = nvda.option_chain('2026-04-20')
call_options = options.calls
Strike = call_options['strike'][0:15] # Strike prices of the first 15 options with expiry date April 20
Market_Ask = call_options['ask'][0:15]# Ask prices of the first 15 options with expiry date April 20

# Parameters for the April 20, 2026 expiration date (Date coded is April 7)

T = 13 / 365.0 
r = 0.037297 

# Assume risk free interest rate is 3.7297% (This value is chosen as the most accurate risk-free
# interest rate at the time of coding (07/04). Taken from the Bank of England's website 
# https://www.bankofengland.co.uk/markets/sonia-benchmark)

N = 100 # 100 time steps


# Calculating historical volatility
u_i = np.log(hist['Close'] / hist['Close'].shift(1))
u_i = np.array([x for x in u_i if not np.isnan(x)]) # get rid of any nan results from the u_i

u_bar = u_i.mean() # mean of the u_i
sigma = sum(np.sqrt((1/(N-1))*(u_i-u_bar)**2)) # historical volatility 
results = []

for i in range(len(Strike)):
    K = Strike[i]
    ask = Market_Ask[i]
    
    # Calculate our model price
    model_price = crr_binomial_tree(S0, K, T, r, sigma, N)
    
    results.append({
        "Strike": K,
        "Market Ask": ask,
        "Model Price": round(model_price, 2),
        "Difference": round(model_price - ask, 2)
    })

# Display as a clean table

df_results = pd.DataFrame(results)
print("\n--- NVDA CRR Model vs. Market Comparison ---")
print(df_results.to_string(index=False))
print(f"\nNote: Calculations based on S0: ${S0:.2f} and Sigma: {sigma:.2%}")


# Notes: Based on the historical values of NVIDIA's stock price, the historical volatility has been
# calculated to be 44.13%. This is promising as the volatility of NVIDIA's stock tends to be around 40%.  
