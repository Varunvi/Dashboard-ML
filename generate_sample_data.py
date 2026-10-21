import pandas as pd
import numpy as np

# Generate realistic monthly crude oil data (2012-01 to 2024-06) ~150 months
np.random.seed(42)
dates = pd.date_range(start='2012-01-01', end='2024-06-01', freq='MS')
n = len(dates)

# Real-world benchmark trends for WTI/Brent:
# 2012-2014: High prices (~95-110)
# 2014-2016: Collapse (~105 down to 35)
# 2016-2019: Moderate recovery (~45 to 75)
# 2020: Covid plunge (drop to 20-30), high volatility
# 2021-2022: Recovery & Ukraine spike (up to 120)
# 2023-2024: Stabilization (70-85)

prices = []
demands = []
usd_indices = []
opec_policies = []

price = 100.0
demand = 90.0
usd = 80.0

for d in dates:
    yr = d.year
    m = d.month
    
    # Trend guidelines
    if yr <= 2013:
        target_price = 102.0
        target_demand = 91.5
        target_usd = 80.5
        policy = np.random.choice(['No Change', 'Increase'], p=[0.7, 0.3])
    elif yr == 2014:
        if m < 7:
            target_price = 105.0
            policy = 'No Change'
        else:
            target_price = 60.0
            policy = 'Increase'
        target_demand = 93.0
        target_usd = 85.0
    elif yr in [2015, 2016]:
        target_price = 45.0 if yr == 2015 else 43.0
        if yr == 2016 and m >= 11:
            policy = 'Cut'
            target_price = 52.0
        else:
            policy = np.random.choice(['No Change', 'Increase'], p=[0.6, 0.4])
        target_demand = 95.0
        target_usd = 96.0
    elif yr in [2017, 2018, 2019]:
        target_price = 60.0 + (yr - 2017) * 4.0
        target_demand = 98.0 + (yr - 2017) * 1.0
        target_usd = 93.0 + np.random.normal(0, 2)
        policy = np.random.choice(['No Change', 'Cut', 'Increase'], p=[0.6, 0.3, 0.1])
    elif yr == 2020:
        if m in [3, 4, 5]:
            target_price = 22.0
            target_demand = 83.0
            policy = 'Cut'
        else:
            target_price = 42.0
            target_demand = 90.0
            policy = 'Cut'
        target_usd = 98.0
    elif yr == 2021:
        target_price = 68.0
        target_demand = 97.0
        target_usd = 92.0
        policy = np.random.choice(['No Change', 'Increase'], p=[0.6, 0.4])
    elif yr == 2022:
        if m in [3, 4, 5, 6]:
            target_price = 112.0
        else:
            target_price = 88.0
        target_demand = 99.5
        target_usd = 104.0
        policy = np.random.choice(['No Change', 'Cut'], p=[0.7, 0.3])
    else: # 2023-2024
        target_price = 78.0
        target_demand = 102.0
        target_usd = 103.5
        policy = np.random.choice(['Cut', 'No Change'], p=[0.5, 0.5])

    price = 0.75 * price + 0.25 * target_price + np.random.normal(0, 3.2)
    demand = 0.8 * demand + 0.2 * target_demand + np.random.normal(0, 0.6)
    usd = 0.85 * usd + 0.15 * target_usd + np.random.normal(0, 0.8)
    
    prices.append(round(price, 2))
    demands.append(round(demand, 2))
    usd_indices.append(round(usd, 2))
    opec_policies.append(policy)

df = pd.DataFrame({
    'Date': dates.strftime('%d-%m-%Y'),
    'Demand (Mbbl/day)': demands,
    'USD Index': usd_indices,
    'Price (USD/bbl)': prices,
    'OPEC Policy': opec_policies
})

# Save to Excel
df.to_excel("Oil prices 2.xlsx", index=False)
print(f"Created Oil prices 2.xlsx with {len(df)} rows.")
