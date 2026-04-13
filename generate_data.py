import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_days = 100
products = ['Товар A', 'Товар B', 'Товар C']

data = []
start_date = datetime(2024, 1, 1)

for product in products:
    base_price = np.random.choice([100, 200, 300])
    
    for day in range(n_days):
        date = start_date + timedelta(days=day)
        competitor_price = base_price * np.random.uniform(0.8, 1.2)
        our_price = base_price * (1 + np.random.normal(0, 0.05))
        price_effect = -0.5 * our_price
        competition_effect = 0.3 * (competitor_price - our_price)
        base_demand = 150
        sales = max(0, int(base_demand + price_effect + competition_effect + np.random.normal(0, 10)))
        
        data.append({
            'date': date,
            'product': product,
            'our_price': round(our_price, 2),
            'competitor_price': round(competitor_price, 2),
            'sales': sales
        })

df = pd.DataFrame(data)
df.to_csv('sales_data.csv', index=False)
print(f"Сгенерировано {len(df)} записей")
print(df.head())