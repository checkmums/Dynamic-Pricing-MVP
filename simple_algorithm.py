import pandas as pd

def recommend_price(row):
    our_price = row['our_price']
    competitor_price = row['competitor_price']
    
    if competitor_price < our_price * 0.9:
        return our_price * 0.99
    if row['sales'] < 100:
        return our_price * 0.97
    return our_price

df = pd.read_csv('sales_data.csv')
df['recommended_price'] = df.apply(recommend_price, axis=1)
df['old_revenue'] = df['our_price'] * df['sales']
df['new_revenue'] = df['recommended_price'] * df['sales']

total_old = df['old_revenue'].sum()
total_new = df['new_revenue'].sum()
growth = (total_new - total_old) / total_old * 100

print(f"\n📊 Результаты:")
print(f"Старая выручка: {total_old:,.0f}")
print(f"Новая выручка: {total_new:,.0f}")
print(f"Рост: {growth:.1f}%")
print("\n📝 Примеры рекомендаций:")
print(df[['product', 'our_price', 'competitor_price', 'sales', 'recommended_price']].head(10))