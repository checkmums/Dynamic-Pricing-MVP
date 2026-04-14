import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('sales_data.csv')

print("📊 Умный алгоритм динамического ценообразования (оптимизация ПРИБЫЛИ)")
print("="*60)

results = []

for product in df['product'].unique():
    print(f"\n🔍 Анализ товара: {product}")
    
    product_df = df[df['product'] == product].copy()
    cost = product_df['cost'].iloc[0]  # себестоимость (одинаковая для товара)
    
    print(f"   Себестоимость: {cost:.2f}")
    
    # Обучаем модель спроса
    X = product_df[['our_price', 'competitor_price']]
    y = product_df['sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    a = model.intercept_
    b = model.coef_[0]  # влияние нашей цены
    c = model.coef_[1]  # влияние цены конкурента
    
    avg_competitor_price = product_df['competitor_price'].mean()
    
    if b < 0:
        # Оптимизируем ПРИБЫЛЬ, а не выручку
        # Прибыль = (цена - себестоимость) * спрос
        # Прибыль = (price - cost) * (a + b*price + c*comp_price)
        # Производная = 0:
        # price_opt = (a - b*cost + c*comp_price) / (-2*b)
        
        optimal_price = (a - b * cost + c * avg_competitor_price) / (-2 * b)
        
        # Ограничения
        min_price = cost * 1.1  # не ниже себестоимости + 10%
        max_price = product_df['our_price'].max() * 1.3
        optimal_price = max(min_price, min(max_price, optimal_price))
        
        current_price = product_df['our_price'].mean()
        
        # Прогноз прибыли
        def profit(price, comp_price, cost, a, b, c):
            sales_pred = a + b * price + c * comp_price
            return (price - cost) * max(0, sales_pred)
        
        current_profit = profit(current_price, avg_competitor_price, cost, a, b, c)
        optimal_profit = profit(optimal_price, avg_competitor_price, cost, a, b, c)
        
        growth = (optimal_profit - current_profit) / current_profit * 100 if current_profit > 0 else 0
        
        results.append({
            'product': product,
            'cost': round(cost, 2),
            'current_price': round(current_price, 2),
            'optimal_price': round(optimal_price, 2),
            'price_change_%': round((optimal_price - current_price) / current_price * 100, 1),
            'margin_current': round((current_price - cost) / current_price * 100, 1),
            'margin_optimal': round((optimal_price - cost) / optimal_price * 100, 1),
            'current_profit': round(current_profit, 0),
            'optimal_profit': round(optimal_profit, 0),
            'growth_%': round(growth, 1)
        })
        
        print(f"   Текущая цена: {current_price:.2f} → Оптимальная: {optimal_price:.2f} ({round((optimal_price - current_price)/current_price*100, 1)}%)")
        print(f"   Текущая маржа: {round((current_price - cost)/current_price*100, 1)}% → Оптимальная: {round((optimal_price - cost)/optimal_price*100, 1)}%")
        print(f"   Прибыль: {current_profit:,.0f} → {optimal_profit:,.0f} (рост {growth:.1f}%)")
    else:
        print(f"   ⚠️ Странная зависимость: цена не снижает спрос")
        results.append({
            'product': product,
            'cost': round(cost, 2),
            'current_price': round(product_df['our_price'].mean(), 2),
            'optimal_price': round(product_df['our_price'].mean(), 2),
            'price_change_%': 0,
            'margin_current': 0,
            'margin_optimal': 0,
            'current_profit': 0,
            'optimal_profit': 0,
            'growth_%': 0
        })

print("\n" + "="*60)
print("📈 ИТОГОВЫЙ ОТЧЁТ ПО ПРИБЫЛИ:")
print("="*60)

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

total_current = result_df['current_profit'].sum()
total_optimal = result_df['optimal_profit'].sum()
total_growth = (total_optimal - total_current) / total_current * 100

print("\n" + "="*60)
print(f"💰 ОБЩАЯ ПРИБЫЛЬ: {total_current:,.0f} → {total_optimal:,.0f}")
print(f"📈 РОСТ ПРИБЫЛИ: {total_growth:.1f}%")
print("="*60)