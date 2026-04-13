import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Загружаем данные
df = pd.read_csv('sales_data.csv')

print("📊 Умный алгоритм динамического ценообразования")
print("="*50)

results = []

for product in df['product'].unique():
    print(f"\n🔍 Анализ товара: {product}")
    
    # Берём данные только по этому товару
    product_df = df[df['product'] == product].copy()
    
    # Подготовка данных для регрессии
    # Предсказываем продажи от цены и цены конкурента
    X = product_df[['our_price', 'competitor_price']]
    y = product_df['sales']
    
    # Обучаем модель
    model = LinearRegression()
    model.fit(X, y)
    
    # Получаем коэффициенты
    a = model.intercept_  # базовый спрос
    b_price = model.coef_[0]  # влияние нашей цены (должно быть отрицательным)
    c_competitor = model.coef_[1]  # влияние цены конкурента
    
    print(f"   Спрос = {a:.1f} + ({b_price:.2f} × цена) + ({c_competitor:.2f} × цена конкурента)")
    
    # Оптимальная цена (производная от выручки = 0)
    # Выручка = цена × спрос = цена × (a + b*цена + c*цена_конкурента)
    # Производная: a + 2*b*цена + c*цена_конкурента = 0
    # Оптимальная цена = -(a + c*цена_конкурента) / (2*b)
    
    avg_competitor_price = product_df['competitor_price'].mean()
    
    if b_price < 0:  # если зависимость корректная (цена снижает спрос)
        optimal_price = -(a + c_competitor * avg_competitor_price) / (2 * b_price)
        
        # Ограничиваем цену разумными пределами
        min_price = product_df['our_price'].min() * 0.8
        max_price = product_df['our_price'].max() * 1.2
        optimal_price = max(min_price, min(max_price, optimal_price))
        
        # Текущая средняя цена
        current_price = product_df['our_price'].mean()
        
        # Считаем прогноз продаж
        current_sales_pred = a + b_price * current_price + c_competitor * avg_competitor_price
        optimal_sales_pred = a + b_price * optimal_price + c_competitor * avg_competitor_price
        
        current_revenue = current_price * max(0, current_sales_pred)
        optimal_revenue = optimal_price * max(0, optimal_sales_pred)
        
        growth = (optimal_revenue - current_revenue) / current_revenue * 100
        
        results.append({
            'product': product,
            'current_price': round(current_price, 2),
            'optimal_price': round(optimal_price, 2),
            'price_change': round((optimal_price - current_price) / current_price * 100, 1),
            'current_revenue': round(current_revenue, 0),
            'optimal_revenue': round(optimal_revenue, 0),
            'growth': round(growth, 1)
        })
        
        print(f"   Текущая цена: {current_price:.2f} → Оптимальная: {optimal_price:.2f} ({round((optimal_price - current_price)/current_price*100, 1)}%)")
        print(f"   Прогноз выручки: {current_revenue:,.0f} → {optimal_revenue:,.0f} (рост {growth:.1f}%)")
    else:
        print(f"   ⚠️ Странная зависимость: цена не снижает спрос (коэффициент {b_price:.2f})")
        print(f"   Оставляем текущую цену без изменений")

# Итоговая таблица
print("\n" + "="*50)
print("📈 ИТОГОВЫЙ ОТЧЁТ:")
print("="*50)

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

total_current = result_df['current_revenue'].sum()
total_optimal = result_df['optimal_revenue'].sum()
total_growth = (total_optimal - total_current) / total_current * 100

print("\n" + "="*50)
print(f"💰 ОБЩАЯ ВЫРУЧКА: {total_current:,.0f} → {total_optimal:,.0f}")
print(f"📈 ОБЩИЙ РОСТ: {total_growth:.1f}%")
print("="*50)