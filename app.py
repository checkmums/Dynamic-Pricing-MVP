import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dynamic Pricing MVP", layout="wide")

st.title("🚀 Алгоритм динамического ценообразования")
st.markdown("---")

# Функция для умного расчёта оптимальных цен
def calculate_optimal_prices(df):
    """Рассчитывает оптимальные цены на основе линейной регрессии"""
    results = []
    
    for product in df['product'].unique():
        product_df = df[df['product'] == product].copy()
        
        # Обучаем модель
        X = product_df[['our_price', 'competitor_price']]
        y = product_df['sales']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Получаем коэффициенты
        a = model.intercept_
        b_price = model.coef_[0]
        c_competitor = model.coef_[1]
        
        avg_competitor_price = product_df['competitor_price'].mean()
        
        # Если зависимость корректная (цена снижает спрос)
        if b_price < 0:
            optimal_price = -(a + c_competitor * avg_competitor_price) / (2 * b_price)
            
            # Ограничиваем цену разумными пределами
            min_price = product_df['our_price'].min() * 0.7
            max_price = product_df['our_price'].max() * 1.3
            optimal_price = max(min_price, min(max_price, optimal_price))
        else:
            # Если зависимость странная, оставляем текущую цену
            optimal_price = product_df['our_price'].mean()
        
        results.append({
            'product': product,
            'optimal_price': round(optimal_price, 2)
        })
    
    return pd.DataFrame(results)

# Боковая панель
with st.sidebar:
    st.header("Управление")
    
    if st.button("🔄 Сгенерировать новые данные"):
        with st.spinner("Генерация данных..."):
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
                    sales = max(0, int(150 + price_effect + competition_effect + np.random.normal(0, 10)))
                    data.append({
                        'date': date, 'product': product, 
                        'our_price': round(our_price, 2),
                        'competitor_price': round(competitor_price, 2), 
                        'sales': sales
                    })
            
            df = pd.DataFrame(data)
            st.session_state['data'] = df
            st.success("✅ Данные сгенерированы!")

# Загрузка или генерация данных
if 'data' not in st.session_state:
    st.info("👈 Нажмите 'Сгенерировать новые данные' в боковой панели")
    st.stop()

df = st.session_state['data']

# Вкладки
tab1, tab2, tab3 = st.tabs(["📊 Данные", "💰 Рекомендации (Умный алгоритм)", "📈 Графики"])

with tab1:
    st.subheader("История продаж")
    st.dataframe(df.head(50))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего записей", len(df))
    col2.metric("Товаров", df['product'].nunique())
    col3.metric("Средние продажи", f"{df['sales'].mean():.0f}")

with tab2:
    st.subheader("📊 Рекомендации на основе МАШИННОГО ОБУЧЕНИЯ")
    st.caption("Алгоритм анализирует эластичность спроса и находит оптимальную цену для максимизации выручки")
    
    # Применяем умный алгоритм
    optimal_prices = calculate_optimal_prices(df)
    
    # Считаем текущие и новые показатели
    results = []
    for _, row in optimal_prices.iterrows():
        product = row['product']
        product_df = df[df['product'] == product]
        
        current_price = product_df['our_price'].mean()
        optimal_price = row['optimal_price']
        
        # Прогнозируем продажи при новой цене (упрощённо)
        # Берём среднюю цену конкурента
        avg_competitor = product_df['competitor_price'].mean()
        
        # Обучаем модель для прогноза
        X = product_df[['our_price', 'competitor_price']]
        y = product_df['sales']
        model = LinearRegression()
        model.fit(X, y)
        
        # Прогноз при текущей цене
        current_sales = model.predict([[current_price, avg_competitor]])[0]
        # Прогноз при оптимальной цене
        optimal_sales = model.predict([[optimal_price, avg_competitor]])[0]
        
        current_revenue = current_price * max(0, current_sales)
        optimal_revenue = optimal_price * max(0, optimal_sales)
        
        growth = ((optimal_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        
        results.append({
            'product': product,
            'current_price': round(current_price, 2),
            'optimal_price': round(optimal_price, 2),
            'change_%': round((optimal_price - current_price) / current_price * 100, 1),
            'current_revenue': round(current_revenue, 0),
            'optimal_revenue': round(optimal_revenue, 0),
            'growth_%': round(growth, 1)
        })
    
    results_df = pd.DataFrame(results)
    
    # Показываем результаты
    st.dataframe(results_df, use_container_width=True)
    
    # Итоговые метрики
    total_current = results_df['current_revenue'].sum()
    total_optimal = results_df['optimal_revenue'].sum()
    total_growth = (total_optimal - total_current) / total_current * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Текущая выручка", f"{total_current:,.0f}")
    col2.metric("Прогнозируемая выручка", f"{total_optimal:,.0f}")
    
    if total_growth > 0:
        col3.metric("Рост выручки", f"+{total_growth:.1f}%", delta="📈 Положительный")
    else:
        col3.metric("Рост выручки", f"{total_growth:.1f}%", delta="⚠️ Требуется настройка")

with tab3:
    st.subheader("Зависимость продаж от цены")
    
    product = st.selectbox("Выберите товар", df['product'].unique())
    product_df = df[df['product'] == product]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(product_df['our_price'], product_df['sales'], 
                        c=product_df['competitor_price'], cmap='coolwarm', alpha=0.6)
    
    # Добавляем линию тренда
    z = np.polyfit(product_df['our_price'], product_df['sales'], 1)
    p = np.poly1d(z)
    ax.plot(product_df['our_price'].sort_values(), 
            p(product_df['our_price'].sort_values()), 
            "r--", alpha=0.8, label="Тренд (чем ниже цена, тем выше продажи)")
    
    ax.set_xlabel("Наша цена")
    ax.set_ylabel("Продажи")
    ax.set_title(f"{product}: эластичность спроса")
    ax.legend()
    plt.colorbar(scatter, label='Цена конкурента')
    st.pyplot(fig)
    
    st.caption("🔴 Красная линия — тренд: при снижении цены продажи растут")
    st.caption("🔵 Синие точки — низкая цена конкурента, 🔴 Красные — высокая")

st.markdown("---")
st.caption("MVP умного динамического ценообразования на основе машинного обучения")