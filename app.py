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

# Функция для умного расчёта оптимальных цен (оптимизация ПРИБЫЛИ)
def calculate_optimal_prices(df):
    """Рассчитывает оптимальные цены на основе линейной регрессии с учётом себестоимости"""
    results = []
    
    for product in df['product'].unique():
        product_df = df[df['product'] == product].copy()
        
        # Берём себестоимость (она одинаковая для товара)
        cost = product_df['cost'].iloc[0]
        
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
            # Оптимизируем ПРИБЫЛЬ: (price - cost) * (a + b*price + c*comp)
            # Оптимальная цена: price_opt = (a - b*cost + c*comp) / (-2*b)
            optimal_price = (a - b_price * cost + c_competitor * avg_competitor_price) / (-2 * b_price)
            
            # Ограничиваем цену разумными пределами
            min_price = cost * 1.1  # не ниже себестоимости + 10%
            max_price = product_df['our_price'].max() * 1.3
            optimal_price = max(min_price, min(max_price, optimal_price))
        else:
            # Если зависимость странная, оставляем текущую цену
            optimal_price = product_df['our_price'].mean()
        
        results.append({
            'product': product,
            'cost': round(cost, 2),
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
                # Себестоимость 50-70% от цены
                cost = base_price * np.random.uniform(0.5, 0.7)
                
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
                        'cost': round(cost, 2),
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
tab1, tab2, tab3 = st.tabs(["📊 Данные", "💰 Рекомендации (Прибыль)", "📈 Графики"])

with tab1:
    st.subheader("История продаж")
    st.dataframe(df.head(50))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего записей", len(df))
    col2.metric("Товаров", df['product'].nunique())
    col3.metric("Средние продажи", f"{df['sales'].mean():.0f}")
    col4.metric("Средняя себестоимость", f"{df['cost'].mean():.0f}")

with tab2:
    st.subheader("📊 Рекомендации на основе МАШИННОГО ОБУЧЕНИЯ")
    st.caption("Алгоритм оптимизирует ПРИБЫЛЬ с учётом себестоимости: (цена - себестоимость) × спрос")
    
    # Применяем умный алгоритм
    optimal_prices = calculate_optimal_prices(df)
    
    # Считаем текущие и новые показатели
    results = []
    for _, row in optimal_prices.iterrows():
        product = row['product']
        product_df = df[df['product'] == product]
        cost = row['cost']
        
        current_price = product_df['our_price'].mean()
        optimal_price = row['optimal_price']
        
        # Берём среднюю цену конкурента
        avg_competitor = product_df['competitor_price'].mean()
        
        # Обучаем модель для прогноза
        X = product_df[['our_price', 'competitor_price']]
        y = product_df['sales']
        model = LinearRegression()
        model.fit(X, y)
        
        a = model.intercept_
        b = model.coef_[0]
        c_competitor = model.coef_[1]
        
        # Функция прибыли
        def calc_profit(price, comp_price, cost_val, a_val, b_val, c_val):
            sales_pred = a_val + b_val * price + c_val * comp_price
            return (price - cost_val) * max(0, sales_pred)
        
        # Текущая прибыль
        current_profit = calc_profit(current_price, avg_competitor, cost, a, b, c_competitor)
        # Прогнозируемая прибыль
        optimal_profit = calc_profit(optimal_price, avg_competitor, cost, a, b, c_competitor)
        
        growth = ((optimal_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
        
        # Текущая и новая маржинальность
        margin_current = (current_price - cost) / current_price * 100
        margin_optimal = (optimal_price - cost) / optimal_price * 100
        
        results.append({
            'product': product,
            'cost': round(cost, 2),
            'current_price': round(current_price, 2),
            'optimal_price': round(optimal_price, 2),
            'change_%': round((optimal_price - current_price) / current_price * 100, 1),
            'margin_current_%': round(margin_current, 1),
            'margin_optimal_%': round(margin_optimal, 1),
            'current_profit': round(current_profit, 0),
            'optimal_profit': round(optimal_profit, 0),
            'growth_%': round(growth, 1)
        })
    
    results_df = pd.DataFrame(results)
    
    # Показываем результаты
    st.dataframe(results_df, use_container_width=True)
    
    # Итоговые метрики
    total_current_profit = results_df['current_profit'].sum()
    total_optimal_profit = results_df['optimal_profit'].sum()
    total_growth = (total_optimal_profit - total_current_profit) / total_current_profit * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Текущая прибыль", f"{total_current_profit:,.0f}")
    col2.metric("Прогнозируемая прибыль", f"{total_optimal_profit:,.0f}")
    
    if total_growth > 0:
        col3.metric("Рост прибыли", f"+{total_growth:.1f}%", delta="📈 Положительный")
    else:
        col3.metric("Рост прибыли", f"{total_growth:.1f}%", delta="⚠️ Требуется настройка")
    
    # Дополнительная информация
    with st.expander("ℹ️ Как работает алгоритм"):
        st.markdown("""
        **Формула оптимизации прибыли:**
        **Оптимальная цена:**
        
**Где:**
- `a` — базовый спрос
- `b` — эластичность (отрицательная: чем выше цена, тем ниже спрос)
- `c` — влияние цены конкурента

**Ограничения:**
- Цена не может быть ниже себестоимости + 10%
- Цена не может быть выше 130% от максимальной исторической цены
""")

with tab3:
    st.subheader("Зависимость продаж от цены")
    
    product = st.selectbox("Выберите товар", df['product'].unique())
    product_df = df[df['product'] == product]
    cost = product_df['cost'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(product_df['our_price'], product_df['sales'], 
                        c=product_df['competitor_price'], cmap='coolwarm', alpha=0.6)
    
    # Добавляем линию тренда
    z = np.polyfit(product_df['our_price'], product_df['sales'], 1)
    p = np.poly1d(z)
    ax.plot(product_df['our_price'].sort_values(), 
            p(product_df['our_price'].sort_values()), 
            "r--", alpha=0.8, label="Тренд спроса")
    
    # Вертикальная линия себестоимости
    ax.axvline(x=cost, color='green', linestyle='--', alpha=0.7, label=f'Себестоимость ({cost:.0f})')
    
    ax.set_xlabel("Наша цена")
    ax.set_ylabel("Продажи")
    ax.set_title(f"{product}: эластичность спроса (себестоимость = {cost:.0f})")
    ax.legend()
    plt.colorbar(scatter, label='Цена конкурента')
    st.pyplot(fig)
    
    st.caption("🔴 Красная линия — тренд спроса (чем ниже цена, тем выше продажи)")
    st.caption("🟢 Зелёная линия — себестоимость (цена не может быть ниже)")
    st.caption("🔵 Синие точки — низкая цена конкурента, 🔴 Красные — высокая")