import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- 1. ROBUST DATA PARSER (No Cities) ---
def parse_sql_data(text):
    """
    Extracts Date, Time, and Price from SQL text.
    Ignores Address/City to prevent errors.
    """
    parsed_data = []
    
    # Split file into rows based on the SQL structure "),"
    rows = text.split("),(")
    
    for row in rows:
        try:
            # Extract Date & Time (Standard format search)
            date_match = re.search(r"'(\d{2}-\d{2}-\d{4})'", row)
            time_match = re.search(r"'(\d{2}:\d{2})'", row)
            
            # Extract Price (Look for numbers with optional €)
            price_match = re.search(r"'(\d+[,.]\d+)\s*€?'", row)
            
            if date_match and time_match:
                entry = {
                    'Date': date_match.group(1),
                    'Time': time_match.group(1),
                    'Price': 0.0
                }
                
                # Process Price if found
                if price_match:
                    raw_price = price_match.group(1)
                    # Fix European format 12,50 -> 12.50
                    clean_price = raw_price.replace(',', '.')
                    entry['Price'] = float(clean_price)
                
                parsed_data.append(entry)
        except:
            continue
            
    return parsed_data

# --- 2. APP LAYOUT ---
st.set_page_config(page_title="Driver AI Lite", page_icon="🚕", layout="wide")
st.title("🚕 Driver AI: Strategic Planner")
st.markdown("### Revenue, Seasonality & Demand Prediction")

uploaded_file = st.file_uploader("Upload SQL Data", type=['sql'])

if uploaded_file is not None:
    string_data = uploaded_file.getvalue().decode("utf-8")
    
    with st.spinner('Training AI Model...'):
        data = parse_sql_data(string_data)
        
        if data:
            df = pd.DataFrame(data)
            
            # --- 3. DATA ENGINEERING ---
            # Create standard Datetime objects
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Datetime'])
            
            # Extract handy features
            df['Hour'] = df['Datetime'].dt.hour
            df['DayOfWeek'] = df['Datetime'].dt.day_name()
            df['DayNum'] = df['Datetime'].dt.dayofweek
            df['Month'] = df['Datetime'].dt.month_name()
            
            # --- 4. TRAIN ML MODEL ---
            # Prepare data: Count bookings per Day+Hour
            demand_data = df.groupby(['DayNum', 'Hour']).size().reset_index(name='Bookings')
            
            if not demand_data.empty:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(demand_data[['DayNum', 'Hour']], demand_data['Bookings'])
                st.success("✅ AI System Ready")
            
            # --- 5. DASHBOARD TABS (3 Tabs Only) ---
            tab1, tab2, tab3 = st.tabs([
                "📅 Yearly Planner", 
                "💰 Revenue Intelligence", 
                "🔮 AI Predictor"
            ])
            
            # --- TAB 1: YEARLY PLANNER ---
            with tab1:
                st.subheader("Yearly Seasonality Trends")
                
                monthly_counts = df['Month'].value_counts()
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December']
                monthly_counts = monthly_counts.reindex(month_order).dropna()
                
                if not monthly_counts.empty:
                    colA, colB = st.columns([2, 1])
                    with colA:
                        fig_m, ax_m = plt.subplots(figsize=(8, 4))
                        sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=ax_m, palette="viridis")
                        ax_m.set_xticklabels(monthly_counts.index, rotation=45)
                        ax_m.set_ylabel("Total Jobs")
                        st.pyplot(fig_m)
                    
                    with colB:
                        best_month = monthly_counts.idxmax()
                        st.success(f"**Busy Season:** {best_month}")
                        st.info("Plan maximum shifts during this month.")
                else:
                    st.warning("Not enough data for yearly trends.")

            # --- TAB 2: REVENUE ---
            with tab2:
                st.subheader("Financial Intelligence")
                
                total_rev = df['Price'].sum()
                avg_fare = df['Price'].mean()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Revenue", f"€{total_rev:,.2f}")
                c2.metric("Avg Fare per Trip", f"€{avg_fare:.2f}")
                
                hourly_price = df.groupby('Hour')['Price'].mean()
                if not hourly_price.empty:
                    best_price_hour = hourly_price.idxmax()
                    c3.metric("Highest Value Hour", f"{best_price_hour}:00")
                    
                    st.write("### 💸 Hourly Price Trends")
                    st.caption("Green line = Average price per trip at that hour.")
                    
                    fig_p, ax_p = plt.subplots(figsize=(10, 4))
                    sns.lineplot(x=hourly_price.index, y=hourly_price.values, ax=ax_p, marker='o', color='green', linewidth=2)
                    ax_p.set_xticks(range(0, 24))
                    ax_p.grid(True, linestyle='--')
                    st.pyplot(fig_p)

            # --- TAB 3: AI PREDICTION ---
            with tab3:
                st.subheader("Future Demand Simulator")
                
                col1, col2 = st.columns(2)
                with col1:
                    pred_day = st.selectbox("Select Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                    pred_hour = st.slider("Select Hour", 0, 23, 18)
                
                day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
                
                if not demand_data.empty:
                    # Machine Learning Prediction
                    prediction = model.predict([[day_map[pred_day], pred_hour]])[0]
                    
                    with col2:
                        st.metric("Forecasted Jobs", f"{int(prediction)}")
                        if prediction > demand_data['Bookings'].mean():
                            st.error("🔥 Busy Shift Expected")
                        else:
                            st.info("💤 Quiet Shift Expected")

        else:
            st.error("Could not parse data. Check SQL file.")