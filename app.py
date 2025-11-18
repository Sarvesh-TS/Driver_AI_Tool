import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- 1. BULLETPROOF DATA PARSER ---
def parse_sql_robust(text):
    """
    Parses SQL using Regex (safest method) and guarantees all columns exist.
    """
    matches = re.findall(r"\((.*?)\)", text)
    parsed_data = []
    
    for match in matches:
        parts = match.split(',')
        parts = [p.strip("' ") for p in parts]
        
        entry = {
            'Date': None, 'Time': None, 'Price': 0.0,
            'Distance': 0.0, 'Duration': 0.0,
            'Status': 'Unknown', 'City': 'Unknown'
        }
        
        if len(parts) >= 25:
            try:
                entry['Date'] = parts[-6]
                entry['Time'] = parts[-4]
                
                price_raw = parts[21].replace('€', '').replace(',', '.')
                if price_raw and price_raw.lower() != 'null':
                    entry['Price'] = float(price_raw)
                    
                dist_raw = parts[16]
                if dist_raw and dist_raw.lower() != 'null':
                    entry['Distance'] = float(dist_raw)
                    
                dur_raw = parts[17]
                if dur_raw and dur_raw.lower() != 'null':
                    entry['Duration'] = float(dur_raw)
                    
                entry['Status'] = parts[20]
                
                addr = parts[12]
                if ',' in addr:
                    entry['City'] = addr.split(',')[-1].strip()
                else:
                    entry['City'] = addr

                if entry['Date']:
                    parsed_data.append(entry)
            except:
                continue
                
    return parsed_data

# --- 2. APP CONFIG ---
st.set_page_config(page_title="Driver AI Ultra", page_icon="🚕", layout="wide")
st.title("🚕 Driver AI: The Complete Operating System")
st.markdown("### Forecasting: Demand, Revenue, Traffic & Risks")

uploaded_file = st.file_uploader("Upload SQL Data", type=['sql'])

if uploaded_file is not None:
    string_data = uploaded_file.getvalue().decode("utf-8")
    
    with st.spinner('Initializing AI Engines...'):
        data = parse_sql_robust(string_data)
        
        if data:
            df = pd.DataFrame(data)
            
            # --- 3. DATA ENGINEERING ---
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Datetime'])
            
            df['Hour'] = df['Datetime'].dt.hour
            df['DayNum'] = df['Datetime'].dt.dayofweek
            df['Month'] = df['Datetime'].dt.month_name()
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0)
            df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
            
            # --- 4. TRAINING AI MODELS ---
            demand_df = df.groupby(['DayNum', 'Hour']).size().reset_index(name='Bookings')
            model_demand = None
            if not demand_df.empty:
                model_demand = RandomForestRegressor(n_estimators=50, random_state=42)
                model_demand.fit(demand_df[['DayNum', 'Hour']], demand_df['Bookings'])

            traffic_df = df[df['Distance'] > 0].copy()
            model_traffic = None
            if not traffic_df.empty:
                model_traffic = RandomForestRegressor(n_estimators=50, random_state=42)
                model_traffic.fit(traffic_df[['Distance', 'Hour']], traffic_df['Duration'])
            
            st.success(f"✅ AI System Online. Processed {len(df)} rides.")

            # --- 5. DASHBOARD TABS (NOW 6 TABS) ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🔮 Demand AI", 
                "💰 Revenue AI", 
                "⏱️ Traffic AI", 
                "⚠️ Ops Status",
                "🗺️ Locations",
                "📅 Planner"
            ])
            
            # TAB 1: DEMAND
            with tab1:
                st.subheader("Future Demand Simulator")
                c1, c2 = st.columns(2)
                with c1:
                    pred_day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                    pred_hour = st.slider("Hour", 0, 23, 17)
                
                day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
                
                if model_demand:
                    val = model_demand.predict([[day_map[pred_day], pred_hour]])[0]
                    with c2:
                        st.metric("Expected Bookings", int(val))
                        if val > demand_df['Bookings'].mean(): 
                            st.error("🔥 High Demand")
                        else: 
                            st.info("💤 Low Demand")
                
                st.write("---")
                pivot = demand_df.pivot(index="DayNum", columns="Hour", values="Bookings")
                pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                fig_h, ax_h = plt.subplots(figsize=(10, 4))
                sns.heatmap(pivot, cmap="YlOrRd", ax=ax_h)
                st.pyplot(fig_h)

            # TAB 2: REVENUE
            with tab2:
                st.subheader("Financial Intelligence")
                col1, col2 = st.columns(2)
                col1.metric("Total Revenue", f"€{df['Price'].sum():,.2f}")
                col2.metric("Avg Fare", f"€{df['Price'].mean():.2f}")
                
                if df['Distance'].sum() > 0:
                    df['RatePerKm'] = df['Price'] / df['Distance']
                    clean_rate = df[(df['RatePerKm'] > 0) & (df['RatePerKm'] < 10)] 
                    
                    st.write("### 💶 Rate per Km Efficiency")
                    hourly_rate = clean_rate.groupby('Hour')['RatePerKm'].mean()
                    fig_r, ax_r = plt.subplots(figsize=(10, 3))
                    sns.lineplot(x=hourly_rate.index, y=hourly_rate.values, ax=ax_r, color='purple', marker='o')
                    ax_r.set_ylabel("€ per Km")
                    st.pyplot(fig_r)

            # TAB 3: TRAFFIC
            with tab3:
                st.subheader("⏱️ Trip Duration Predictor")
                tc1, tc2 = st.columns(2)
                with tc1:
                    dist_input = st.number_input("Trip Distance (km)", value=5.0)
                    time_input = st.slider("Traffic at Hour", 0, 23, 9)
                
                if model_traffic:
                    pred_duration = model_traffic.predict([[dist_input, time_input]])[0]
                    with tc2:
                        st.metric("Estimated Time", f"{int(pred_duration)} mins")
                    
                    st.write("### 🚦 Traffic Curve")
                    hours = range(0, 24)
                    times = model_traffic.predict([[dist_input, h] for h in hours])
                    fig_t, ax_t = plt.subplots(figsize=(10, 3))
                    ax_t.plot(hours, times, marker='o', color='orange')
                    ax_t.grid(True)
                    st.pyplot(fig_t)

            # TAB 4: OPS STATUS (Only Status Now)
            with tab4:
                st.subheader("⚠️ Operational Health")
                st.write("**Booking Status Breakdown**")
                st.bar_chart(df['Status'].value_counts())

            # TAB 5: LOCATIONS (New Dedicated Tab!)
            with tab5:
                st.subheader("🗺️ Geographic Hotspots")
                st.write("Top Performing Cities (Demand Source)")
                
                top_cities = df['City'].value_counts().head(10)
                st.bar_chart(top_cities)
                
                with st.expander("View Raw City Data"):
                    st.dataframe(df['City'].value_counts())

            # TAB 6: PLANNER
            with tab6:
                st.subheader("📅 Seasonality")
                monthly = df['Month'].value_counts()
                order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                monthly = monthly.reindex(order).dropna()
                st.bar_chart(monthly)

        else:
            st.error("Could not parse data. Please check file format.")