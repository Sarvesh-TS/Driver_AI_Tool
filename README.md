# 🚕 Driver AI: Predictive Analytics & Operating System

### Enterprise-grade Fleet Management & Forecasting Tool

**Driver AI** is a locally hosted analytics platform designed to transform raw SQL operational data into actionable strategic insights. It leverages **Machine Learning (Random Forest)** to predict future demand, analyze financial efficiency, and estimate traffic impact on trip durations.

---

## 🧠 Key Intelligence Modules

The application consists of 5 core intelligence modules:

### 1. 🔮 Demand AI (Forecasting)
* **Technology:** Random Forest Regressor
* **Function:** Predicts exact booking volumes for any future date and hour.
* **Output:** Classifies shifts as **"🔥 High Demand"** or **"💤 Low Demand"** to assist with driver rostering.
* **Visualization:** Weekly Heatmap (Day vs. Hour) to identify demand hotspots instantly.

### 2. 💰 Revenue Intelligence
* **Function:** Analyzes financial performance beyond simple totals.
* **Key Insight:** **"Rate per Km"** analysis identifies which hours yield the highest profit margins, optimizing fleet allocation for high-value trips rather than just high-volume trips.
* **Metrics:** Total Revenue, Average Fare, and Efficiency Curves.

### 3. ⏱️ Traffic & Duration AI
* **Technology:** Predictive Modeling (Distance + Time → Duration)
* **Function:** Estimates trip duration based on time-of-day traffic patterns.
* **Use Case:** Helps calculate true hourly wages by accounting for rush-hour delays.

### 4. ⚠️ Operational Risks
* **Function:** Monitors fleet health and stability.
* **Metrics:** Completion vs. Cancellation ratios and top-performing cities/locations.

### 5. 📅 Strategic Planner
* **Function:** Long-term seasonality analysis.
* **Output:** Identifies the busiest and slowest months of the year for maintenance scheduling and holiday planning.

---

## 🛠️ Tech Stack

* **Core Logic:** Python 3.x
* **Interface:** Streamlit (Web-based Dashboard)
* **Data Engineering:** Pandas & Regular Expressions (Robust SQL Parsing)
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`)
* **Visualization:** Seaborn & Matplotlib

---

## 🚀 How to Run Locally

### Prerequisites
Ensure you have Python installed.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Driver_AI_Tool.git](https://github.com/YOUR_USERNAME/Driver_AI_Tool.git)
cd Driver_AI_Tool