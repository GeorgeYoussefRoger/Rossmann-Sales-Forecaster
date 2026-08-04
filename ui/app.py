import streamlit as st
import requests
from datetime import date

st.set_page_config(layout="wide")
st.title("🛒 Rossmann Sales Forecaster")
st.write("Enter store information to get a sales forecast for the selected date.")
st.caption("This model was trained on historical data from 2013–2015.")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("🏪 Store Details")
    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
    store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])
    assortment = st.selectbox("Assortment", ["a", "b", "c"],
                                format_func=lambda x: {"a":"Basic","b":"Extra","c":"Extended"}[x])
    forecast_date = st.date_input("Forecast Date", min_value=date(2013, 1, 1), max_value=date(2015, 7, 31), value=date(2015, 6, 1))
    day_of_week = forecast_date.isoweekday()
    
with c2:
    st.subheader("📊 Historical Sales Data")
    lag_7 = st.number_input("Sales 1 Week Ago ($)", min_value=0.0, value=5200.0, step=50.0,
                            help = "Actual sales from exactly 7 days ago.")
    lag_14 = st.number_input("Sales 2 Weeks Ago ($)", min_value=0.0, value=5100.0, step=50.0,
                             help = "Actual sales from exactly 14 days ago.")
    rolling_7 = st.number_input("Last 7 Days Average ($)", min_value=0.0, value=5150.0, step=50.0,
                                help = "Average daily sales over the past week.")
    rolling_30 = st.number_input("Last 30 Days Average ($)", min_value=0.0, value=5000.0, step=50.0,
                                 help = "Average daily sales over the past month.")

with c3:
    st.subheader("🥊 Competition") 
    comp_distance = st.number_input("Distance to Competitor (m)", min_value=0.0, value=1000.0, step=10.0)
    comp_month = st.number_input("Competitor Open Since Month", min_value=1, max_value=12, value=9, step=1)
    comp_year = st.number_input("Competitor Open Since Year", min_value=1900, max_value=2015, value=2010, step=1)

with c4:
    st.subheader("📣 Promotions & Holidays")
    state_holiday = st.selectbox("State Holiday", ["0", "a", "b", "c"],
                                    format_func=lambda x: {"0":"None","a":"Public","b":"Easter","c":"Christmas"}[x])
    school_holiday = st.selectbox("School Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")
    promo = st.selectbox("Promo Active", [0, 1], format_func=lambda x: "Yes" if x else "No")
    promo2 = st.selectbox("Promo2 Enrolled", [0, 1], format_func=lambda x: "Yes" if x else "No")
    if promo2:
        promo2_week = st.number_input("Promo2 Since Week", min_value=1, max_value=52, value=14, step=1)
        promo2_year = st.number_input("Promo2 Since Year", min_value=1900, max_value=2015, value=2010, step=1)
        promo_interval = st.selectbox("Promo Interval", ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"])
    else:
        promo2_week = 0
        promo2_year = 0
        promo_interval = "0"

payload = {
    "Store": store_id,
    "DayOfWeek": day_of_week,
    "Date": str(forecast_date),
    "Promo": promo,
    "StateHoliday": state_holiday,
    "SchoolHoliday": school_holiday,
    "StoreType": store_type,
    "Assortment": assortment,
    "CompetitionDistance": comp_distance,
    "CompetitionOpenSinceMonth": comp_month,
    "CompetitionOpenSinceYear": comp_year,
    "Promo2": promo2,
    "Promo2SinceWeek": promo2_week,
    "Promo2SinceYear": promo2_year,
    "PromoInterval": promo_interval,
    "sales_lag_7": lag_7,
    "sales_lag_14": lag_14,
    "rolling_mean_7": rolling_7,
    "rolling_mean_30": rolling_30,
}

if st.button("Forecast", use_container_width=True):
    with st.spinner("Generating forecast..."):
        try:
            response = requests.post(f"http://localhost:8000/predict", json=payload, timeout=10)
            response.raise_for_status()
            prediction = response.json()
            predicted_sales = prediction.get("prediction")
            st.metric("Predicted Sales", f"${predicted_sales:,.2f}")
        except Exception as e:
            st.error(f"Error: {e}")