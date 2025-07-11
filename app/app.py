import streamlit as st
import pandas as pd
import joblib



model = joblib.load('app/rf_selling_score_best.joblib')


st.title("Product Selling Score Predictor")
st.write("Enter product details to predict selling score (1 = Best, 5 = Worst)")

days_since_launch = st.number_input('Days since launch', min_value=0, max_value=10000, value=10)
total_stock_sold_percentage = st.number_input('Total Stock Sold Percentage', min_value=0.0, max_value=5.0, value=0.5)
alltime_perday_quantity = st.number_input('Alltime Perday Quantity', min_value=0.0, max_value=10.0, value=1.0)
alltime_perday_view = st.number_input('Alltime Perday View', min_value=0.0, max_value=4000.0, value=10.0)
alltime_perday_atc = st.number_input('Alltime Perday ATC', min_value=0.0, max_value=150.0, value=1.0)

if st.button('Predict Selling Score'):
    new_data = pd.DataFrame([{
        'Days_since_launch': days_since_launch,
        'Total_Stock_Sold_Percentage': total_stock_sold_percentage,
        'Alltime_Perday_Quantity': alltime_perday_quantity,
        'Alltime_Perday_View': alltime_perday_view,
        'Alltime_Perday_ATC': alltime_perday_atc
    }])
    pred_score = model.predict(new_data)[0]
    st.success(f"Predicted selling score (1=Best, 5=Worst): {pred_score}")
