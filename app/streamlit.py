import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# model
def load_model():
    
    import joblib
    return joblib.load('model_rf_new.joblib')

model = load_model()

def e_commerce_prediction(input_data):
    # Konversi input data menjadi DataFrame
    input_df = pd.DataFrame([input_data], columns=['Cost', 'Purchases', 'Importance', 'Discount', 'Weight (gram)', 'Warehouse_A', 'Warehouse_B', 'Warehouse_C', 'Warehouse_D', 'Warehouse_F', 'Shipment_Flight', 'Shipment_Road', 'Shipment_Ship', 'Reorder_purchases'])

    # Lakukan prediksi
    prediction = model.predict(input_df)
    
    # Format hasil prediksi
    if prediction[0] == 1:
        return 'Late'
    else:
        return 'On time'


def main():
    # Memberikan judul aplikasi
    st.markdown(
        """
        <div style='display: flex; align-items: center; margin-bottom: 20px;'>
            <img src='https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi1wySGm9lFze7qvDIteXSCMFxZJKvPVGc4Cy0tBLYf5Ajcl-uJsJ8RkUgqCjhzwB-K0cy_lvEMLYuCh7huiC4JJzo9byGv5PBusHm1hEKLuWN0JiRa7x2rNPV1l6o7MfX8g_w_4UaXh-UolBnD1Ke0jJyhRVUafrGJLP6cs1qbCMdw6TiqpaCgu3VARA/s320/sigma.png' style='width: 100px; height: 100px;'>
            <h1 style='margin-left: 20px;'>E-Commerce Late Shipping Prediction Project</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mendapatkan input data dari pengguna
    Cost = st.number_input('Product Cost (USD)')
    Purchases = st.number_input('Purchases (How Many You Purchase The Product)')
    Importance = st.number_input('Importance (0= low, 1= medium, 2=high)', min_value=0, max_value=2)
    Discount = st.number_input('Discount (%)')
    Weight = st.number_input('Weight (gram)')
    Warehouse_A = st.checkbox('Warehouse A')
    Warehouse_B = st.checkbox('Warehouse B')
    Warehouse_C = st.checkbox('Warehouse C')
    Warehouse_D = st.checkbox('Warehouse D')
    Warehouse_F = st.checkbox('Warehouse F')
    Shipment_Flight = st.checkbox('Shipment by Flight')
    Shipment_Road = st.checkbox('Shipment by Road')
    Shipment_Ship = st.checkbox('Shipment by Ship')
    Reorder_purchases = st.checkbox('Is it Reorder?')

    
    # Kode untuk prediksi
    diagnosis = ''
    
    # Membuat tombol untuk prediksi
    if st.button('Check Shipping Status'):
        diagnosis = e_commerce_prediction([Cost, Purchases, Importance, Discount, Weight, Warehouse_A, Warehouse_B, Warehouse_C, Warehouse_D, Warehouse_F, Shipment_Flight, Shipment_Road, Shipment_Ship, Reorder_purchases])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
