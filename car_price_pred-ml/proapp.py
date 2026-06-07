from matplotlib.colors import Colormap
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Car Value Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Accent
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1E3A8A; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
    div.stButton > button:first-child { background-color: #1E3A8A; color: white; width: 100%; border-radius: 8px; height: 3em; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Intelligent Car Valuation Engine")
st.markdown("Provide the vehicle parameters below to compute an optimized market price estimate using our trained Random Forest pipeline.")
st.write("---")

# 2. Safe Loading of the Pipeline Data Payload
@st.cache_resource
def load_deployment_pipeline():
    try:
        with open('car_price_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("🚨 Missing Artifact: 'car_price_pipeline.pkl' was not found in this directory. Run your updated training script first!")
        return None

payload = load_deployment_pipeline()

if payload is not None:
    # Extract structural elements from our saved dictionary payload
    model = payload['model']
    mappings = payload['mappings']
    final_feature_order = payload['features']

    # 3. Form Layout Strategy
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📋 Core Metadata")
        # Text dropdown parameters loaded directly from saved category mappings
        manufacturer = st.selectbox("Manufacturer Brand", sorted(mappings['Manufacturer']))
        
        # Filter models belonging strictly to the selected manufacturer to keep UI clean
        model_name = st.selectbox("Vehicle Model Series", sorted(mappings['Model']))
        category = st.selectbox("Vehicle Classification", sorted(mappings['Category']))
        color = st.selectbox("Exterior Paint Palette", sorted(mappings['Color']))

    with col2:
        st.markdown("### ⚙️ Mechanical Attributes")
        prod_year = st.slider("Production Manufacture Year", 1990, 2026, 2018)
        engine_vol = st.number_input("Engine Volume Displacement (Liters)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        cylinders = st.slider("Engine Cylinder Count", 1, 16, 4)
        fuel_type = st.selectbox("Fuel Configuration System", sorted(mappings['Fuel type']))
        gearbox = st.selectbox("Transmission Gear Box Variant", sorted(mappings['Gear box type']))

    with col3:
        st.markdown("### 🛣️ Usage & Safety Metrics")
        mileage = st.number_input("Odometer Mileage Reading (km)", min_value=0, max_value=1000000, value=75000, step=1000)
        levy = st.number_input("Government Registration Levy ($)", min_value=0, max_value=20000, value=800, step=50)
        airbags = st.slider("Safety Airbags Configured", 0, 16, 6)
        
        # Mapping numerical layout targets matching back to training configurations
        drive_wheels_label = st.selectbox("Drive Train Type", ["Front Wheel Drive", "Rear Wheel Drive", "4x4 / AWD"])
        drive_wheels_map = {"Front Wheel Drive": 1, "Rear Wheel Drive": 2, "4x4 / AWD": 4}
        drive_wheels = drive_wheels_map[drive_wheels_label]

        doors_label = st.selectbox("Doors Count Configuration", ["4 Doors", "2 Doors", "5 Doors / Hatch"])
        doors_map = {"4 Doors": 4, "2 Doors": 2, "5 Doors / Hatch": 5}
        doors = doors_map[doors_label]
        
        leather_label = st.radio("Premium Leather Interior Trim", ["Yes", "No"], horizontal=True)

    st.write("---")

    # 4. Inversion / Inference Logic Implementation
    if st.button("Calculate Market Valuation", type="primary"):
        # Convert incoming string configurations back to matching structural cat codes
        def get_encoded_value(column_name, selection):
            try:
                return mappings[column_name].index(selection)
            except ValueError:
                return -1 # Handle anomalous categories safely

        # Assemble individual categorical code transformations
        encoded_manufacturer = get_encoded_value('Manufacturer', manufacturer)
        encoded_model = get_encoded_value('Model', model_name)
        encoded_category = get_encoded_value('Category', category)
        encoded_fuel = get_encoded_value('Fuel type', fuel_type)
        encoded_gearbox = get_encoded_value('Gear box type', gearbox)
        encoded_color = get_encoded_value('Color', color)
        encoded_leather = get_encoded_value('Leather interior', leather_label)

        # Structure input data dictionary referencing exact numerical vs encoded partitions
        input_data = {
            'Levy': levy,
            'Prod. year': prod_year,
            'Engine volume': engine_vol,
            'Mileage': mileage,
            'Cylinders': cylinders,
            'Drive wheels': drive_wheels,
            'Doors': doors,
            'Airbags': airbags,
            'Manufacturer': encoded_manufacturer,
            'Model': encoded_model,
            'Category': encoded_category,
            'Leather interior': encoded_leather,
            'Fuel type': encoded_fuel,
            'Gear box type': encoded_gearbox,
            'Color': Colormap if 'Color' not in mappings else encoded_color
        }

        # Structure payload sequence order to perfectly reflect DataFrame training parameters
        input_df = pd.DataFrame([input_data])[final_feature_order]

        try:
            # Generate target predictions from our ensemble matrix array
            predicted_price = model.predict(input_df)[0]
            
            # Floor out edge calculations mathematically resulting from extreme parameters
            if predicted_price < 150:
                predicted_price = 500.00
                
            st.balloons()
            st.success(f"## 💵 Estimated Valuation Result: **${predicted_price:,.2f}**")
            
        except Exception as prediction_error:
            st.error(f"Execution Error during matrix operation: {prediction_error}")