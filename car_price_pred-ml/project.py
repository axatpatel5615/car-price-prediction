import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# 1. DATA LOADING & PRIMARY INSPECTION
df = pd.read_csv('car_price_prediction.csv')

pd.set_option('display.max_columns', None)
df.drop_duplicates(inplace=True)

# 2. DATA CLEANING & RE-ENGINEERING
# Clean Levy
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
df['Levy'].fillna(df['Levy'].mean(), inplace=True)

# FIX: Safely parse decimal numbers from Engine volume (ignoring words like 'Turbo')
df['Engine volume'] = df['Engine volume'].str.split().str[0]
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')
df['Engine volume'].fillna(df['Engine volume'].median(), inplace=True)

# Clean Mileage
df['Mileage'] = df['Mileage'].str.replace(r'\D', '', regex=True)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Mileage'].fillna(df['Mileage'].median(), inplace=True)

# Clean Doors string values
df['Doors'] = df['Doors'].replace({
    '04-May': 5,
    '02-Mar': 3,
    '>5': 6
})
df['Doors'] = pd.to_numeric(df['Doors'])

# Clean Drive wheels mapping values
df['Drive wheels'] = df['Drive wheels'].replace({
    'Front': 1,
    'Rear': 2,
    '4x4': 4
})

# Outlier Management: Filter price extremes and astronomical mileage anomalies
df = df[df['Price'] < df['Price'].quantile(0.99)]
df = df[df['Mileage'] < df['Mileage'].quantile(0.99)]


# 3. STATISTICAL VALIDATION & CORRELATION
# Quick correlation validation on numeric subsets
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("--- Baseline Pearson Feature Correlations with Price ---")
print(df[numeric_cols].corr()['Price'].sort_values(ascending=False))

# Chi-Square Test to check dependency on categorical values
alpha = 0.05
df['Price_bins'] = pd.qcut(df['Price'], q=4, labels=False)
categorical_cols = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Wheel', 'Color']

print("\n--- Running Chi-Square Contingency Tests ---")
cols_to_keep = []
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df['Price_bins'])
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    if p_value < alpha:
        print(f"Keep '{col}' (p-value: {p_value:.5f})")
        cols_to_keep.append(col)
    else:
        print(f"Drop '{col}' (p-value: {p_value:.5f})")

# Clean up temporary bins
df.drop(columns=['Price_bins'], inplace=True)

# 4. ENCODING & MODEL PIPELINE PREPARATION
# Isolate predictive features (Explicitly dropping random unique text identifiers like ID)
X_raw = df.drop(columns=['Price', 'ID'], errors='ignore')
y = df['Price']

# Define structured list of variables to encode
cat_features = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Color']
num_features = ['Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Drive wheels', 'Doors', 'Airbags']

# We preserve a category mapping dict so that our UI app can accurately parse user string selections
category_mappings = {}
X_encoded = X_raw.copy()

for col in cat_features:
    X_encoded[col] = X_encoded[col].astype(str).str.strip()
    X_encoded[col] = X_encoded[col].astype('category')
    # Save the order mapping category indices to labels
    category_mappings[col] = list(X_encoded[col].cat.categories)
    X_encoded[col] = X_encoded[col].cat.codes

# Filter features to just the finalized aligned columns
final_feature_order = num_features + cat_features
X_encoded = X_encoded[final_feature_order]

# 5. HIGH ACCURACY MACHINE LEARNING ENGINE
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)

print("\nTraining Ensemble RandomForest Machine Learning Model...")
# Utilizing an Ensemble Regressor to properly extract non-linear patterns
model = RandomForestRegressor(
    n_estimators=150, 
    max_depth=18, 
    min_samples_split=4,
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

print(f"Optimized Test R2 Score: {r2:.4f}")
print(f"Adjusted R2 Score:      {adjusted_r2:.4f}")

# ==========================================
# 6. EXPORTING PIPELINE ARTIFACT FOR UI
# ==========================================
# Bundle the model with its structural metadata so the UI script works flawlessly
# export_payload = {
#     'model': model,
#     'features': final_feature_order,
#     'cat_features': cat_features,
#     'num_features': num_features,
#     'mappings': category_mappings
# }

# with open('car_price_pipeline.pkl', 'wb') as f:
#     pickle.dump(export_payload, f)

# print("\nSuccess! Saved deployment pipeline to 'car_price_pipeline.pkl'.")