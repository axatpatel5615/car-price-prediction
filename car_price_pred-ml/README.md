# 🚗 Car Price Prediction Engine

An end-to-end Machine Learning solution designed to predict the resale valuation of automobiles using historical data variables. This project features a clean data engineering pipeline, a trained machine learning model, and an interactive **Streamlit** dashboard web interface for real-time inference.

## 🚀 Key Features

* **Data Cleaning & Pipeline Management:** Robust pre-processing rules that handle missing features, clear statistical outliers, and format unstructured raw text for algorithmic compatibility.
* **Automated Feature Engineering:** Custom encoding rules to manage high-cardinality nominal values (e.g., Car Brand, Model Name, Fuel Type, and Transmission) while ensuring zero data leakage between training slices and production inputs.
* **Optimized Regressor Pipeline:** Employs a Scikit-Learn regression architecture to output numerical pricing trends based on vehicle mileage, age, condition, and market status.
* **Interactive Inference UI:** A reactive web portal powered by Streamlit, allowing recruiters or consumers to dynamically input unique vehicle specs via sidebar selectors and instantly compute resale estimates.

## 🛠️ Architecture & Tech Stack

* **Programming Language:** Python 3.10+
* **Data Engineering & Analytics:** Pandas, NumPy
* **Machine Learning Engine:** Scikit-Learn (Model Selection, Preprocessing, Linear Modeling, Evaluation Metrics)
* **Web Dashboard Framework:** Streamlit
* **Model Serialization:** Pickle / Joblib 

---

## 📁 Repository Blueprint

```text
├── car_price_prediction.csv   # Target historical automotive dataset
├── project.py                 # Core ML training pipeline (Cleaning, Splitting, Training, Saving)
├── app.py                     # Streamlit web application layout and user inference logic
├── requirements.txt           # Declared project dependencies for fast deployment
└── README.md                  # Comprehensive project portfolio documentation
