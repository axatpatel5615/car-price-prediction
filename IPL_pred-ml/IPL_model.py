import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load prepared data
df = pd.read_csv('ipl_processed_data.csv')

X = df.drop(columns=['result'])
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up category transformer pipelines
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'venue'])
], remainder='passthrough')

# Bind processing transformers and algorithm rules tightly inside a single deployment package
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("=========================================")
print(f"🚀 Model Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("=========================================")

# Save the live deployment artifact file
with open('ipl_probability_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)
print("Pipeline serialized successfully as 'ipl_probability_pipeline.pkl'!")
