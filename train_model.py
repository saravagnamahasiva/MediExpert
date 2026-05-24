import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Step 1: Load Training Data ===
data_path = 'datasets/training_data.csv'
df = pd.read_csv(data_path)

# === Step 2: Separate Features and Labels ===
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# === Step 3: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train Model ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Step 5: Evaluate Model ===
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained. Accuracy: {accuracy:.2f}")

# === Step 6: Save Model ===
os.makedirs('model', exist_ok=True)
joblib.dump(clf, 'model/disease_model.pkl')
print("Model saved to model/disease_model.pkl")
