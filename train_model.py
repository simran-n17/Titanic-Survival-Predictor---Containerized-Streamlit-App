# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)

# Preprocess
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data.fillna({
    'Age': data['Age'].median(),
    'Fare': data['Fare'].median()
}, inplace=True)

# Train
features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
X = data[features]
y = data['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'titanic_model.pkl')
print("Model saved to titanic_model.pkl")