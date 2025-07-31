
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    # Load dataset
    df = pd.read_csv('data.csv')
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    
    # Convert labels
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy (All 30 Features): {accuracy * 100:.2f}%")
    
    # Save
    joblib.dump(model, 'breast_cancer_model.pkl')
    return model

if __name__ == "__main__":
    train_model()