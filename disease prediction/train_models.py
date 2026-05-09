import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_data

def train_and_evaluate():
    # Load dataset (downloads if necessary)
    df = load_data()
    
    # Preprocessing
    df = df.dropna()
    
    # Features and target
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the feature names for the app
    feature_names = X_train.columns.tolist()
    os.makedirs('models', exist_ok=True)
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"Saved {len(feature_names)} feature names.")

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    print("\n--- Model Evaluation ---")
    best_rf_model = None
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"[{name}]")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-Score : {f1:.4f}\n")
        
        if name == "Random Forest":
            best_rf_model = model
            
    # Save the Random Forest model
    joblib.dump(best_rf_model, 'models/random_forest_model.pkl')
    print("Saved Random Forest model to models/random_forest_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
