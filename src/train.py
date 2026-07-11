import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def main():
    print("Loading cleaned data...")
    df = pd.read_csv('data/heart_disease_cleaned.csv')
    
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set up MLflow tracking URI to save locally in the 'mlruns' folder
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    # Define our two models
    models = {
        "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random_Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    best_model = None
    best_auc = 0
    
    for model_name, classifier in models.items():
        print(f"Training {model_name}...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=model_name):
            
            # Create a Pipeline: This ensures scaling is saved WITH the model for inference
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classifier)
            ])
            
            # Train the pipeline
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            metrics = evaluate_model(pipeline, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")
            
            print(f"--> {model_name} Metrics: {metrics}")
            
            # Keep track of the best model based on ROC-AUC score
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = pipeline
                
    # Save the best pipeline as a .pkl file for the API to use later
    print("\nSaving the best model to 'models/best_model.pkl'...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Training and tracking complete!")

if __name__ == "__main__":
    main()