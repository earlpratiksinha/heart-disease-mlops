import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, f1_score, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_curve, RocCurveDisplay)

warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates model and logs visual artifacts to MLflow."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    # Generate and log Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # Generate and log ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(f"ROC Curve: {model_name}")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()
    
    return metrics

def main():
    print("Loading cleaned data...")
    df = pd.read_csv('data/heart_disease_cleaned.csv')
    
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    models_params = {
        "Logistic_Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {"classifier__C": [0.1, 1.0, 10.0]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"classifier__n_estimators": [50, 100, 200]}
        }
    }
    
    best_model = None
    best_auc = 0
    
    for model_name, config in models_params.items():
        print(f"Tuning {model_name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', config["model"])
        ])
        
        clf = GridSearchCV(pipeline, config["params"], cv=5, scoring='roc_auc')
        clf.fit(X_train, y_train)
        
        print(f"--> Best {model_name} CV ROC-AUC: {clf.best_score_:.4f}")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(clf.best_params_)
            mlflow.log_metric("cv_roc_auc", clf.best_score_)
            
            metrics = evaluate_model(clf.best_estimator_, X_test, y_test, model_name)
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(clf.best_estimator_, "model")
            
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = clf.best_estimator_
                
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Training and tracking complete!")

if __name__ == "__main__":
    main()