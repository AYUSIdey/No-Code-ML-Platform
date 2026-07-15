import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
import os
import joblib # Added for model saving

# Define a directory to store our trained models temporarily
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate(file_path, target_column, task_type):
    """
    Automated Machine Learning pipeline for training and evaluation.
    """
    if not os.path.exists(file_path):
        return {"status": "error", "message": "File not found"}
        
    try:
        # 1. Load the Data
        df = pd.read_csv(file_path)
        
        if target_column not in df.columns:
            return {"status": "error", "message": f"Target column '{target_column}' not found in dataset."}
            
        # 2. Drop rows where the TARGET is missing
        df = df.dropna(subset=[target_column])
        
        # 3. Separate Features (X) and Target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # If classification, encode the target column
        label_mapping = None
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_mapping = {int(i): str(val) for i, val in enumerate(le.classes_)}

        # 4. Identify column types for X
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

        # 5. Build the Preprocessing Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # 6. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 7. Select Models to Train
        if task_type == 'classification':
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
        elif task_type == 'regression':
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
            }
        else:
            return {"status": "error", "message": "Invalid task type"}

        results = {
            "status": "success",
            "task_type": task_type,
            "target_column": target_column,
            "data_split": {
                "train_size": len(X_train),
                "test_size": len(X_test)
            },
            "models": {}
        }

        best_model_name = ""
        best_score = -float("inf") if task_type == 'classification' else float("inf")

        # 8. Train, Evaluate, and SAVE Each Model
        for model_name, model in models.items():
            # Combine preprocessing and modeling into a single pipeline
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])
                                  
            # TRAIN THE MODEL!
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # --- EXTRACT FEATURE IMPORTANCE (Explainable AI) ---
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                try:
                    # Try to get actual column names from the preprocessor
                    feature_names = preprocessor.get_feature_names_out()
                    # Clean up the names (Scikit-learn adds prefixes like 'num__' or 'cat__')
                    feature_names = [name.split('__')[-1] for name in feature_names]
                except:
                    # Fallback if names can't be extracted
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                    
                # Zip them together, convert to standard Python float for JSON, and sort descending
                feat_imp = [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(feature_names, importances)]
                feat_imp = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)[:15] # Keep top 15
                feature_importance = feat_imp
            # ---------------------------------------------------
            
            # --- TASK 2: SAVE THE MODEL TO DISK ---
            # Create a safe filename without spaces or weird characters
            safe_model_name = model_name.replace(" ", "_")
            safe_target = target_column.replace(" ", "_").replace("/", "_")
            model_filename = f"{safe_model_name}_{safe_target}.joblib"
            model_filepath = os.path.join(MODELS_DIR, model_filename)
            
            # This saves the entire pipeline (imputation + scaling + ML algorithm)
            joblib.dump(clf, model_filepath)
            # --------------------------------------

            model_metrics = {
                "download_file": model_filename, # Pass this back to the frontend
                "feature_importance": feature_importance # Pass XAI data to frontend
            }

            if task_type == 'classification':
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # Cross Validation
                cv_scores = cross_val_score(clf, X, y, cv=3)
                cm = confusion_matrix(y_test, y_pred).tolist()
                
                roc_data = None
                roc_auc = None
                if len(le.classes_) == 2 and hasattr(model, "predict_proba"):
                    y_pred_proba = clf.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    roc_data = [{"fpr": round(f, 4), "tpr": round(t, 4)} for f, t in zip(fpr, tpr)]

                model_metrics.update({
                    "accuracy": round(acc, 4),
                    "macro_avg_f1": round(report['macro avg']['f1-score'], 4),
                    "macro_avg_recall": round(report['macro avg']['recall'], 4),
                    "cv_score_mean": round(cv_scores.mean(), 4),
                    "confusion_matrix": cm,
                    "roc_data": roc_data,
                    "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                })
                
                if acc > best_score:
                    best_score = acc
                    best_model_name = model_name
                    
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                model_metrics.update({
                    "rmse": round(rmse, 4),
                    "r2_score": round(r2, 4)
                })
                
                if rmse < best_score:
                    best_score = rmse
                    best_model_name = model_name

            results["models"][model_name] = model_metrics

        if task_type == 'classification':
            results["classes"] = label_mapping
            
        results["best_model"] = best_model_name

        return results

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Let's create a dummy classification dataset
    import json
    dummy_data = {
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Salary": [50000, 60000, 75000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        "Department": ["IT", "HR", "IT", "Marketing", "HR", "IT", "Marketing", "IT", "HR", "Marketing"],
        "Promoted": ["No", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes"]
    }
    df_dummy = pd.DataFrame(dummy_data)
    test_filename = "dummy_ml_test.csv"
    df_dummy.to_csv(test_filename, index=False)
    
    print(f"--- Training Classification Model ---")
    result = train_and_evaluate(test_filename, target_column="Promoted", task_type="classification")
    print(json.dumps(result, indent=2))
    
    if os.path.exists(test_filename):
        os.remove(test_filename)