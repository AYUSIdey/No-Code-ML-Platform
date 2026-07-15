import os
import zipfile
import tempfile
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

# Try to import TensorFlow for Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def analyze_vision_dataset(zip_file_path):
    """
    Analyzes the ZIP file to extract folder names (classes) and image counts.
    Returns data mapped to our Tabular UI structure so the dashboard works instantly!
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Find the root folder containing the class subfolders
        extract_path = temp_dir
        subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        
        # Handle cases where the zip contains a single parent folder
        if len(subdirs) == 1:
            extract_path = os.path.join(temp_dir, subdirs[0])
            subdirs = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
            
        classes = sorted(subdirs)
        if len(classes) < 2:
            return {"status": "error", "message": "Dataset must have at least 2 folders (classes)."}
            
        total_images = 0
        class_counts = {}
        preview_data = []
        
        for class_name in classes:
            class_path = os.path.join(extract_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            count = len(images)
            class_counts[class_name] = count
            total_images += count
            
            preview_data.append({
                "Image Class (Folder)": class_name,
                "Total Images": count,
                "Status": "Ready for Training"
            })

        return {
            "status": "success",
            "filename": os.path.basename(zip_file_path),
            "total_rows": total_images,
            "total_columns": len(classes), # We map 'columns' to 'classes' for the UI
            "column_types": {"Image Classes": "categorical"},
            "missing_values": {"Corrupted Images": 0},
            "preview": preview_data,
            "summary_stats": {}
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir)


def train_vision_model(zip_file_path):
    """
    Trains both traditional ML (flattened images) and Deep Learning CNNs (MobileNetV2).
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # 2. Find the root folder
        extract_path = temp_dir
        subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if len(subdirs) == 1:
            extract_path = os.path.join(temp_dir, subdirs[0])
            subdirs = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
            
        classes = sorted(subdirs)
        
        # 3. Load Images (RGB, 96x96 for MobileNetV2 compatibility)
        X_raw = []
        y_raw = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(extract_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        # Convert to RGB
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((96, 96))
                        X_raw.append(np.array(img))
                        y_raw.append(class_idx)
                    except Exception as e:
                        print(f"Skipping {img_name}: {e}")
                        
        if not X_raw:
            return {"status": "error", "message": "No valid images found in the zip file."}

        # Normalize pixels to 0-1 range
        X = np.array(X_raw) / 255.0
        y = np.array(y_raw)
        
        # Split Data
        X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Flatten for Random Forest & Logistic Regression
        X_train_flat = X_train_cnn.reshape(X_train_cnn.shape[0], -1)
        X_test_flat = X_test_cnn.reshape(X_test_cnn.shape[0], -1)
        
        results = {
            "status": "success",
            "task_type": "classification",
            "target_column": "Image Classes",
            "data_split": {
                "train_size": len(X_train_cnn),
                "test_size": len(X_test_cnn)
            },
            "models": {},
            "classes": {str(i): cls for i, cls in enumerate(classes)}
        }
        
        best_score = -1
        best_model_name = ""
        
        # ==========================================
        # PHASE 1: TRADITIONAL ML (Fast, flattened)
        # ==========================================
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
        }
        
        for model_name, model in models.items():
            model.fit(X_train_flat, y_train)
            y_pred = model.predict(X_test_flat)
            
            # Save Model
            safe_name = model_name.replace(" ", "_")
            model_filename = f"{safe_name}_vision.joblib"
            joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Extract ROC Data (Binary only)
            roc_data, roc_auc = None, None
            if len(classes) == 2 and hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_flat)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]

            results["models"][model_name] = {
                "accuracy": round(acc, 4),
                "macro_avg_f1": round(report['macro avg']['f1-score'], 4),
                "macro_avg_recall": round(report['macro avg']['recall'], 4),
                "cv_score_mean": round(acc, 4),
                "confusion_matrix": cm,
                "roc_data": roc_data,
                "roc_auc": round(roc_auc, 4) if roc_auc else None,
                "download_file": model_filename
            }
            
            if acc > best_score:
                best_score = acc
                best_model_name = model_name

        # ==========================================
        # PHASE 2: DEEP LEARNING (MobileNetV2)
        # ==========================================
        if TF_AVAILABLE:
            try:
                # 1. Load Pre-trained Base
                base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
                base_model.trainable = False # Freeze weights for transfer learning
                
                # 2. Add our custom classification head (FIXED NEURON COUNT)
                num_neurons = len(classes) if len(classes) > 2 else 1
                
                cnn = Sequential([
                    base_model,
                    GlobalAveragePooling2D(),
                    Dense(num_neurons, activation='softmax' if len(classes) > 2 else 'sigmoid')
                ])
                
                cnn.compile(
                    optimizer='adam', 
                    loss='sparse_categorical_crossentropy' if len(classes) > 2 else 'binary_crossentropy', 
                    metrics=['accuracy']
                )
                
                # 3. Train the model (5 epochs is usually enough for transfer learning!)
                cnn.fit(X_train_cnn, y_train, epochs=5, verbose=0)
                
                # 4. Predict
                if len(classes) > 2:
                    y_pred_proba = cnn.predict(X_test_cnn)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred_proba_raw = cnn.predict(X_test_cnn)
                    y_pred_proba = np.column_stack((1 - y_pred_proba_raw, y_pred_proba_raw))
                    y_pred = (y_pred_proba_raw > 0.5).astype(int).flatten()

                # 5. Evaluate
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, y_pred).tolist()
                
                # Extract ROC Data (Binary only)
                roc_data, roc_auc = None, None
                if len(classes) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
                
                # 6. Save Model (.keras is the new standard TF format)
                cnn_filename = f"MobileNetV2_vision.keras"
                cnn.save(os.path.join(MODELS_DIR, cnn_filename))
                
                results["models"]["MobileNetV2 (CNN)"] = {
                    "accuracy": round(acc, 4),
                    "macro_avg_f1": round(report['macro avg']['f1-score'], 4),
                    "macro_avg_recall": round(report['macro avg']['recall'], 4),
                    "cv_score_mean": round(acc, 4),
                    "confusion_matrix": cm,
                    "roc_data": roc_data,
                    "roc_auc": round(roc_auc, 4) if roc_auc else None,
                    "download_file": cnn_filename
                }
                
                if acc > best_score:
                    best_score = acc
                    best_model_name = "MobileNetV2 (CNN)"
                    
            except Exception as e:
                # Return the specific error to the UI instead of silently printing it
                return {"status": "error", "message": f"CNN Error: {str(e)}"}
        else:
            # Tell the UI if TensorFlow isn't installed
            return {"status": "error", "message": "TensorFlow is not installed in this environment. Please run 'pip install tensorflow' inside your (venv)."}
        
        results["best_model"] = best_model_name
        return results
        
    finally:
        shutil.rmtree(temp_dir)