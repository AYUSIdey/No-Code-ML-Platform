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
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
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
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        extract_path = temp_dir
        subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        
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
            "total_columns": len(classes), 
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
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        extract_path = temp_dir
        subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if len(subdirs) == 1:
            extract_path = os.path.join(temp_dir, subdirs[0])
            subdirs = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
            
        classes = sorted(subdirs)
        
        X_raw = []
        y_raw = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(extract_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((96, 96))
                        X_raw.append(np.array(img))
                        y_raw.append(class_idx)
                    except Exception as e:
                        print(f"Skipping {img_name}: {e}")
                        
        if not X_raw:
            return {"status": "error", "message": "No valid images found in the zip file."}

        X = np.array(X_raw) / 255.0
        y = np.array(y_raw)
        
        X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        
        best_score = -1.0
        best_model_name = ""
        
        # ==========================================
        # PHASE 1: TRADITIONAL ML
        # ==========================================
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
        }
        
        for model_name, model in models.items():
            model.fit(X_train_flat, y_train)
            y_pred = model.predict(X_test_flat)
            
            safe_name = model_name.replace(" ", "_")
            model_filename = f"{safe_name}_vision.joblib"
            joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
            
            # Explicitly cast metrics to Python floats to prevent FastAPI JSON crashes
            acc = float(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Ensure Confusion Matrix is a pure Python nested list of ints
            cm_raw = confusion_matrix(y_test, y_pred).tolist()
            cm = [[int(val) for val in row] for row in cm_raw]
            
            roc_data, roc_auc = None, None
            if len(classes) == 2 and hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_flat)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = float(auc(fpr, tpr))
                roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]

            results["models"][model_name] = {
                "accuracy": round(acc, 4),
                "macro_avg_f1": round(float(report['macro avg']['f1-score']), 4),
                "macro_avg_recall": round(float(report['macro avg']['recall']), 4),
                "cv_score_mean": round(acc, 4),
                "confusion_matrix": cm,
                "roc_data": roc_data,
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                "download_file": model_filename
            }
            
            if acc > best_score:
                best_score = acc
                best_model_name = model_name

        # ==========================================
        # PHASE 2: DEEP LEARNING (MobileNetV2 + Grad-CAM)
        # ==========================================
        if TF_AVAILABLE:
            try:
                num_neurons = len(classes) if len(classes) > 2 else 1
                
                # We build the model 'flat' so Grad-CAM can easily hook into the convolutional layers
                base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
                base_model.trainable = False 
                
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                outputs = Dense(num_neurons, activation='softmax' if len(classes) > 2 else 'sigmoid')(x)
                
                cnn = Model(inputs=base_model.input, outputs=outputs)
                
                cnn.compile(
                    optimizer='adam', 
                    loss='sparse_categorical_crossentropy' if len(classes) > 2 else 'binary_crossentropy', 
                    metrics=['accuracy']
                )
                
                cnn.fit(X_train_cnn, y_train, epochs=5, verbose=0)
                
                if len(classes) > 2:
                    y_pred_proba = cnn.predict(X_test_cnn)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred_proba_raw = cnn.predict(X_test_cnn)
                    y_pred_proba = np.column_stack((1 - y_pred_proba_raw, y_pred_proba_raw))
                    y_pred = (y_pred_proba_raw > 0.5).astype(int).flatten()

                # Cast all numpy types to pure python types
                acc = float(accuracy_score(y_test, y_pred))
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm_raw = confusion_matrix(y_test, y_pred).tolist()
                cm = [[int(val) for val in row] for row in cm_raw]
                
                roc_data, roc_auc = None, None
                if len(classes) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = float(auc(fpr, tpr))
                    roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
                
                # --- GRAD-CAM GENERATION ---
                gradcam_images = []
                try:
                    import matplotlib
                    matplotlib.use('Agg') # Prevents GUI thread crash on servers
                    import io
                    import base64
                    
                    # Take 2 random images from the test set
                    sample_indices = np.random.choice(len(X_test_cnn), min(2, len(X_test_cnn)), replace=False)
                    
                    # Safely locate the last convolutional layer
                    last_conv_layer_name = base_model.layers[-1].name
                    grad_model = tf.keras.models.Model(cnn.inputs, [cnn.get_layer(last_conv_layer_name).output, cnn.output])
                    
                    for idx in sample_indices:
                        img_array = X_test_cnn[idx:idx+1]
                        actual_class = classes[int(y_test[idx])]
                        
                        with tf.GradientTape() as tape:
                            last_conv_layer_output, preds = grad_model(img_array)
                            if len(classes) == 2:
                                class_channel = preds[0, 0]
                            else:
                                pred_index = tf.argmax(preds[0])
                                class_channel = preds[0, pred_index]

                        # Calculate gradients
                        grads = tape.gradient(class_channel, last_conv_layer_output)
                        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                        
                        # Weigh the feature maps by the gradients
                        last_conv_layer_output = last_conv_layer_output[0]
                        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
                        heatmap = tf.squeeze(heatmap)
                        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                        heatmap = heatmap.numpy()
                        
                        # Apply Color map
                        heatmap_norm = np.uint8(255 * heatmap) if np.max(heatmap) > 0 else heatmap
                        
                        # FIX: Matplotlib 3.9+ removed get_cmap from cm. Use colormaps registry instead.
                        jet = matplotlib.colormaps["jet"]
                        jet_colors = jet(np.arange(256))[:, :3]
                        jet_heatmap = jet_colors[heatmap_norm]
                        
                        # Superimpose onto original image
                        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                        jet_heatmap = jet_heatmap.resize((96, 96))
                        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
                        
                        img_original = img_array[0] * 255
                        superimposed_img = jet_heatmap * 0.4 + img_original
                        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                        
                        # Convert to base64 for React UI
                        buf_grad = io.BytesIO()
                        superimposed_img.save(buf_grad, format="JPEG")
                        grad_b64 = base64.b64encode(buf_grad.getvalue()).decode("utf-8")
                        
                        buf_orig = io.BytesIO()
                        orig_pil = tf.keras.preprocessing.image.array_to_img(img_original)
                        orig_pil.save(buf_orig, format="JPEG")
                        orig_b64 = base64.b64encode(buf_orig.getvalue()).decode("utf-8")
                        
                        gradcam_images.append({
                            "label": f"Class: {actual_class}",
                            "original": orig_b64,
                            "gradcam": grad_b64
                        })
                except Exception as e:
                    print(f"GradCAM Generation Failed: {e}")
                # ---------------------------

                cnn_filename = f"MobileNetV2_vision.keras"
                cnn.save(os.path.join(MODELS_DIR, cnn_filename))
                
                results["models"]["MobileNetV2 (CNN)"] = {
                    "accuracy": round(acc, 4),
                    "macro_avg_f1": round(float(report['macro avg']['f1-score']), 4),
                    "macro_avg_recall": round(float(report['macro avg']['recall']), 4),
                    "cv_score_mean": round(acc, 4),
                    "confusion_matrix": cm,
                    "roc_data": roc_data,
                    "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                    "gradcam_images": gradcam_images, 
                    "download_file": cnn_filename
                }
                
                if acc > best_score:
                    best_score = acc
                    best_model_name = "MobileNetV2 (CNN)"
                    
            except Exception as e:
                return {"status": "error", "message": f"CNN Error: {str(e)}"}
        else:
            return {"status": "error", "message": "TensorFlow is not installed in this environment."}
        
        results["best_model"] = best_model_name
        return results
        
    finally:
        shutil.rmtree(temp_dir)