import os
import io
import zipfile
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

DATASET_ZIP = 'Jaundice_Dataset.zip'
MODEL_OUT = 'jaundice_ensemble_model.joblib'

def extract_features(img_bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    im_resized = im.resize((128, 128))
    arr = np.array(im_resized, dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    yellow_index = (r + g) - 2.0 * b
    b_diff = (r - b) + (g - b)
    
    lum = 0.299*r + 0.587*g + 0.114*b
    bright_mask = lum > np.percentile(lum, 35)
    if np.sum(bright_mask) > 10:
        sclera_r, sclera_g, sclera_b = np.mean(r[bright_mask]), np.mean(g[bright_mask]), np.mean(b[bright_mask])
        sclera_yellow = np.mean(yellow_index[bright_mask])
        sclera_bdiff = np.mean(b_diff[bright_mask])
    else:
        sclera_r, sclera_g, sclera_b, sclera_yellow, sclera_bdiff = np.mean(r), np.mean(g), np.mean(b), np.mean(yellow_index), np.mean(b_diff)
        
    hsv = im_resized.convert('HSV')
    hsv_arr = np.array(hsv, dtype=np.float32)
    h, s, v = hsv_arr[:,:,0]/255.0, hsv_arr[:,:,1]/255.0, hsv_arr[:,:,2]/255.0
    
    ycbcr = im_resized.convert('YCbCr')
    y_arr = np.array(ycbcr, dtype=np.float32)
    cb, cr = y_arr[:,:,1]/255.0, y_arr[:,:,2]/255.0
    
    feats = [
        np.mean(r), np.std(r), np.mean(g), np.std(g), np.mean(b), np.std(b),
        np.mean(yellow_index), np.std(yellow_index),
        sclera_r, sclera_g, sclera_b, sclera_yellow, sclera_bdiff,
        np.mean(h), np.mean(s), np.mean(v),
        np.mean(cb), np.std(cb), np.mean(cr), np.std(cr)
    ]
    for channel in [r, g, b, yellow_index]:
        hist, _ = np.histogram(channel, bins=12, range=(float(np.min(channel)), float(np.max(channel))))
        feats.extend(hist / 16384.0)
        
    return np.array(feats, dtype=np.float32)

def main():
    if not os.path.exists(DATASET_ZIP):
        print(f"Error: {DATASET_ZIP} not found.")
        return
        
    print(f"Loading {DATASET_ZIP}...")
    z = zipfile.ZipFile(DATASET_ZIP)
    
    def load_split(prefix):
        X, y = [], []
        for label, folder in [(0, 'Healthy'), (1, 'Jaundice')]:
            path_prefix = f'Jaundice_Dataset/{prefix}/{folder}/'
            files = [n for n in z.namelist() if n.startswith(path_prefix) and n.endswith('.jpg')]
            for f in files:
                feats = extract_features(z.read(f))
                X.append(feats)
                y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = load_split('train')
    X_test, y_test = load_split('test')
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("Training Ensemble (RandomForest + ExtraTrees)...")
    
    rf = RandomForestClassifier(n_estimators=160, max_depth=12, random_state=42)
    et = ExtraTreesClassifier(n_estimators=160, max_depth=12, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('et', et)], voting='soft')
    ensemble.fit(X_train, y_train)
    
    preds = ensemble.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Healthy', 'Jaundice']))
    
    joblib.dump(ensemble, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == '__main__':
    main()
