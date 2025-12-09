
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

IMAGE_PATH = "Tensorflow/workspace/images/collected_images"

ROOT = Path(IMAGE_PATH)
CSV = ROOT / "landmarks.csv"
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)
X = df.drop(columns=['label']).values.astype(np.float32)
y_text = df['label'].values
le = LabelEncoder(); y = le.fit_transform(y_text)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
print("Test acc:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump({'model':clf, 'scaler':scaler, 'label_encoder': le}, OUT / "gesture_baseline.pkl")
print("Saved model to", OUT / "gesture_baseline.pkl")
