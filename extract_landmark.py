import cv2, pandas as pd
from pathlib import Path
import mediapipe as mp

IMAGE_PATH = "Tensorflow/workspace/images/collected_images"

ROOT = Path(IMAGE_PATH)
OUTCSV = ROOT / "landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.25)

rows = []
for imgf in sorted(ROOT.rglob("*.jpg")):
    if "bad" in imgf.parts:
        continue
    img = cv2.imread(str(imgf))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        continue
    lm = res.multi_hand_landmarks[0]
    feat = []
    for p in lm.landmark:
        feat += [p.x, p.y, p.z]
    label = imgf.parent.name if imgf.parent != ROOT else imgf.name.split("_")[0]
    rows.append(feat + [label])

hands.close()

cols = [f"x{i}" for i in range(63)] + ["label"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUTCSV, index=False)
print("Saved", OUTCSV, "rows:", len(df))
print(df['label'].value_counts())

#cleaning up landmarks with bad labels

CSV = Path(IMAGE_PATH) / "landmarks.csv"
OUT = Path(IMAGE_PATH) / "landmarks.cleaned.csv"

df = pd.read_csv(CSV)
bad_mask = df['label'].str.startswith('.')
print("Removing labels that start with dot:", sorted(df.loc[bad_mask, 'label'].unique()))
df_clean = df[~bad_mask].reset_index(drop=True)
df_clean.to_csv(OUT, index=False)
print("Saved cleaned CSV to", OUT)
print(df_clean['label'].value_counts())
