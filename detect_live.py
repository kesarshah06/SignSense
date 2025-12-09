import time, joblib, numpy as np, cv2, signal, sys
import mediapipe as mp
from pathlib import Path

MODEL_PATH = Path("Tensorflow/workspace/images/collected_images/model/gesture_baseline.pkl")
CAM_INDEX = 0 #for default camera
WINDOW_NAME = "Gesture Detector - Press ESC to quit"

data = joblib.load(MODEL_PATH) #load model, scaler, label encoder
clf = data['model']
scaler = data['scaler']
le = data['label_encoder']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) #using mediapipe hands

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened(): #if camera not opened
    print("Cannot open camera", CAM_INDEX)
    sys.exit(1)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

def handle_exit(signum, frame): #for a smooth exit on signals
    print("exiting...")
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        label_text = ""
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            feat = []
            h, w, _ = frame.shape
            xs, ys = [], []
            for p in lm.landmark:
                feat += [p.x, p.y, p.z]
                xs.append(int(p.x * w))
                ys.append(int(p.y * h))

            X = np.array(feat).reshape(1, -1)
            Xs = scaler.transform(X)
            pred = clf.predict(Xs)[0]
            label_text = le.inverse_transform([pred])[0]

            for c in res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, c, mp_hands.HAND_CONNECTIONS)

            x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)

            box_color = (0, 0, 0)    
            text_color = (255, 255, 255)  
            bg_color = box_color
            thickness = 4

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, thickness)


            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_th = 2
            text = f"{label_text}"

            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_th)
            text_x = x_min
            text_y = y_min - 10
            if text_y - text_h - baseline < 0:
                text_y = y_min + text_h + 10

            rect_tl = (text_x - 5, text_y - text_h - baseline - 3)
            rect_br = (text_x + text_w + 5, text_y + 5)
            cv2.rectangle(frame, rect_tl, rect_br, bg_color, cv2.FILLED)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_th, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Released camera and closed.")
