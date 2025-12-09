import cv2 #opencv 
import os #for file path
import time
import uuid #naming images
from IPython.display import display, clear_output
import PIL.Image
import numpy as np

IMAGE_PATH = "Tensorflow/workspace/images/collected_images"

labels = ['A', 'B', 'C', 'thankyou', 'yes', 'no', 'peace', 'ok']
n_samples = 200

os.makedirs(IMAGE_PATH, exist_ok=True)

# open camera (index 0 since that worked for you)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open. Check permissions and index.")

try:
    for label in labels:
        folder = os.path.join(IMAGE_PATH, label)
        os.makedirs(folder, exist_ok=True)
        print(f"Collecting images for: {label}  -> saving to {folder}")
        saved = 0
        attempts = 0
        while saved < n_samples and attempts < n_samples * 10:
            attempts += 1
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Empty frame (attempt {attempts}); retrying...")
                time.sleep(1)
                continue

            fname = os.path.join(folder, f"{label}_{saved:04d}_{str(uuid.uuid1())[:8]}.jpg")
            ok = cv2.imwrite(fname, frame)
            if ok:
                saved += 1
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = PIL.Image.fromarray(img_rgb)
                clear_output(wait=True)
                display(pil.resize((320,240)))
                print(f"Saved {saved}/{n_samples}: {fname}")
            else:
                print("Failed to write image; retrying...")

            time.sleep(1)
        time.sleep(10)
        print(f"Done with {label}. Saved {saved} images.")
finally:
    cap.release()
    clear_output(wait=True)
    print("Camera released. Collection finished.")