import cv2 as cv
import numpy as np
import pickle
import os
import time
from train import forward_prop, get_predictions
from config import MODEL_WEIGHTS_PATH

with open(MODEL_WEIGHTS_PATH, "rb") as f:
    W1, b1, W2, b2 = pickle.load(f)

def preprocess_roi(roi):
    roi = cv.flip(roi, 1)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th = 255 - th
    kernel = np.ones((3, 3), np.uint8)
    th = cv.dilate(th, kernel, iterations=1)
    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    digit = th
    if contours:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        if w > 10 and h > 10:
            digit = th[y:y+h, x:x+w]
    digit = cv.resize(digit, (28, 28), interpolation=cv.INTER_AREA)
    cv.imshow("preprocessed", digit)
    x = digit.reshape(28*28, 1).astype(np.float32) / 255.0
    return x

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("camera error")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape
    ch = h / 2
    cw = w / 2
    rh = ch / 3
    rw = cw / 3
    tl = (int(cw - rw), int(ch - rh))
    br = (int(cw + rw), int(ch + rh))
    cv.rectangle(frame, tl, br, (0, 0, 0), 2)
    cv.imshow("camera", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord(' '):
        roi = cv.getRectSubPix(frame, (int(2*rw), int(2*rh)), (cw, ch))
        x = preprocess_roi(roi)
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, x)
        pred = get_predictions(A2)[0]
        conf = float(np.max(A2))
        if conf >= 0.9:
            print("digit:", pred, "conf:", round(conf, 2))
        else:
            print("no number, conf:", round(conf, 2))
        os.makedirs("captures", exist_ok=True)
        cv.imwrite(f"captures/roi_{int(time.time())}.png", roi)
    if key == ord('q'):
        break

cam.release()
cv.destroyAllWindows()