import cv2
import time
import numpy as np
import winsound
import os
from ultralytics import YOLO
from datetime import datetime

# ================= CONFIG =================
model = YOLO("fire.pt")
model.fuse()

video_path = "video.mp4"

SAVE_FOLDER = "outputs"
LOG_FILE = "detection_log.txt"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ================= CAMERA =================
cap_cam = cv2.VideoCapture(0)   # webcam
cap_phone = cv2.VideoCapture(1) # OBS iPhone (đổi nếu cần)
cap_vid = cv2.VideoCapture(video_path)

# giảm lag nhưng vẫn giữ chất lượng
cap_cam.set(3, 640)
cap_cam.set(4, 480)

cap_phone.set(3, 640)
cap_phone.set(4, 480)

# ================= LOG =================
def write_log(event):
    now = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {event}\n")

# ================= RESIZE =================
def resize_keep_ratio(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    x = (w - nw)//2
    y = (h - nh)//2

    canvas[y:y+nh, x:x+nw] = resized
    return canvas

# ================= FIRE =================
def is_fire(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0,120,140), (60,255,255))

    return np.sum(mask > 0) / (region.size/3) > 0.10

# ================= SMOKE =================
def is_smoke(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0,0,50), (180,60,200))

    return np.sum(mask > 0) / (region.size/3) > 0.15

# ================= PROCESS =================
def process(frame):
    results = model(frame)[0]

    fire = False
    smoke = False

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue

        conf = float(box.conf[0])
        if conf < 0.35:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            continue

        if is_fire(region):
            fire = True
            color = (0,0,255)
            label = "FIRE"
        elif is_smoke(region):
            smoke = True
            color = (200,200,200)
            label = "SMOKE"
        else:
            continue

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    return frame, fire, smoke

# ================= LOOP =================
last_alert = 0

while True:
    ret1, cam = cap_cam.read()
    ret2, phone = cap_phone.read()
    ret3, vid = cap_vid.read()

    if not ret1:
        break

    # fallback nếu phone lỗi
    if not ret2 or phone is None:
        phone = np.zeros((320,480,3), dtype=np.uint8)
        cv2.putText(phone,"NO PHONE",(120,160),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    if not ret3:
        cap_vid.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue

    # ================= DETECT =================
    cam,f1,s1 = process(cam)
    vid,f3,s3 = process(vid)

    # 🔥 PHONE chỉ detect mỗi 2 frame (giảm lag)
    if int(time.time()*10) % 2 == 0:
        phone,f2,s2 = process(phone)
    else:
        f2, s2 = False, False

    fire = f1 or f2 or f3
    smoke = s1 or s2 or s3

    # ================= UI =================
    cam = resize_keep_ratio(cam, 480, 320)
    phone = resize_keep_ratio(phone, 480, 320)
    vid = resize_keep_ratio(vid, 960, 540)

    # ================= ALERT =================
    if fire or smoke:
        if time.time() - last_alert > 2:
            winsound.Beep(1200,500)
            last_alert = time.time()

            t = int(time.time())
            cv2.imwrite(f"{SAVE_FOLDER}/alert_{t}.jpg", vid)
            write_log("FIRE DETECTED")

    # ================= DASHBOARD =================
    top = np.hstack((cam, phone))
    full = np.vstack((top, vid))

    # ================= STATUS =================
    if fire:
        status = "FIRE"
        color = (0,0,255)
    elif smoke:
        status = "SMOKE"
        color = (200,200,200)
    else:
        status = "SAFE"
        color = (0,255,0)

    cv2.putText(full, f"STATUS: {status}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    cv2.imshow("🔥 AI Fire Monitoring FINAL", full)

    if cv2.waitKey(1) == 27:
        break

cap_cam.release()
cap_phone.release()
cap_vid.release()
cv2.destroyAllWindows()