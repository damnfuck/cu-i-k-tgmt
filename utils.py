import cv2
import numpy as np

def detect_fire_smoke(frame, fgbg, kernel):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # FIRE
    lower_fire = np.array([0, 100, 100])
    upper_fire = np.array([35, 255, 255])

    # SMOKE
    lower_smoke = np.array([0, 0, 0])
    upper_smoke = np.array([179, 60, 120])

    mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)
    mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)

    fgmask = fgbg.apply(frame)

    mask_fire = cv2.bitwise_and(mask_fire, fgmask)
    mask_smoke = cv2.bitwise_and(mask_smoke, fgmask)

    mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)
    mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel)

    mask_fire = cv2.dilate(mask_fire, kernel, iterations=2)
    mask_smoke = cv2.dilate(mask_smoke, kernel, iterations=2)

    return mask_fire, mask_smoke