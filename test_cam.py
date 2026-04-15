import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"CAM {i}", frame)
        print("CAM", i, "OK")
    else:
        print("CAM", i, "FAIL")
    cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()