import cv2
from ultralytics import YOLO
from data.classes import classNames

model = YOLO("yolo-Weights/yolov8n.pt")

cap = cv2.VideoCapture("./data/sample.mp4")
cap.set(3, 640)
cap.set(4, 480)

# Processing the video every n frames
n = 8

# Assigning frames from 0
frame_counter = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    results = model(img, stream=True)

    frame_counter += 1
    if frame_counter % n != 0:
        continue

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 3)

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()