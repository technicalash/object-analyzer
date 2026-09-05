import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    if not ret:
        print("failed to capture frame")
        break
    results = model(frame, verbose=False)
    result = results[0]
    boxes = result.boxes
    
    for box, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        
        x1, y1, x2, y2 = map(int, box.tolist())
        confidence = float(confidence)
        if confidence<0.5:
            continue
        class_id = int(class_id)
        class_name = model.names[class_id]
        cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    cv2.imshow("YOLO Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()