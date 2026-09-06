import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
cap=cv2.VideoCapture(0)
previous_centers={}
trajectory_canvas = None

while True:
    ret, frame=cap.read()
    if not ret:
        print("failed to capture frame")
        break
    if trajectory_canvas is None:
        trajectory_canvas = np.zeros_like(frame)
        
    results = model.track(frame, persist=True, verbose=False)
    result = results[0]
    boxes = result.boxes
    if boxes.id is None:
        continue
    
    for box, confidence, class_id, track_id  in zip(boxes.xyxy, boxes.conf, boxes.cls, boxes.id):
        
        x1, y1, x2, y2 = map(int, box.tolist())
        confidence = float(confidence)
        if confidence<0.3:
            continue
        track_id = int(track_id)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)
        previous_center = previous_centers.get(track_id)
        if previous_center is not None:
            dx = center_x - previous_center[0]
            dy = center_y - previous_center[1]
            
            cv2.line(
                trajectory_canvas,
                previous_center,
                center,
                (255, 0, 0),
                2
            )
        previous_centers[track_id] = center

        
        class_id = int(class_id)
        class_name = model.names[class_id]
        cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        label = f"{class_name} | ID: {track_id} | {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
    output = cv2.add(frame, trajectory_canvas)
    cv2.imshow("YOLO Object Detection", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()