import cv2

cap=cv2.VideoCapture(0)
previous_gray = None
while True:
    ret, frame=cap.read()
    if not ret:
        print("failed to capture frame")
        break
    
    #grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is None:
        previous_gray = gray
        continue
    
    #difference of frames
    difference = cv2.absdiff(previous_gray, gray)
    
    #thresholding
    _, threshold = cv2.threshold(
        difference,
        30,
        255,
        cv2.THRESH_BINARY
    )
    
    #contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
    previous_gray = gray
    cv2.imshow("object analyzer",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()