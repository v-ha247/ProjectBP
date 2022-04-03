import cv2

# cap = cv2.VideoCapture("D:/System/OneDrive/Obrázky/silvestr/20181231_205201.mp4")
cap = cv2.VideoCapture(0)

ret, frame2 = cap.read()
while True:
    ret, frame1 = cap.read()
    if not ret:
        continue
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame2 = frame1

    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0) 
    _ , thresholdimg = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresholdimg, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = False
    area = ()
    if contours != ():
        biggest_motion = 2000
        for c in contours:
            found_area = cv2.contourArea(c)   
            if found_area > biggest_motion:
                biggest_motion = found_area
                area = cv2.boundingRect(c)
                motion = True       
        else:
            motion_center = ()
    if motion:
        (x, y, w, h) = area
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255,0,0), 2) 
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Frame", cv2.resize(frame1, (1280, 720)))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()