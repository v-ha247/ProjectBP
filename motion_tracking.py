import cv2

# cap = cv2.VideoCapture("D:/System/OneDrive/Obrázky/silvestr/20181231_205201.mp4")
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()
while True:  
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    # cv2.imshow("diff", cv2.resize(diff, (1280, 720)))

    blur = cv2.GaussianBlur(diff, (5, 5), 0) 
    # cv2.imshow("blur", cv2.resize(blur, (1280, 720)))
    
    _ , thresholdimg = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresholdimg", cv2.resize(thresholdimg, (1280, 720)))

    # dilated = cv2.dilate(thresholdimg, None, iterations=3)
    # cv2.imshow("dilated", cv2.resize(dilated, (1280, 720)))

    contours, _ = cv2.findContours(thresholdimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.drawContours(frame2, contours, -1, (0, 255, 0), 2)
    if motion:
        (x, y, w, h) = area
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (255,0,0), 2) 

    cv2.imshow("Frame", cv2.resize(frame2, (1280, 720)))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    frame2 = frame1
    ret, frame1 = cap.read()
    if not ret:
        continue

cap.release()
cv2.destroyAllWindows()