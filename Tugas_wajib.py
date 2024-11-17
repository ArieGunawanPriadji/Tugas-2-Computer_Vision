import cv2 as cv

cap = cv.VideoCapture('Video/object_video.mp4')

lower_bound = (0, 51, 121)  
upper_bound = (254, 255, 255)  

min_area = 100  

scale_factor = 0.25

out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Selesai memproses video.")
        break

    frame = cv.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)

    if out is None:
        height, width, _ = frame.shape
        out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv_frame, lower_bound, upper_bound)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > min_area:  
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)


            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow('Frame', frame)
    cv.imshow('Mask', mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
