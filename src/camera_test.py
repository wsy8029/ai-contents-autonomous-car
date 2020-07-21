import cv2

camera = cv2.VideoCapture(-1)

while camera.isOpened():
    _,frame = camera.read()
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break
