import cv2
body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

vid = cv2.VideoCapture('walking.avi')

while True:
    ret, frame = vid.read()
    grayscale_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(grayscale_video)
    print(len(bodies))

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Bodies", frame)

    if cv2.waitKey(25) == 32:
        break

vid.release()
cv2.destroyAllWindows()