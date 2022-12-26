import cv2

cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

faceDet = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
eyeDet = cv2.CascadeClassifier("haarcascade_eye.xml")
mouthDet = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

facePart = cv2.imread('parts/face.png')
eyePart = cv2.imread('parts/eye.png')
mouthPart = cv2.imread('parts/mouth.png')

while (cam.isOpened()):
    ret, frame = cam.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = faceDet.detectMultiScale(gframe)
    eye = eyeDet.detectMultiScale(gframe)
    mouth = mouthDet.detectMultiScale(gframe)

    if(len(face)>0):
        x, y, w, h = face[0]
        faceRes = cv2.resize(facePart, (w+20, h+20))
        frame[y-10:y+h+10, x-10:x+w+10] = faceRes
        if(len(eye)>0):
            for x, y, w, h in eye[0:2]:
                eyeRes = cv2.resize(eyePart, (w,h))
                frame[y:y+h, x:x+w] = eyeRes
        if(len(mouth)>0):
            x, y, w, h = mouth[0]
            mouthRes = cv2.resize(mouthPart, (w,h))
            frame[y:y+h, x:x+w] = mouthRes
        
    cv2.imshow("Bui Tuver", frame)

    if (cv2.waitKey(1)&0xff == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
