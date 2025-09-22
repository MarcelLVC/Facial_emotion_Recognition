import cv2
import numpy as np
from keras.models import load_model

model = load_model('model_50Epoch.h5')

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

label_dict ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'sad',6:'Surprise'}

# frame = cv2.imread('testpic1.jpg')

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 3)

    for x,y,w,h in faces:
        sub_face = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face, (48, 48))
        normalize = resized/255
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x,y-30), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, label_dict[label], (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    
    cv2.imshow('Emotion Recognition',frame)
    end = cv2.waitKey(1)

    # press q to end
    if end==ord('q'):
        break

video.release()
cv2.destroyAllWindows()

