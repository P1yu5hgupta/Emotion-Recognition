import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from model_utils import define_model, model_weights
import tensorflow as tf
from tensorflow import keras

def main():
    model = define_model()
    model = model_weights(model)
    print('Model loaded')

    result = np.array((1,7))
    faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    # list of given emotions
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

    # store the emoji coreesponding to different emotions
    emoji_faces = []
    save_loc = 'save_loc/1.jpg'
    for index, emotion in enumerate(EMOTIONS):
        emoji_faces.append(cv2.imread('emojis/' + emotion.lower()  + '.png', -1))
    # set video capture device , webcam in this case
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)  # WIDTH
    video_capture.set(4, 480)  # HEIGHT

    prev_time = time.time()
    once =False

    while True:
        ret, frame = video_capture.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find face in the frame
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x-10, y-70),(x+w+20, y+h+40), (15, 175, 61), 4)

            # roi_gray = gray[y:y+h, x:x+w]
            color_img = frame[y-90:y+h+70, x-50:x+w+50]
            curr_time = time.time()
            cv2.imwrite(save_loc, color_img)
            img = cv2.imread(save_loc, 0)
            
            if img is not None:
                if curr_time - prev_time >=1:

                    img = cv2.resize(img, (48, 48))
                    img = np.reshape(img, (1, 48, 48, 1))
                    # do prediction
                    result = model.predict(img)
                    prev_time = time.time()
                    total_sum = np.sum(result[0])
                    once=True
                    # select the emoji face with highest confidence
                    emoji_face = emoji_faces[np.argmax(result[0])]
            if once==True:
                for index, emotion in enumerate(EMOTIONS):
                    text = str(round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                    # for drawing progress bar
                    cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                    (255, 0, 0), -1)
                    # for putting emotion labels
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                    # for putting percentage confidence
                    cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
                for c in range(0, 3):
                    # for doing overlay we need to assign weights to both foreground and background
                    foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
            break
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()