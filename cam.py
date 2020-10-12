#user = input()
#user = str(user)

import numpy as np

from keras.preprocessing import image
from keras.models import model_from_json
import time
import tensorflow as tf
#GG
#load model
model = model_from_json(open("Model_Weights/fer238.json", "r").read(), custom_objects={'softmax_v2': tf.nn.softmax})
#load weights
model.load_weights("Model_Weights/fer238.h5")

import cv2
# Load the cascade
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class VideoCamera(object):

    def __init__(self):
        #self.video=cv2.VideoCapture('Video/Debate.mp4')
        self.video=cv2.VideoCapture(0)
        time.sleep(5.0)

    def __del__(self):
    #releasing camera
        self.video.release()

    def get_frame(self):
    

        ret,test_img=self.video.read()# captures frame and returns boolean value and captured image
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        # Face properties

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x,y,w,h) in faces_detected:
            
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=4)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray1=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray1)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            predictions = model.predict(img_pixels)   
                
            #find max indexed array
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            #u.append(str(user))
            
            label = emotions[np.argmax(predictions)]  # Get label with most probability
            confidence = np.max(predictions).round(decimals = 2) 
            #confidence *= 100 # Multiple probability by 100
            detect = dict()
            detect['label'] = label
            detect['score'] = str(max_index).split(".")[0]
            cv2.putText(test_img, "You look " + predicted_emotion +":" +str(confidence), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
           

                      
            
        resized_img = cv2.resize(test_img, (1000, 700))
        #cv2.imshow('Facial emotion analysis',resized_img)
        # Window name in which image is displayed 
        #window_name = 'Facial Emotion analysis'
    
        # text 
        text = 'Emotion analysis Detector - Powered by ProSyn3'
        text1 = 'Enjoy your picture'
    
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        # org 
        org = (50, 50)
        org1 = (50, 600)
    
        # fontScale 
        fontScale = 1
    
        # Red color in BGR 
        color = (0, 0, 255) 
    
        # Line thickness of 2 px 
        thickness = 2
    
        # Using cv2.putText() method 
        resized_img = cv2.putText(resized_img, text, org, font, fontScale,  
                         color, thickness, cv2.LINE_AA, False)
        resized_img = cv2.putText(resized_img, text1, org1, font, fontScale,  
                     color, thickness, cv2.LINE_AA, False) 
        _, jpeg = cv2.imencode('.jpg', resized_img)
    
        # Using cv2.putText() method 
        #resized_img = cv2.putText(resized_img, text, org, font, fontScale, 
                          #color, thickness, cv2.LINE_AA, True)
        #cv2.putText(resized_img, "ProSyn3" , (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return jpeg.tobytes()





 

        
