import cv2
import numpy as np
import pyscreenshot as ImageGrab
import tensorflow as tf

X_COORD = 50
Y_COORD = 200
WIDTH = 800
HEIGHT = 350 + 120

IMG_WIDTH = 100
IMG_HEIGHT = 100

model = tf.keras.models.load_model('model-017.model')
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'Mask', 1:'No Mask'}
color_dict={0:(0,255,0), 1:(0,0,255)} # change this

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    screen_grab = np.array(ImageGrab.grab(bbox= (X_COORD, Y_COORD, X_COORD + WIDTH, Y_COORD + HEIGHT)))
    gray = cv2.cvtColor(screen_grab,cv2.COLOR_BGR2GRAY) # Gray image
    color = cv2.cvtColor(screen_grab,cv2.COLOR_BGR2RGB) # Color image

    faces = detector.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30, 30),
       )

    for(x,y,w,h) in faces:
        gray1 = gray[y:y+h,x:x+w]
        gray_resize = cv2.resize(gray1, (IMG_WIDTH, IMG_HEIGHT)) # Resizing
        gray_normal = np.array(gray_resize) / 255 # Normalizing
        image = gray_normal.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1) # Reshaping
    
        prediction = model.predict(image)
        prob = np.amax(prediction) * 100
        image_id = np.argmax(prediction)
        label = labels_dict[image_id]
        # print(prob)
        # print(label)
        cv2.rectangle(color, (x,y), (x+w,y+h), color_dict[image_id], 2)
        cv2.putText(color, str(label), (x+5,y-5), font, 0.45, color_dict[image_id], 1)
        cv2.putText(color, f"{round(prob, 2)}%", (x+70,y-5), font, 0.4, color_dict[image_id], 1)
        
    cv2.imshow('Real Time Mask Detection',color) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cv2.destroyAllWindows()


