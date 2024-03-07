import cv2 
import tensorflow

from time import sleep
import tensorflow

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import model_from_json



from keras.preprocessing import image
import tensorflow

import numpy as np
class_labels = ['angry','eye_contact','happy','neutral','sad' , 'smile' , 'surprise']


json_file = open(r'F:\emotion_facerecognition\final details of project_emotions\densnet\model_behaviour_analysis_densenet169_401 (2).json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(r"F:\emotion_facerecognition\final details of project_emotions\densnet\best_weights_densenet169_401classes.hdf5")
print("Loaded Model from disk")

path = r'F:\emotion_facerecognition\haarcascade_frontalface_default.xml'
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set background to white 
rectangle_bar = (255,255,255)

#make black image
img = np.zeros((500 , 500))

#set some text 
text = 'Some text in box'

(text_width , text_height) = cv2.getTextSize(text , font , fontScale=font_scale , thickness = 1)[0]

text_offset_x = 10
text_offset_y = img.shape[0] - 25

#make the coordinates of box with small padding of two pixels
box_cords = ((text_offset_x , text_offset_y) , (text_offset_x + text_width +2 , text_offset_y-text_height-2))
cv2.rectangle(img,box_cords[0] , box_cords[1] , rectangle_bar , cv2.FILLED)

cap = cv2.VideoCapture(1)

cv2.destroyAllWindows()


if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('cannot open webcam')

while True:
    ret , frame = cap.read()
    facecascade = cv2.CascadeClassifier('F:\emotion_facerecognition\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray  , 1.1 , 4)
    for x,y,w,h in faces :
        roi_gray = gray[y:y+h , x:x+w]
        roi_colr = frame[y:y+h , x:x+w]
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255 , 0 , 0) , 2)
        facess = facecascade.detectMultiScale(roi_gray)
        if len(facess) ==0:
            print('face not detected')
        else:
            for (ex , ey , ew , eh) in facess:
                face_roi = roi_colr[ey:ey+eh , ex:ex+ew] # crop face
                final_iamge = cv2.resize(face_roi , (224 ,224))
                final_iamge = np.expand_dims(final_iamge,axis = 0)
                final_iamge = final_iamge/255.0


            if np.sum([final_iamge])!=0:

            

        # make a prediction on the ROI, then lookup the class

                preds = loaded_model.predict(final_iamge)[0]
                print("\nprediction = ",preds)
                label=class_labels[preds.argmax()]
                print("\nprediction max = ",preds.argmax())
                print("\nlabel = ",label)
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            print("\n\n")
        cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
