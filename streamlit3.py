import streamlit as st
from PIL import Image
import numpy as np

import tensorflow
# Load pre-trained emotion recognition model
from tensorflow import keras
from keras.layers import Dense , Dropout
from keras.models import Sequential, load_model , model_from_json
# opening and store file in a variable

json_file = open(r'F:\emotion_facerecognition\final details of project_emotions\densnet\model_behaviour_analysis_densenet169_401 (2).json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights(r"F:\emotion_facerecognition\final details of project_emotions\densnet\best_weights_densenet169_401classes.hdf5")

print("Loaded Model from disk")


# Define emotion labels
emotion_labels = ['angry', 'eye_contact', 'happy', 'neutral', 'sad', 'smile', 'surprise']
# Function to preprocess image for DenseNet model
def preprocess_image(image):
    # Resize image to match model input size
    resized_image = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(resized_image)
    # Preprocess input for DenseNet model
    processed_image = img_array / 255.0
    return processed_image

# Function to predict top 3 emotions from image
def predict_top3_emotions(image):
    # Preprocess image
    processed_image = preprocess_image(image)
    # Expand dimensions to match model input shape
    input_image = np.expand_dims(processed_image, axis=0)
    # Make prediction
    prediction = loaded_model.predict(input_image)[0]
    # Get indices of top 3 predicted emotions
    top3_indices = prediction.argsort()[-3:][::-1]
    # Get top 3 predicted emotion labels and probabilities
    top3_emotions = [(emotion_labels[i], prediction[i]) for i in top3_indices]
    return top3_emotions

# Main Streamlit app
def main():
    st.title('Facial Emotion Recognition with DenseNet')
    st.write('Upload an image containing a face to detect emotions.')

    # Upload image file
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict top 3 emotions
        top3_emotions = predict_top3_emotions(image)
        
        # Display top 3 emotions and probabilities
        st.write('Top 3 Predicted Emotions:')
        for emotion, probability in top3_emotions:
            st.write(f'{emotion}: {probability:.2f}')

# Run the app
if __name__ == '__main__':
    main()