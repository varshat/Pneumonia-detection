import streamlit as st
from PIL import Image
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Adamax

st.title('Covid/Pneumonia Detection')

uploaded_image = st.file_uploader('Choose an image')
if uploaded_image is not None:
    with st.spinner(text="Fetching measures"):
    
        image_path = uploaded_image #'C:\\Users\\varsh\\Downloads\\xray.jpeg'
        image = Image.open(image_path)
        print(image)


        # Preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Prediction on loaded model
        loaded_model = tf.keras.models.load_model('D:\\Data Science\\Deep Learning Projects\\Pneumonia-Detection\\model.h5', compile=False)
        loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])


        # Make predictions
        predictions = loaded_model.predict(img_array)
        class_labels = ['Covid', 'Normal', 'Viral Pneumonia']
        score = tf.nn.softmax(predictions[0])       
        
        if (class_labels[tf.argmax(score)]=='Covid'):   
            st.title("Covid detected")
            st.image("images\\dashboard_img.gif")
            print(f"{class_labels[tf.argmax(score)]}")
        elif (class_labels[tf.argmax(score)]=='Normal'):   
            st.title("Normal Lungs")
            st.image("images\\normal.gif")
            print(f"{class_labels[tf.argmax(score)]}")
        elif (class_labels[tf.argmax(score)]=='Viral Pneumonia'):   
            st.title("Pneumonia detected")
            st.image("images\\pneumonia.gif")
            print(f"{class_labels[tf.argmax(score)]}")
    



