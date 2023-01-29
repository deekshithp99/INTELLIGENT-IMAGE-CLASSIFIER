import streamlit as st
import tensorflow as tf
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model(r'intel_image.h5')
    return model
model=load_model()
st.write("""
         INTELLIGENT IMAGE CLASSIFIER
         """
        )
file=st.file_uploader("Please upload an image",type=["jpg","png","jpeg"])
import cv2
from PIL import Image
import numpy as np
def import_and_predict(image_data,model):
    image=image_data.resize((150,150))
    image1=tf.keras.preprocessing.image.img_to_array(image)
    image2=image1[np.newaxis,...]
    prediction=model.predict(image2)
    return prediction
if file is None:
    st.text("PLEASE UPLOAD AN IMAGE")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    predictions=import_and_predict(image,model)
    class_names=['buildings','forest', 'glacier' ,'mountain', 'sea', 'street']
    string="The given image is most likely to be a: "+ class_names[np.argmax(predictions)]
    st.success(string)
