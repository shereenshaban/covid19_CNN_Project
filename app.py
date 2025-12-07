
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image


st.title("COVID-19 X-Ray Classifier")
classes=["COVID","Normal","Viralpneumonia"]
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Coved.keras")
model=load_model()
if model:
    st.success("model loaded")
    upload_image=st.file_uploader("upload x_Ray image",type=['jpg','png','jpeg'])
    if upload_image:
        img=Image.open(upload_image)
        st.image(img)
        if st.button("predicte"):
            img_rgb=img.convert("RGB")
            img_resize=img_rgb.resize((224,224))
            img_arr=image.img_to_array(img_resize)/255
            img_array=np.expand_dims(img_arr,axis=0)
            prediction=model.predict(img_array)     
            predicted_index = np.argmax(prediction, axis=1)[0]
            st.subheader(f"prediction: {classes[predicted_index]}")
            st.write(f"confidance:{prediction}")

