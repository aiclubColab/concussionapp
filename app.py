
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import pickle
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


# constants
IMG_SIZE = (224, 224)
IMG_ADDRESS = "https://insideclimatenews.org/wp-content/uploads/2023/03/wildfire_thibaud-mortiz-afp-getty-2048x1365.jpg"
IMAGE_NAME = "user_image.png"
CLASS_LABEL = ['Concussion', 'No Concussion']
CLASS_LABEL.sort()


@st.cache_resource
def get_MobileNetV2_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = keras.applications.MobileNetV2(include_top = False, weights = "imagenet", input_shape = (224, 224, 3), classes = 1000, classifier_activation = "softmax")
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions


# get the featurization model
MobileNetV2_featurized_model = get_MobileNetV2_model()
# load ultrasound image
classification_model = load_sklearn_models("best_ml_model")


# web app
def run_app():
    # title
    st.title("Wild Fire Classification")
    # image
    st.image(IMG_ADDRESS, caption = "Wild Fire Classification - Sattelite Images")

    # input image
    st.subheader("Please Upload a Sattelite Image")

    # file uploader
    image = st.file_uploader("Please Upload a Sattelite Image", type = ["jpg", "png", "jpeg"], accept_multiple_files = False, help = "Upload an Image")

    if image:
            user_image = Image.open(image)
            # save the image to set the path
            user_image.save(IMAGE_NAME)
            # set the user image
            st.image(user_image, caption = "User Uploaded Image")

            #get the features
            with st.spinner("Processing......."):
                image_features = featurization(IMAGE_NAME, MobileNetV2_featurized_model)
                model_predict = classification_model.predict(image_features)
                result_label = CLASS_LABEL[model_predict[0]]
                st.success(f"Prediction: {result_label}")