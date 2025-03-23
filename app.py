import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load the model
@st.cache_resource
def load_my_model():
    return load_model("models/clasifing_PLANE_DRONE_model.keras")

model = load_my_model()

# Prediction function
def predict(image):
    try:
        img = image.convert("RGB")  # here to make sure png images change to RBG so 3 channels
        img = img.resize((224, 224))  # Resize image
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_arr_proc = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_arr_proc)
        class_index = np.argmax(prediction, axis=1)
        class_labels = ['Drone', 'AirPlane']
        predicted_class_label = class_labels[class_index[0]]
        return predicted_class_label
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("AI Plane vs Drone Classifier")
st.write("Upload an image to classify whether it is a **plane** or **drone**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying:")
    result = predict(image)
    st.write(f"Prediction: **{result}**")
