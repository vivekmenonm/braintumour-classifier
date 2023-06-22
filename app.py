import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("braintumour_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess the image
def preprocess_image(image):
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Return the preprocessed image
    return normalized_image_array

# Function to predict the image class
def predict_image_class(image):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Load the image into the array
    data[0] = image
    
    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], round(confidence_score, 2)*100

# Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Brain Tumour Classifier", page_icon="🧠")

    st.title("Brain Tumour Classifier")

    # Upload image file
    uploaded_file = st.file_uploader("Upload the MRI", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        preprocessed_image = preprocess_image(image)
        
        # Predict the image class
        class_name, confidence_score = predict_image_class(preprocessed_image)

        # Display the prediction
        st.subheader("RESULT")
        st.write("Class:", "**" + class_name + "**")
        st.write("Prediction Probability:", "**" + str(confidence_score) + "%**")


# Run the app
if __name__ == "__main__":
    main()
