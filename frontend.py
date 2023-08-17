# Written by Anmol Gulati
# Import required libraries
import streamlit as st
import os
import numpy as np
import pickle
import tensorflow
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

# Load precomputed feature embeddings and corresponding filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()  # Add a max pooling layer to extract global features
])

# Set the title for the Streamlit app
st.title('Fashion Recommender System')


# Define a function to save an uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


# Define a function to extract features from an image using the ResNet50 model
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Load and resize the image
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess the image for the ResNet50 model
    result = model.predict(preprocessed_img).flatten()  # Extract features using the model
    normalized_result = result / norm(result)  # Normalize the features

    return normalized_result


# Define a function to recommend similar fashion items based on features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])  # Find nearest neighbors based on features
    return indices


# Allow the user to upload an image file
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):  # Save the uploaded file
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Extract features from the uploaded image
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)
        # st.text(features)

        # Recommend similar fashion items based on features
        indices = recommend(features, feature_list)

        # Display the recommended fashion items using columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])
    else:
        st.header("Error in file upload")  # Display an error message if file upload fails
