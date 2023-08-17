# Written by Anmol Gulati
# Import the required libraries
import tensorflow
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from tqdm import tqdm

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a Keras Sequential model with the ResNet50 base and a GlobalMaxPooling2D layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()  # Add a max pooling layer to extract global features
])

# Define a function to extract features from an image using the ResNet50 model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Load and resize the image
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess the image for the ResNet50
    result = model.predict(preprocessed_img).flatten()  # Extract features using the model
    normalized_result = result / norm(result)  # Normalize the features

    return normalized_result


# Create a list to store file names in the 'images' directory
filenames = []

# Iterate through files in the 'images' directory and add their paths to the list
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# Display the total number of files and the first 5 file names
# print(len(filenames))  # Uncomment to print total number of files
# print(filenames[0:5])  # Uncomment to print first 5 file names

# Create a list to store extracted features from images
feature_list = []

# Iterate through the list of filenames and extract features using the ResNet50 model
for file in tqdm(filenames):  # Use tqdm for a progress bar during the iteration
    feature_list.append(extract_features(file, model))

# Save the extracted feature list and filenames as pickle files
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

