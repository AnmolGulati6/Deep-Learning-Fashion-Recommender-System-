# Written by Anmol Gulati
# Import the required libraries

import pickle
import tensorflow
import numpy as np
import cv2
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load precomputed feature embeddings and corresponding filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb' ))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()  # Add a max pooling layer to extract global features
])

# Load and preprocess the sample image for feature extraction
img = image.load_img('sample/clothing3.jpeg', target_size=(224, 224))  # Load and resize the image
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess the image for the ResNet50 model
result = model.predict(preprocessed_img).flatten()  # Extract features using the model
normalized_result = result / norm(result)  # Normalize the features

# Initialize a NearestNeighbors model for finding similar items
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)  # Fit the model with precomputed feature embeddings

# Find nearest neighbors based on the normalized features of the sample image
distances, indices = neighbors.kneighbors([normalized_result])

# Print the indices of the nearest neighbors
print(indices)

# Display the nearest neighbor images using OpenCV
for file in indices[0][1:6]:  # Iterate through the indices of the nearest neighbors (excluding the sample image)
    temp_img = cv2.imread(filenames[file])  # Read the image using OpenCV
    cv2.imshow('output', cv2.resize(temp_img, (512,512)))  # Resize and display the image
    cv2.waitKey(0)  # Wait for a key press before closing the image window
