
# Deep Learning Fashion Recommender System

Fashion Recommender System is a state-of-the-art fashion recommender system that harnesses the power of deep learning to provide personalized fashion recommendations. This project employs advanced image processing techniques and a sophisticated recommendation algorithm to curate a unique fashion experience tailored to each user's preferences.

## Features

- Seamlessly upload fashion images for personalized recommendations.
- Leverage deep learning models for image feature extraction and analysis.
- Discover visually similar fashion items using content-based filtering.
- Explore a diverse range of fashion products from a comprehensive dataset.
- Fine-tune and extend the recommender system to adapt to evolving trends.

## How It Works

System employs the ResNet50 deep learning model pre-trained on ImageNet to extract rich feature embeddings from fashion product images. These embeddings are then used to calculate visual similarities among fashion items, enabling the system to provide top-notch recommendations. The use of Nearest Neighbors and content-based filtering ensures a highly personalized shopping experience for every user.

## Installation

1. Clone the repository

2. Download the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and place the `images` folder in the project directory.

3. Run the `main.py` script to generate the necessary feature embeddings and filenames pickle files:

4. Launch the Streamlit app:
   ```
   streamlit run frontend.py
   ```

## Dataset

FashionVue is built upon the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle. This extensive dataset provides a diverse collection of +45,000 fashion product images, forming the foundation for our recommendation engine.

