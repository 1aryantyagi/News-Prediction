# News Prediction Project

## Overview
The News Prediction project is a machine learning application designed to classify news articles into predefined categories. It utilizes a Convolutional Neural Network (CNN) for text classification, leveraging a pre-trained Word2Vec model for word embeddings. The application is built using Flask, allowing users to train the model and make predictions via a web interface.

## Project Structure
The project consists of the following files:

- *app.py*: The main entry point of the application. It sets up a Flask web server and defines two routes: /train for training the model and /predict for making predictions. It utilizes the PredictionService class to manage model loading and predictions.

- *cnn_model.py*: This file defines the NewsCNN class, which is responsible for building the convolutional neural network model. It includes methods to create the model architecture, including convolutional layers, pooling layers, and dense layers.

- *data_preprocessor.py*: This file contains the DataPreprocessor class, which handles text preprocessing tasks. It includes methods for cleaning text, tokenizing, and converting text to word vectors using a pre-trained Word2Vec model.

- *model_trainer.py*: This file defines the ModelTrainer class, which is responsible for training the CNN model. It includes methods to load data, preprocess it, train the model, and save the trained model along with the label encoder.

- *requirements.txt*: This file lists the dependencies required for the project, including Flask, Keras, NumPy, Pandas, and Gensim.

## Setup Instructions
1. *Clone the Repository*: 
   Clone the repository to your local machine using:
   
   git clone https://github.com/1aryantyagi/News-Prediction
   

2. *Install Dependencies*: 
   Navigate to the project directory and install the required packages using:
   
   pip install -r requirements.txt
   

3. *Download Word2Vec Model*: 
   Ensure you have the GoogleNews-vectors-negative300.bin file available in the project directory for word vectorization.

## Usage
1. *Start the Flask Server*: 
   Run the application using:
   
   python app.py
   
   The server will start on http://localhost:5000.

2. *Train the Model*: 
   To train the model, send a POST request to the /train endpoint with the training data file and model path. The data file should be a CSV containing two columns: text and label.

3. *Make Predictions*: 
   To make predictions, send a POST request to the /predict endpoint with a JSON payload containing the model_path and the text to be classified.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- This project utilizes the Keras library for building the CNN model.
- The Gensim library is used for handling word embeddings.
