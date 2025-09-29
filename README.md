# Hate-Speech-Detection
This project implements a deep learning model to detect hate speech in text using Natural Language Processing (NLP) techniques and TensorFlow. The model leverages tokenization, word embeddings, and a Bidirectional LSTM architecture for accurate classification. The model will classify hate speech into categories like "toxic", "threat", "insult" or "identity hate", providing an efficient and scalable solution for real-time content moderation.

# Project Workflow
1. Dataset Loading
   Dataset used: Jigsaw Toxic Comment Classification Challenge
   Load the CSV file using pandas.
2. Data Preprocessing
   Text Vectorization using TextVectorization layer
   Converts raw text into padded integer sequences ready for embedding.
3. Data Pipeline Construction
   Constructed using TensorFlow's tf.data API:
   map() — Apply transformations to the dataset.
   cache() — Cache data in memory for performance.
   shuffle() — Randomize the order of data.
   batch() — Group samples into batches.
   prefetch() — Overlap data preprocessing and model execution.
   This setup improves training efficiency and GPU utilization by preloading data.
4. Data Splitting
   Split dataset into training, validation, and test sets to evaluate model performance effectively.
5. Model Development
   Neural network architecture:
   Embedding Layer — Converts tokens to dense vectors.
   Bidirectional LSTM — Captures contextual information from both past and future states.
   Dropout Layer — Prevents overfitting.
   Dense Layers — For classification logic.
   Final Layer — Includes activation function (sigmoid) for binary classification.
6. Interface Implementation
   A simple user interface or API endpoint can be implemented to input text and return predictions (hate speech or not hate speech).
7. Model Evaluation
   Evaluate model on metrics like accuracy, precision, recall, and F1-score using the test dataset.
   Visualize performance via confusion matrix and training/validation curves.
