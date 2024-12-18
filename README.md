# Turkish-Tweet-Classifier

This project implements a machine learning pipeline for classifying Turkish tweets into predefined categories using text preprocessing, TF-IDF feature extraction, and k-NN classification. The project supports stratified 10-fold cross-validation to evaluate the model's performance.

## Features
- **Text Preprocessing**: Includes tokenization, stopword removal, punctuation removal, and stemming with TurkishMorphology.
- **TF-IDF Representation**: Converts preprocessed text into a vectorized TF-IDF matrix for analysis.
- **Classification**: Uses k-Nearest Neighbors (k-NN) with cosine similarity for classification.
- **Cross-Validation**: Implements stratified 10-fold cross-validation to ensure robust evaluation.
- **Performance Metrics**: Calculates precision, recall, and F1 scores for each class, along with macro and micro averages.
