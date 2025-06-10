# Phishing-URL-Detection-with-RNN-and-LSTM
Project Overview

This project implements Simple Recurrent Neural Network (SimpleRNN) and Long Short-Term Memory (LSTM) models to detect phishing URLs using a publicly available dataset, involving data preprocessing, tokenization, and performance evaluation with accuracy, precision, recall, F1-score, and confusion matrices. The objective is to compare the effectiveness of SimpleRNN and LSTM in capturing spatial dependencies for phishing detection.

Features





Data Preprocessing: Cleans the dataset by removing duplicates and encoding labels for binary classification.



Tokenization and Padding: Converts URLs into tokenized sequences and pads them to a uniform length for RNN compatibility.



Model Training: Implements SimpleRNN and LSTM architectures with optimized hyperparameters for phishing URL detection.



Performance Evaluation: Assesses models using accuracy, precision, recall, F1-score, and confusion matrix visualizations.



Model Comparison: Visualizes performance differences between SimpleRNN and LSTM to highlight their strengths in detecting phishing patterns.

Project Structure

Phishing-URL-Detection-RNN/

├── Simple-RNN.py            # SimpleRNN model implementation and evaluation

├── LSTM-Model.py            # LSTM model implementation and evaluation

├── Accuracy-com.py          # Comparison of SimpleRNN and LSTM accuracies

├── phishing_site_urls.csv   # Input dataset

├── README.md                # Project documentation

Installation & Setup Instructions





Clone the Repository:

git clone https://github.com/m-abdullah3/Phishing-URL-Detection-RNN.git
cd Phishing-URL-Detection-RNN



Set Up a Python Environment:





Ensure Python 3.8+ is installed.



Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:





Install required libraries using pip:

pip install pandas scikit-learn tensorflow matplotlib seaborn numpy



Prepare the Dataset:





Place the phishing_site_urls.csv file in the project directory.

Download from https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls



Run the Models:





Train and evaluate the SimpleRNN model:

python Simple-RNN.py



Train and evaluate the LSTM model:

python LSTM-Model.py



Compare model accuracies:

python Accuracy-com.py

Tech Stack / Libraries Used





Python: Core programming language.



Pandas: Data manipulation and preprocessing.



Scikit-learn: Label encoding and train-test splitting.



TensorFlow/Keras: Deep learning model implementation (SimpleRNN, LSTM, Embedding).



Matplotlib/Seaborn: Visualization of confusion matrices and accuracy comparisons.



NumPy: Numerical operations for sequence padding.

License

This project is licensed under the MIT License - see the LICENSE file for details.
