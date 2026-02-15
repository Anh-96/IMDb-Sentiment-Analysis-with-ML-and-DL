# 🎬 IMDb Sentiment Analysis with Machine Learning & Deep Learning

An end-to-end NLP project that compares traditional Machine Learning, Deep Learning, and Transformer-based models for sentiment classification on movie reviews.

This repository demonstrates the full workflow from raw text → preprocessing → feature engineering → classical ML → deep learning → BERT fine-tuning → model comparison.


📌 Project Overview

This project focuses on classifying IMDb movie reviews into:

* Positive 😊

* Negative 😞

and answering a practical question:

  How much better is BERT compared to traditional ML and Deep Learning models?

The goal is not only accuracy, but also understanding:

* When simple models are enough

* When deep learning helps

* When transformers truly shine


## 🧠 Models Implemented
🟢 Traditional ML

* TF-IDF + Logistic Regression (baseline)

* TF-IDF + SVM

* Hyperparameter tuning with GridSearch

🔵 Deep Learning

Neural Network baseline

CNN for text classification

LSTM / BiLSTM

🔴 Transformer

BERT fine-tuning for sentiment classification


## 📂 Project Structure
00_install.ipynb              → Environment setup
01_eda.ipynb                  → Exploratory Data Analysis
02_text_cleaning.ipynb        → Text preprocessing pipeline
03_tfidf_vectorization.ipynb  → TF-IDF feature engineering
04_logistic_regression_baseline.ipynb
05A_model_tuning.ipynb        → GridSearch tuning
05B_model_comparison.ipynb    → ML model comparison
06_baseline_deep_learning.ipynb
07_cnn.ipynb                  → CNN model
08_lstm_bilstm.ipynb          → LSTM & BiLSTM
09_bert.ipynb                 → BERT fine-tuning


## 🔍 Workflow
1️⃣ EDA

* Review length distribution

* Class balance

* Text statistics

2️⃣ Text Cleaning

* Lowercasing

* Removing HTML tags

* Removing punctuation

* Stopword removal

* Lemmatization

3️⃣ Feature Engineering

* TF-IDF (unigram & bigram)

* max_features tuning

4️⃣ Classical ML Baselines

* Logistic Regression

* SVM

* GridSearch optimization

5️⃣ Deep Learning Models

* Tokenization & padding

* Embedding layers

* CNN & LSTM architectures

6️⃣ Transformer (BERT)

* Tokenization with pretrained model

* Fine-tuning on IMDb reviews

* Evaluation using F1 score


## 📊 Key Insights

* From experimentation across models:

* TF-IDF + Logistic Regression is a strong baseline

* CNN/LSTM improve performance but require more tuning

* BERT delivers the best contextual understanding

* Transformers outperform others on:

  - long reviews

  - mixed sentiment

  - ambiguous language

However:

* ML models train extremely fast ⚡

* BERT is computationally expensive 🐢

So in real production:

  Simple models can still be very competitive.


## 📈 Evaluation Metrics

* Accuracy

* F1 Score (main metric)

* Precision / Recall

* Confusion Matrix

* Error analysis on misclassified samples

## 🧪 Example Predictions

The models were tested on difficult, ambiguous reviews such as:

* "It wasn’t bad, but I wouldn’t watch it again."

* "Strangely enjoyable despite its flaws."

These cases highlight the advantage of contextual models like BERT.


## 🛠️ Tech Stack

* Python

* Scikit-learn

* TensorFlow / Keras

* PyTorch

* HuggingFace Transformers

* Pandas / NumPy / Matplotlib


## 🎯 What This Project Demonstrates

This project showcases:

* End-to-end NLP pipeline design

* Feature engineering for text data

* Model comparison methodology

* Hyperparameter tuning

* Deep learning for NLP

* Transformer fine-tuning

* Error analysis mindset


## 🚀 Possible Improvements

Future directions:

* DistilBERT (faster alternative to BERT)

* RoBERTa fine-tuning

* Ensemble models

* Deployment as an API

* Real-time sentiment prediction UI


## 📎 Dataset

IMDb movie reviews dataset:

* ~50,000 labeled reviews

* Balanced positive/negative classes

* Widely used NLP benchmark


## 👨‍💻 Author

Built as a hands-on learning project to explore:

* Machine Learning

* Deep Learning

* NLP pipelines

* Transformer models
