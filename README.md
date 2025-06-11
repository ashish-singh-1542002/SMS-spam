# SMS/Email Spam Detector: Advanced Spam Filtering Engine

## Table of Contents
- [Overview](#overview)
- [Dataset and Data Cleaning](#dataset-and-data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Model Performance Metrics](#model-performance-metrics)

## Overview
SMS/Email Spam Detector is an intelligent, machine-learning-powered tool designed to accurately classify spam in text messages and emails.

## Dataset and Data Cleaning

The dataset was significantly enhanced by fine-tuning GPT-3.5 to generate synthetic SMS data, improving its diversity and realism for model training.

- Utilized Spam-Ham dataset from Kaggle, enriched with additional spam messages.
- Final composition: 69% ham, 31% spam.
- Data cleaning involved the removal of duplicates and null values.

## Exploratory Data Analysis
- Used NLTK to derive metrics like the number of characters, words, and sentences in messages.
- Calculated statistical measures (mean, std, percentiles, etc.) to understand message trends.
- Visualizations include pairplots and heatmaps for feature correlation.
- Identified maximum correlation between the number of characters and the message label (spam/ham).

## Data Preprocessing
- Employed NLTK for text preprocessing tasks such as lowercasing, stemming/lemmatization, tokenization, and removal of stopwords.
- Identified the top 10 words used in each type of message (spam/ham).

## Machine Learning Algorithms

### Individual Classifiers
- Initialized classifiers with custom hyperparameters like `SVC(kernel='sigmoid', gamma=1.0)`, `KNeighborsClassifier()`, `BernoulliNB()`, etc.

### Ensemble Methods
- Best performing classifiers were Bernoulli Naive Bayes (NB), ExtraTreesClassifier (ETC), and Support Vector Classifier (SVC).
- Applied Voting and Stacking classifiers to combine the top-performing models for optimal results.

## Model Performance Metrics
- Achieved outstanding accuracy, precision, recall, and F1 scores. See the table below for metrics:

| Method | Accuracy  | Precision | Recall   | F1 Score |
|--------|-----------|-----------|----------|----------|
| NB     | 0.978888  | 0.980392  | 0.947867 | 0.963855 |
| ETC    | 0.973962  | 0.972973  | 0.938389 | 0.955368 |
| SVC    | 0.971851  | 0.972772  | 0.931280 | 0.951574 |

- Utilized Python's `pickle` library to serialize the final model for deployment.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The project is licensed under the MIT License.
