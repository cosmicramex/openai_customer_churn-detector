# NLP-Based Customer Churn Prediction System

An end-to-end Natural Language Processing (NLP) pipeline that predicts customer churn from written feedback and generates analytical visualizations.

This project analyzes user reviews of ChatGPT to detect dissatisfaction signals that indicate a potential churn event. The system processes raw text feedback, extracts linguistic features, trains machine learning models, evaluates performance, and produces visual reports.

---

# Project Overview

Customer churn occurs when users stop using a product or service. Before churn happens, customers often express dissatisfaction in the form of written feedback such as reviews, complaints, or support messages.

Manually analyzing thousands of such messages is impractical. This project builds an automated system that:

- Processes customer feedback text
- Extracts meaningful linguistic features
- Trains machine learning models to predict churn
- Evaluates model performance
- Generates analytical visualizations

The system was trained using **ChatGPT user review datasets** to identify patterns associated with churn intent.

---

# Features

• End-to-end NLP pipeline  
• Text preprocessing and cleaning  
• Tokenization and lemmatization  
• Stop-word removal  
• N-gram feature extraction  
• TF-IDF vectorization  
• Machine learning model training  
• Model evaluation metrics  
• Churn prediction interface  
• Visualization reports using Matplotlib and Seaborn  

---

# Example Use Case

Example customer feedback:

```
"The service is too slow and expensive. I am thinking about cancelling."
```

System output:

```
Churn Probability: 0.91
Prediction: Likely to Churn
```

---

# System Architecture

```
Customer Feedback Dataset
        |
        v
Text Preprocessing
        |
        v
Tokenization + Lemmatization
        |
        v
Stopword Removal
        |
        v
Feature Extraction (N-grams + TF-IDF)
        |
        v
Machine Learning Models
        |
        v
Prediction
        |
        v
Evaluation Metrics
        |
        v
Visualization Reports
```

---

# Dataset

The model was trained using **ChatGPT user reviews** collected from public review sources.

Dataset format:

```
feedback,churn
"service is slow and expensive",1
"excellent product and helpful responses",0
"thinking about cancelling subscription",1
"very useful and fast responses",0
```

Label definitions:

```
1 → Customer churn
0 → Customer retained
```

Recommended dataset size:

```
Minimum: 500 records
Preferred: 1000–5000 records
```

---

# Project Structure

```
nlp-churn-prediction/

dataset/
customer_churn_dataset.csv

src/
data_loader.py
preprocessing.py
feature_extraction.py
model_training.py
prediction.py
visualization.py

notebooks/
analysis.ipynb

results/
accuracy_comparison.png
churn_distribution_pie.png
confusion_matrix.png
prediction_distribution.png
important_words_chart.png

reports/
project_report.pdf
presentation.ppt

README.md
requirements.txt
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/nlp-churn-prediction.git
cd nlp-churn-prediction
```

Create virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Dependencies

```
pandas
numpy
nltk
spacy
scikit-learn
matplotlib
seaborn
jupyter
```

Download NLP resources:

```
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
```

Download spaCy model:

```
python -m spacy download en_core_web_sm
```

---

# Text Preprocessing Pipeline

The preprocessing module performs several cleaning steps.

### 1 Lowercase Conversion

```
"Service Is Slow"
→ "service is slow"
```

### 2 Remove Punctuation

```
"service is slow!!!"
→ "service is slow"
```

### 3 Remove Numbers

```
"service is slow 123"
→ "service is slow"
```

### 4 Remove Extra Spaces

```
"service   is   slow"
→ "service is slow"
```

---

# Tokenization

Tokenization splits sentences into words.

Example:

```
Input:
"service is too slow"

Output:
["service", "is", "too", "slow"]
```

---

# Stop Word Removal

Common words are removed.

Example:

```
Input:
"service is too slow"

Output:
["service", "slow"]
```

Stop words include:

```
is
the
and
to
with
am
are
```

---

# Lemmatization

Lemmatization converts words to base forms.

Examples:

```
running → run
cancelled → cancel
services → service
```

Example transformation:

```
"customers cancelling services"
→ ["customer", "cancel", "service"]
```

---

# Feature Extraction

Machine learning models require numerical features.

Two feature extraction techniques are used.

---

## N-Grams

N-grams capture word sequences.

Example sentence:

```
"customer support is bad"
```

Unigrams:

```
customer
support
bad
```

Bigrams:

```
customer support
support bad
```

Important churn indicators:

```
cancel subscription
poor support
bad service
```

---

## TF-IDF Vectorization

TF-IDF converts text into numerical vectors.

Example vector:

```
[0.32, 0.01, 0.44, 0.00, 0.12]
```

Each value represents the importance of a word relative to the dataset.

---

# Machine Learning Models

Three classification models are trained and compared.

---

## Naive Bayes

Advantages:

```
Fast training
Efficient for text classification
Performs well on sparse data
```

---

## Logistic Regression

Advantages:

```
Interpretable
Strong baseline model
Good performance on NLP tasks
```

---

## Random Forest

Advantages:

```
Captures complex patterns
Robust against overfitting
Handles nonlinear relationships
```

---

# Model Training

Training pipeline:

```
1 Load dataset
2 Preprocess text
3 Extract features (TF-IDF)
4 Split dataset
5 Train models
6 Evaluate performance
```

Dataset split:

```
Training Data: 80%
Testing Data: 20%
```

---

# Evaluation Metrics

Several metrics are used to evaluate model performance.

### Accuracy

Percentage of correct predictions.

### Precision

Proportion of predicted churn cases that are correct.

### Recall

Proportion of actual churn cases detected.

### F1 Score

Balance between precision and recall.

Example results:

```
Accuracy = 0.86
Precision = 0.84
Recall = 0.82
F1 Score = 0.83
```

---

# Prediction System

Example input:

```
"I am unhappy with customer support and will cancel soon."
```

Processing steps:

```
Text cleaning
Tokenization
Stop word removal
Lemmatization
TF-IDF vectorization
Model prediction
```

Output:

```
Churn Probability: 0.91
Prediction: Churn
```

---

# Visualization Reports

The system generates several analytical charts.

---

## Churn Distribution

Purpose:

Show proportion of churn vs retained customers.

Graph:

Pie chart.

Example:

```
Churned Customers: 35%
Retained Customers: 65%
```

---

## Model Accuracy Comparison

Purpose:

Compare performance of machine learning models.

Graph:

Bar chart.

Example:

```
Naive Bayes         82%
Logistic Regression 86%
Random Forest       88%
```

---

## Confusion Matrix

Purpose:

Show prediction correctness.

Example matrix:

```
                Predicted
               Yes    No

Actual Yes     120    30
Actual No      20     230
```

Visualization:

Heatmap.

---

## Prediction Distribution

Purpose:

Show number of predicted churn vs non-churn cases.

Graph:

Bar chart.

---

## Most Important Churn Words

Purpose:

Show frequent churn-related keywords.

Examples:

```
cancel
refund
slow
expensive
support
```

Graph:

Bar chart.

---

# Running the Project

Run training:

```
python src/model_training.py
```

Generate visualizations:

```
python src/visualization.py
```

Make prediction:

```
python src/prediction.py
```

---

# Future Improvements

Potential improvements:

```
Add transformer models (BERT)
Add sentiment analysis
Deploy model using FastAPI
Add real-time prediction API
Use larger datasets
Integrate dashboards (Plotly / Streamlit)
```

---

# Applications

This system can be used for:

```
Customer feedback analysis
Product review monitoring
Customer success analytics
Subscription churn prediction
Support ticket classification
```

---

# Author

Built as an applied NLP project exploring machine learning methods for customer churn detection using text feedback data.

---

# License

MIT License
