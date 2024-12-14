import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

import kagglehub
path = kagglehub.dataset_download("jackksoncsie/spam-email-dataset")
data_01 = pd.read_csv(path + "/emails.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

X_sentences = [clean_text(text) for text in data_01['text']]

vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

X = vectorizer.fit_transform(X_sentences).toarray()
Y = data_01["spam"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print(f"Before SMOTE - Train class distribution:\n{Y_train.value_counts()}")

smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

print(f"After SMOTE - Resampled train class distribution:\n{Y_train_resampled.value_counts()}")

model_02 = MultinomialNB()

model_02.fit(X_train_resampled, Y_train_resampled)

joblib.dump(model_02, "Model.joblib")

joblib.dump(vectorizer, "vectorizer.joblib")
