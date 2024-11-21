This repository contains the code for performing sentiment analysis on a dataset using Natural Language Processing (NLP) techniques. The main steps include data preprocessing, feature extraction, model training, and evaluation. The project uses various tools such as Word2Vec, TF-IDF, and machine learning algorithms like SVM and Naive Bayes.

### Requirements
Before you begin, ensure you have installed the following dependencies:

```js
pip install pandas seaborn imbalanced-learn gensim wordcloud matplotlib nltk ydata-profiling scikit-learn joblib Sastrawi
```

Steps 
### 1. Import Libraries
Start by importing necessary libraries for data manipulation, NLP processing, and machine learning:

```js
import os, pickle, re
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from gensim.models import Word2Vec
from ydata_profiling import ProfileReport

import nltk, string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist
from random import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.classify import SklearnClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc

%matplotlib inline
```

### 2. Load and Explore Dataset
Load the dataset and examine its structure:
```js
DATA_PATH = './dataset.csv'
df = pd.read_csv(DATA_PATH)
df.head()
df.info()
```

Check the distribution of classes:
```js
count = df['class'].value_counts()
labels = list(count.index)

plt.figure(figsize=(4, 4))
plt.pie(count, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution', fontsize=16)
plt.axis('equal')  
plt.show()
```
![image](https://github.com/user-attachments/assets/4466229d-a8a1-4ff2-88d9-88f1a4537981)

### 3. Generate Word Clouds
Create word clouds for each sentiment class (Negative, Positive, Neutral):

```js
def generate_word_cloud(dataframe, text_column, class_column=None, target_class=None):
    if class_column and target_class:
        text_data = dataframe[dataframe[class_column] == target_class][text_column]
    else:
        text_data = dataframe[text_column]

    text_data = text_data.dropna().astype(str)
    text_combined = " ".join(text_data)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(target_class + ' Sentiment')
    plt.axis('off')
    
generate_word_cloud(df, text_column='text', class_column='class', target_class='Negative')
generate_word_cloud(df, text_column='text', class_column='class', target_class='Positive')
generate_word_cloud(df, text_column='text', class_column='class', target_class='Neutral')
```

### 4. Preprocess Text Data
Tokenize the text, remove punctuation, stopwords, and apply stemming:
```js
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def preprocessing(document):
    words = word_tokenize(document.lower())
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in stop_words]
    return words

df['text_preprocessed'] = df['text'].apply(preprocessing)
```

### 5. Handle Class Imbalance
Use oversampling to balance the dataset:

```js
ros = RandomOverSampler(sampling_strategy={cls: max_count for cls in count.index})
X_resampled, y_resampled = ros.fit_resample(df.drop('class', axis=1), df['class'])

df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=df.drop('class', axis=1).columns),
                          pd.Series(y_resampled, name='class')], axis=1)
```

![image](https://github.com/user-attachments/assets/60d508e7-58ef-47c4-b0c0-fe8ff85d0c34)

### 6. Feature Extraction
Apply TF-IDF and Word2Vec for feature extraction:
```js
# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_preprocessed'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf = pd.concat([df[['text_preprocessed', 'class']], tfidf_df], axis=1)

# Word2Vec Model
tokenized_text = [word_tokenize(text.lower()) for text in df['text_preprocessed']]
word2vec_model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, sg=0)
word2vec_model.save("word2vec_model")
```

### 7. Train Machine Learning Models
Train SVM and Naive Bayes classifiers:
```js
X_train, X_test, y_train, y_test = train_test_split(X_resampled, df['class'], random_state=42)

# SVM Model
SVM = svm.SVC(probability=True)
SVM.fit(X_train, y_train)
joblib.dump(SVM, 'svm_model.joblib')

# Naive Bayes Model
nb_model = MultinomialNB().fit(X_train, y_train)
joblib.dump(nb_model, 'multinomial_nb_model.joblib')
```

### 8. Evaluate Model Performance
Evaluate model performance using various metrics like accuracy, precision, recall, and F1-score:
```js
def check_scores(clf, X_train, X_test, y_train, y_test):
    model = OneVsRestClassifier(clf).fit(X_train, y_train)
    predicted_class = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, predicted_class))

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, predicted_class)

    print("Train accuracy score:", train_accuracy)
    print("Test accuracy score:", test_accuracy)
```

![image](https://github.com/user-attachments/assets/3868fadd-bcec-4af0-b361-83b5d5c6244f)




