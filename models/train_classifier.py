import sys
import os
import re

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from joblib import dump

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('vader_lexicon',quiet=True)


def load_data(database_filepath):
    """
    load_data
    Loads data from a SQLite database file and prepares it for use in a machine learning model.

    Input:
    database_filepath(str): The path to the SQLite database file.

    Returns:
    X(pandas.Series): 
        A pandas Series containing the messages from the loaded data.
    y(pandas.DataFrame): 
        A pandas DataFrame containing the labels for the loaded data. The DataFrame has the following changes:
        - The 'id', 'message', 'original', and 'genre' columns have been dropped.
    columns(pandas.Index): 
        A pandas Index containing the column names of the y DataFrame.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("MainTable", engine)
    df.dropna(subset=["message"], inplace=True)
    X = df["message"]
    y = df.drop(columns=["id","message","original","genre"])
    return X,y,y.columns



def tokenize(text):
    """
    tokenize
    Normalizes, tokenizes, removes stopwords and lemmatizes text for use in a machine learning model.

    Input:
    text(str): The text to be tokenized.

    Returns:
    text(list): 
        A list containing the normalized, tokenized, stopwords-removed, and lemmatized tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip()) #normalize
    text = word_tokenize(text) #tokanize
    
    # should I even remove stopwords? tweets are short enough as it is
    text = [w for w in text if w not in stopwords.words("english")] 

    # iterate through each token
    text = [WordNetLemmatizer().lemmatize(w) for w in text]

    return text

class SentimentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out =  pd.DataFrame([x for x in X.apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x))])
        return out


def build_model():
    """
    build_model
    Builds a machine learning model using the scikit-learn Pipeline and MultiOutputClassifier classes. Optimizes parameters using Gridsearch.

    Returns:
    model(object): 
        A scikit-learn Pipeline object that includes a TfidfVectorizer, a SentimentTransformer (based on NLTK's SentimentIntensityAnalyzer), and a 
        MultiOutputClassifier(RandomForestClassifier).
    """
    pipeline = Pipeline([
        ("Transf", FeatureUnion([
            ('TfidVect', TfidfVectorizer(tokenizer=tokenize, norm=None)),
            ('Sentiment', SentimentTransformer())
        ])),
        ("clf",MultiOutputClassifier(RandomForestClassifier(n_estimators=12)))
    ])
    parameters = {
        "Transf__TfidVect__norm" : [None,"l1","l2"],
        "Transf__TfidVect__use_idf" : [True,False],
        "clf__estimator__criterion" : ["gini", "entropy"],
        "clf__estimator__n_estimators" : [8,12]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_improved = model.predict(X_test)
    print(classification_report(Y_test, y_pred_improved, target_names=category_names)) 

def save_model(model, model_filepath):
    dump(model, os.path.join(model_filepath , 'model.joblib'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()