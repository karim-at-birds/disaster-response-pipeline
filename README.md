# Disaster Response Pipeline Project

This project aims to classify tweets related to disasters using machine learning techniques. The goal is to help responders quickly identify relevant tweets during a crisis and take appropriate actions to help those affected by the disaster. 

The project uses Python and various libraries, including Scikit-learn, Pandas, and NLTK, to process and classify the tweets. The machine learning model is a Random Forest Classifier, trained on text features extracted via Scikit-learns TfidfVectorizer as well as features from NLTK's Sentiment Intensity Analyzer.
The dataset used for training is disaster data from Appen (formally Figure 8).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Files

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- DisasterResponse.db # database to save clean data to
models
|- train_classifier.py
|- model.joblib # saved model
README.md



### Credits

The project was developed as part of the Datascience Nanodegree by Udacity.