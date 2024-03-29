# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe

    Input:
    messages_filepath: filepath to messages csv file
    categories_filepath: filepath to categories csv file

    Returns:
    df datafram merging categories and messages
    '''

# load messages dataset
    messages = pd.read_csv(messages_filepath)
# load categories dataset
    categories = pd.read_csv(categories_filepath)
# merge datasets
    df = messages.merge(categories, on="id")
    return df

def clean_data(df):
    """
    clean_data
    Cleans and prepares a pandas DataFrame containing tweets.

    Input:
    df(pandas.DataFrame): The DataFrame containing the data to be cleaned.

    Returns:
    df(pandas.DataFrame): 
        The cleaned DataFrame, with the following changes:
        - The original 'categories' column has been split into separate columns, with category names as column headers.
        - The values in the category columns have been converted to integers.
        - Duplicates have been removed from the DataFrame.
        - The "related" category has been binarised.
    """

# create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
# select the first row of the categories dataframe
    row = categories.iloc[0,:]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
    category_colnames = row.str[:-2]

# rename the columns of `categories`
    categories.columns = category_colnames
# converting to int
    categories = categories.applymap(lambda x:int(x[-1]))

# drop the original categories column from `df`
    df.drop(columns="categories", inplace=True)

# binarise "related" category
    df.related.replace(2,1,inplace=True)

# remove duplicates
    df.drop_duplicates(inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    return df

def save_data(df, database_filename):
    """
    save_data
    Save DataFrame to SQLite database.
    Input:
    df (pandas.DataFrame): DataFrame to save to the database.
    database_filename (str): File path to the SQLite database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MainTable', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()