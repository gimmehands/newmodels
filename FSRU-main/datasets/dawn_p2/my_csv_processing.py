import pandas as pd
import string
import os
import re


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Remove punctuation from the 'tweet' column
    df['tweet'] = df['tweet'].apply(remove_punctuation)
    # Normalize whitespace in the 'tweet' column
    df['tweet'] = df['tweet'].apply(normalize_whitespace)

    # Create a new file name for the processed data
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_processed{ext}"

    # Save the processed data to a new CSV file
    df.to_csv(new_file_path, index=False)


# Process both train_data.csv and test_data.csv
process_csv('train_data.csv')
process_csv('test_data.csv')