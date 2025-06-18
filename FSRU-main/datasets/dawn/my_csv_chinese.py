import pandas as pd
import os
import re

def contains_chinese(text):
    # Check if the text contains any Chinese characters
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def detect_chinese_in_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Iterate through the rows and check for Chinese content in the 'tweet' column
    for index, row in df.iterrows():
        if contains_chinese(row['tweet']):
            print(f"Row {index + 1}:")
            print(f"Image URL: {row['image_url']}")
            print(f"Tweet: {row['tweet']}")
            print(f"Label: {row['label']}")
            print("-" * 40)
            # df.drop(index, inplace=True)  # Delete the row containing Chinese characters

    # Save the modified DataFrame back to the CSV file
    # df.to_csv(file_path, index=False)

# Process both train_data.csv and test_data.csv
if __name__ == '__main__':
    detect_chinese_in_csv('train_data.csv')
    detect_chinese_in_csv('test_data.csv')