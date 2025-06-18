import os
import pandas as pd

def extract_data(file_path, label):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            image_urls = lines[i + 1].strip().split('|')
            tweet = lines[i + 2].strip()
            if tweet:  # Check if tweet is not empty
                data.append((image_urls, tweet, label))
    return data

def check_image_exists(image_urls, rumor_dir, nonrumor_dir):
    for url in image_urls:
        image_name = url.split('/')[-1]
        # image_name = url.split('/')[-1].split('.')[0]
        if os.path.exists(os.path.join(rumor_dir, image_name)) or os.path.exists(os.path.join(nonrumor_dir, image_name)):
            return image_name.split('.')[0]
    return None

def create_csv(data, csv_path, rumor_dir, nonrumor_dir):
    rows = []
    for image_urls, tweet, label in data:
        image_name = check_image_exists(image_urls, rumor_dir, nonrumor_dir)
        if image_name:
            rows.append([image_name, tweet, label])
    df = pd.DataFrame(rows, columns=['image_url', 'tweet', 'label'])
    df.to_csv(csv_path, index=False)

def main():
    base_path = './weibo/tweets/'
    rumor_dir = './weibo/rumor_images/'
    nonrumor_dir = './weibo/nonrumor_images/'

    train_rumor_data = extract_data(os.path.join(base_path, 'train_rumor.txt'), 1)
    train_nonrumor_data = extract_data(os.path.join(base_path, 'train_nonrumor.txt'), 0)
    test_rumor_data = extract_data(os.path.join(base_path, 'test_rumor.txt'), 1)
    test_nonrumor_data = extract_data(os.path.join(base_path, 'test_nonrumor.txt'), 0)

    train_data = train_rumor_data + train_nonrumor_data
    test_data = test_rumor_data + test_nonrumor_data

    create_csv(train_data, './weibo/train_data.csv', rumor_dir, nonrumor_dir)
    create_csv(test_data, './weibo/test_data.csv', rumor_dir, nonrumor_dir)

if __name__ == '__main__':
    main()