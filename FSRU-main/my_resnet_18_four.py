import argparse
import random
import numpy as np
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from sklearn.metrics import classification_report
from arguments import parse_arguments
from four_data_loader import *
from torchvision.models import resnet18
import json
import datetime
from dataset_type import dataset_type_dict

warnings.filterwarnings("ignore")

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_np(x):
    return x.data.cpu().numpy()

class MyDataset(Dataset):
    def __init__(self, image, label):
        self.image = list(image)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.image[item], self.label[item]

def shuffle_dataset(image, label):
    index = np.arange(len(label))
    np.random.shuffle(index)
    image = image[index]
    label = label[index]
    return image, label

def save_results(args, results):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_file = os.path.join(args.output_path, f'results_{time.strftime('%Y%m%d-%H%M')}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Loading data ...')

    text, image, label, W = load_data(args)
    image, label = shuffle_dataset(image, label)

    dataset_size = len(label)
    train_size = int(0.77 * dataset_size)
    test_size = dataset_size - train_size
    train_image, test_image = image[:train_size], image[train_size:]
    train_label, test_label = label[:train_size], label[train_size:]

    train_dataset = MyDataset(train_image, train_label)
    test_dataset = MyDataset(test_image, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)  # Change to 4 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_acc = 0.0
    class_metrics = {i: {'precision': 0, 'recall': 0, 'f1': 0} for i in range(4)}  # Change to 4 classes

    for epoch in range(args.num_epoch):
        start_time = time.time()  # Record start time

        model.train()
        train_losses = []
        train_acc = []

        for images, labels in train_loader:
            images = to_var(images).to(device)
            labels = to_var(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_acc.append((predicted == labels).sum().item() / labels.size(0))

        model.eval()
        valid_acc = 0.0
        valid_pre = 0.0
        valid_recall = 0.0
        valid_f1 = 0.0
        class_report = {i: {'precision': 0, 'recall': 0, 'f1': 0} for i in range(4)}  # Change to 4 classes

        with torch.no_grad():
            for images, labels in test_loader:
                images = to_var(images).to(device)
                labels = to_var(labels).to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                valid_acc += (predicted == labels).sum().item() / labels.size(0)

                report = classification_report(labels.cpu(), predicted.cpu(), output_dict=True)
                valid_pre += report['weighted avg']['precision']
                valid_recall += report['weighted avg']['recall']
                valid_f1 += report['weighted avg']['f1-score']

                for cls in class_report.keys():
                    class_report[cls]['precision'] += report[str(cls)]['precision']
                    class_report[cls]['recall'] += report[str(cls)]['recall']
                    class_report[cls]['f1'] += report[str(cls)]['f1-score']

        valid_acc /= len(test_loader)
        valid_pre /= len(test_loader)
        valid_recall /= len(test_loader)
        valid_f1 /= len(test_loader)

        for cls in class_report.keys():
            class_report[cls]['precision'] /= len(test_loader)
            class_report[cls]['recall'] /= len(test_loader)
            class_report[cls]['f1'] /= len(test_loader)

        end_time = time.time()  # Record end time
        duration = end_time - start_time  # Calculate duration

        print(f'Epoch[{epoch + 1}/{args.num_epoch}], Duration: {duration:.2f}s, Loss: {np.mean(train_losses):.8f}, '
              f'Train_Accuracy: {np.mean(train_acc):.5f}, Valid_accuracy: {valid_acc:.5f}')

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_pre = valid_pre
            best_valid_recall = valid_recall
            best_valid_f1 = valid_f1
            class_metrics = class_report
            print('Best...')

    results = {
        'accuracy': best_valid_acc,
        'precision': best_valid_pre,
        'recall': best_valid_recall,
        'f1': best_valid_f1,
        'class_metrics': class_metrics
    }

    save_results(args, results)
    print(results)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()

    main(args)