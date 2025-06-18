import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from my_data_loader import load_data
from urbanclip_models import UrbanCLIP_init
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor
from datasets import MyDataset
import os
import numpy as np
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from datetime import datetime


"""
UrbanCLIP 主函数
"""

def to_var(x):
    if torch.cuda.is_available():
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
    else:
        x = torch.as_tensor(x, dtype=torch.float32)
    return x

def shuffle_dataset(text, image, label):
    assert len(text) == len(image) == len(label)
    rp = np.random.permutation(len(text))
    text = text[rp]
    image = image[rp]
    label = label[rp]
    return text, image, label


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def extract_embeddings(model, data_loader, device):
    model.eval()
    text_embeds_list, image_embeds_list, labels_list = [], [], []
    with torch.no_grad():
        for text, image, label in data_loader:
            text, image = text.to(device), image.to(device)
            text_embeds, image_embeds = model(text, images=image, return_embeddings=True)
            text_embeds_list.append(text_embeds)
            image_embeds_list.append(image_embeds)
            labels_list.append(label)
    text_embeds = torch.cat(text_embeds_list)
    image_embeds = torch.cat(image_embeds_list)
    labels = torch.cat(labels_list)
    return TensorDataset(text_embeds, image_embeds, labels)

def train_mlp(mlp, train_loader, criterion, optimizer, device, num_epochs):
    mlp.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for text_embeds, image_embeds, labels in train_loader:
            text_embeds, image_embeds, labels = text_embeds.to(device), image_embeds.to(device), labels.to(device)
            inputs = torch.cat((text_embeds, image_embeds), dim=1)
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

def evaluate_mlp(mlp, test_loader, device):
    mlp.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_embeds, image_embeds, labels in test_loader:
            text_embeds, image_embeds, labels = text_embeds.to(device), image_embeds.to(device), labels.to(device)
            inputs = torch.cat((text_embeds, image_embeds), dim=1)
            outputs = mlp(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


def save_results(results, folder='result_urbanclip'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(folder, f'results_{timestamp}.json')
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)


class Args:
    def __init__(self, seed, data_path, dataset_type, num_epoch, seq_len, batch_size=64):
        self.seed = seed
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.num_epoch = num_epoch
        self.seq_len = seq_len
        self.batch_size = batch_size

def main():
    args = Args(
        seed=3,
        data_path='./datasets/traffic-camera-norway-images/',
        dataset_type='lm',  # or 'ds'
        # num_epoch=5,
        num_epoch=50,
        seq_len=80,
        batch_size=64
    )

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Loading data ...')
    text, image, label, W = load_data(args)
    text, image, label = shuffle_dataset(text, image, label)

    # Split the dataset into training and validation sets
    split_idx = int(0.77 * len(text))
    train_text, test_text = text[:split_idx], text[split_idx:]
    train_image, test_image = image[:split_idx], image[split_idx:]
    train_label, test_label = label[:split_idx], label[split_idx:]

    train_data = {'text': train_text, 'image': train_image, 'label': train_label}
    test_data = {'text': test_text, 'image': test_image, 'label': test_label}

    train_loader = DataLoader(dataset=MyDataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=MyDataset(test_data), batch_size=args.batch_size, shuffle=False)

    # 输出训练集和测试集的大小
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Initialize image encoder
    vit = SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=200,
        dim=224,
        depth=3,
        heads=8,
        mlp_dim=512,
        patch_dropout=0.5
    )
    vit = Extractor(vit, return_embeddings_only=True, detach=False)

    # Initialize model
    model = UrbanCLIP_init(
        dim=128,  # model dimension
        img_encoder=vit,  # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
        image_dim=224,  # image embedding dimension, if not the same as model dimensions
        num_tokens=W.shape[0],  # number of text tokens
        unimodal_depth=3,  # depth of the unimodal transformer
        multimodal_depth=3,  # depth of the multimodal transformer
        dim_head=32,  # dimension per attention head
        heads=4,  # number of attention heads
        caption_loss_weight=1.,  # weight on the autoregressive caption loss
        contrastive_loss_weight=1.,  # weight on the contrastive loss between image and text CLS embeddings
    ).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Training loop
    print('Training UrbanCLIP model ...')
    num_epochs = args.num_epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (text, image, label) in enumerate(train_loader):
            text, image, label = text.to(device), image.to(device), label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass to get the loss
            loss = model(text, images=image, return_loss=True)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    # Save the model
    torch.save(model.state_dict(), 'urbanclip_model.pth')

    # Freeze UrbanCLIP model
    print('Freezing UrbanCLIP model ...')
    freeze_model(model)

    # Extract embeddings and create DataLoader for embeddings
    print('Extracting embeddings ...')
    train_dataset = extract_embeddings(model, train_loader, device)
    test_dataset = extract_embeddings(model, test_loader, device)
    train_embed_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_embed_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define MLP model
    input_dim = train_dataset[0][0].size(0) + train_dataset[0][1].size(0)
    mlp = MLP(input_dim=input_dim, hidden_dim=256, output_dim=2).to(device)

    # Train MLP model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
    print('Training MLP model ...')
    train_mlp(mlp, train_embed_loader, criterion, optimizer, device, args.num_epoch)

    # Evaluate MLP model
    preds, labels = evaluate_mlp(mlp, test_embed_loader, device)

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision_0 = precision_score(labels, preds, pos_label=0)
    recall_0 = recall_score(labels, preds, pos_label=0)
    f1_0 = f1_score(labels, preds, pos_label=0)
    precision_1 = precision_score(labels, preds, pos_label=1)
    recall_1 = recall_score(labels, preds, pos_label=1)
    f1_1 = f1_score(labels, preds, pos_label=1)

    results = {
        'num_epochs': num_epochs,
        'dataset': args.dataset_type,
        'accuracy': accuracy,
        'f1': f1,
        'class_0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1': f1_0
        },
        'class_1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1': f1_1
        }
    }

    print('Results:')
    print(results)

    # Save results
    save_results(results)


if __name__ == "__main__":
    main()