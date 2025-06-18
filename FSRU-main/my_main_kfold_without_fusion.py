"""
An Lao
"""
import argparse
import random
import math
import numpy as np
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import classification_report
from arguments import parse_arguments
from my_data_loader import *
from datasets import MyDataset
from model import FSRU, FSRU_new, FSRU_new_without_fusion
from loss import FullContrastiveLoss, SelfContrastiveLoss, CaptionLoss
import json
from dataset_type import dataset_type_dict

warnings.filterwarnings("ignore")

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def to_var(x):
    if torch.cuda.is_available():
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
    else:
        x = torch.as_tensor(x, dtype=torch.float32)
    return x

def to_np(x):
    return x.data.cpu().numpy()

def get_kfold_data(k, i, text, image ,label):
    fold_size = text.shape[0] // k

    val_start = i * fold_size
    if i != k-1:
        val_end = (i + 1) * fold_size
        text_valid, image_valid, label_valid = text[val_start:val_end], image[val_start:val_end], label[val_start:val_end]
        text_train = np.concatenate((text[0:val_start], text[val_end:]), axis=0)
        image_train = np.concatenate((image[0:val_start], image[val_end:]), axis=0)
        label_train = np.concatenate((label[0:val_start], label[val_end:]), axis=0)
    else:
        text_valid, image_valid, label_valid = text[val_start:], image[val_start:], label[val_start:]
        text_train = text[0:val_start]
        image_train = image[0:val_start]
        label_train = label[0:val_start]

    return text_train, image_train, label_train, text_valid, image_valid, label_valid

def count(labels):
    r, nr = 0, 0
    for label in labels:
        if label == 0:
            nr += 1
        elif label == 1:
            r += 1
    return r, nr

def shuffle_dataset(text, image, label):
    assert len(text) == len(image) == len(label)
    rp = np.random.permutation(len(text))
    text = text[rp]
    image = image[rp]
    label = label[rp]

    return text, image, label


def save_results(args, results):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Generate the output file name with date and time
    output_file = os.path.join(args.output_path, f"results_{time.strftime('%Y%m%d-%H%M')}.json")

    # Prepare the results dictionary
    results_dict = {
        'seed': args.seed,
        'dataset_type': args.dataset_type,
        'alpha': args.alpha,
        'beta': args.beta,
        'caption_rate': args.caption_rate,
        'num_epoch': args.num_epoch,
        'remarks': args.remarks,
        'results': results
    }

    # Save the results to the file
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

def main(args):
    # device = torch.device(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Loading data ...')

    text, image, label, W = load_data(args)
    text, image, label = shuffle_dataset(text, image, label)

    K = args.k
    print('Using K:', K, 'fold cross validation...')

    valid_acc_sum, valid_pre_sum, valid_recall_sum, valid_f1_sum = 0., 0., 0., 0.
    valid_nr_pre_sum, valid_nr_recall_sum, valid_nr_f1_sum = 0., 0., 0.
    valid_r_pre_sum, valid_r_recall_sum, valid_r_f1_sum = 0., 0., 0.

    train, valid = {}, {}
    for i in range(K):
        print('-' * 25, 'Fold:', i + 1, '-' * 25)
        train['text'], train['image'], train['label'], valid['text'], valid['image'], valid['label'] = \
            get_kfold_data(K, i, text, image, label)

        train_loader = DataLoader(dataset=MyDataset(train), batch_size=args.batch_size, shuffle=False)
        valid_loader = DataLoader(dataset=MyDataset(valid), batch_size=args.batch_size, shuffle=False)

        print('Building model...')

        model = FSRU_new_without_fusion(W, args.vocab_size, args.d_text, args.seq_len, args.img_size, args.patch_size, args.d_model,
                     args.num_filter, args.num_class, args.num_layer, args.dropout)
        model.to(device)

        if torch.cuda.is_available():
            print("CUDA")
            model.cuda()

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1 = 0., 0., 0., 0.

        loss_list = []
        acc_list = []
        for epoch in range(args.num_epoch):
            train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
            start_time = time.time()
            cls_loss = []

            # train
            model.train()
            for j, (train_text, train_image, train_labels) in enumerate(train_loader):
                num_r, num_nr = count(train_labels)
                train_text, train_image, train_labels = to_var(train_text), to_var(train_image), to_var(train_labels)

                # Forward + Backward + Optimize
                criterion_full = FullContrastiveLoss(batch_size=train_text.shape[0], num_r=num_r, num_nr=num_nr)
                criterion_self = SelfContrastiveLoss(batch_size=train_text.shape[0])
                criterion_caption = CaptionLoss()
                optimizer.zero_grad()

                text_outputs, image_outputs, label_outputs, _, logits, text_embed = model(train_text, train_image)

                loss = criterion(label_outputs, train_labels.long())
                loss_full = criterion_full(text_outputs, image_outputs, train_labels.long())
                loss_self = criterion_self(text_outputs, image_outputs)
                # loss_caption = criterion_caption(logits, text_embed)
                loss_caption = criterion_caption(logits, train_text)
                # loss_caption = 0

                train_loss = loss + args.alpha * loss_full + args.beta * loss_self + loss_caption * args.caption_rate
                # print('Loss:',loss.item(), 'Loss_full:', loss_full.item() * args.alpha, 'Loss_self:', loss_self.item() * args.beta,
                #       'Loss_caption:', loss_caption.item() * args.caption_rate)
                # print('Train_loss:', train_loss.item())
                train_loss.backward()
                optimizer.step()
                pred = torch.max(label_outputs, 1)[1]
                train_accuracy = torch.eq(train_labels, pred.squeeze()).float().mean()  # .sum() / len(train_labels)
                train_losses.append(train_loss.item())
                train_acc.append(train_accuracy.item())
                cls_loss.append(loss.item())

            if epoch % args.decay_step == 0:
                for params in optimizer.param_groups:
                    params['lr'] *= args.decay_rate

            # valid
            model.eval()
            valid_pred, valid_y = [], []
            with torch.no_grad():
                for j, (valid_text, valid_image, valid_labels) in enumerate(valid_loader):
                    valid_text, valid_image, valid_labels = to_var(valid_text), to_var(valid_image), to_var(
                        valid_labels)

                    _, _, label_outputs, _, _, _ = model(valid_text, valid_image)
                    label_outputs = F.softmax(label_outputs, dim=1)
                    pred = torch.max(label_outputs, 1)[1]
                    if j == 0:
                        valid_pred = to_np(pred.squeeze())
                        valid_y = to_np(valid_labels.squeeze())
                    else:
                        valid_pred = np.concatenate((valid_pred, to_np(pred.squeeze())), axis=0)
                        valid_y = np.concatenate((valid_y, to_np(valid_labels.squeeze())), axis=0)
            # cur_valid_acc = np.mean(valid_acc)
            cur_valid_acc = metrics.accuracy_score(valid_y, valid_pred)
            valid_pre = metrics.precision_score(valid_y, valid_pred, average='macro')
            valid_recall = metrics.recall_score(valid_y, valid_pred, average='macro')
            valid_f1 = metrics.f1_score(valid_y, valid_pred, average='macro')
            duration = time.time() - start_time
            print(
                'Epoch[{}/{}], Duration:{:.8f}, Loss:{:.8f}, Train_Accuracy:{:.5f}, Valid_accuracy:{:.5f}'.format(
                    epoch + 1, args.num_epoch, duration, np.mean(train_losses), np.mean(train_acc),
                    cur_valid_acc))
            loss_list.append(np.mean(cls_loss))
            acc_list.append(cur_valid_acc)

            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                best_valid_pre = valid_pre
                best_valid_recall = valid_recall
                best_valid_f1 = valid_f1
                print('Best...')
                # print(metrics.classification_report(valid_y, valid_pred, digits=4))
                target_names = ['non-rumor', 'rumor']
                report = metrics.classification_report(valid_y, valid_pred, output_dict=True, target_names=target_names)
                nr_report = report['non-rumor']
                best_valid_nr_pre = nr_report['precision']
                best_valid_nr_recall = nr_report['recall']
                best_valid_nr_f1 = nr_report['f1-score']
                r_report = report['rumor']
                best_valid_r_pre = r_report['precision']
                best_valid_r_recall = r_report['recall']
                best_valid_r_f1 = r_report['f1-score']

        valid_acc_sum += best_valid_acc
        valid_pre_sum += best_valid_pre
        valid_recall_sum += best_valid_recall
        valid_f1_sum += best_valid_f1
        print('best_valid_acc:{:.6f}, best_valid_pre:{:.6f}, best_valid_recall:{:.6f}, best_valid_f1:{:.6f}'.
              format(best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1))
        valid_nr_pre_sum += best_valid_nr_pre
        valid_nr_recall_sum += best_valid_nr_recall
        valid_nr_f1_sum += best_valid_nr_f1
        valid_r_pre_sum += best_valid_r_pre
        valid_r_recall_sum += best_valid_r_recall
        valid_r_f1_sum += best_valid_r_f1

    print('=' * 40)
    print('Accuracy:{:.5f}, F1:{:.5f}'.format(valid_acc_sum / K, valid_f1_sum / K))

    one_name, zero_name = "Rumor", "Non-Rumor"
    # if args.dataset_type == 'af':
    #     one_name, zero_name = "Fire", "Accident"
    # elif args.dataset_type == 'ds':
    #     one_name, zero_name = "Sparse", "Dense"
    # # 获取dataset_type的两个字符对应的名称
    type_1 = dataset_type_dict[args.dataset_type[0]]
    type_2 = dataset_type_dict[args.dataset_type[1]]

    # 使用这些名称替换one_name和zero_name的赋值
    one_name, zero_name = type_1, type_2

    print('{} Precision:{:.5f}, {} Recall:{:.5f}, {} F1:{:.5f}'.format(
        one_name, valid_r_pre_sum / K, one_name, valid_r_recall_sum / K, one_name, valid_r_f1_sum / K))
    print('{} Precision:{:.5f}, {} Recall:{:.5f}, {} F1:{:.5f}'.format(
        zero_name, valid_nr_pre_sum / K, zero_name, valid_nr_recall_sum / K, zero_name, valid_nr_f1_sum / K))

    # Collect results
    results = {
        'accuracy': valid_acc_sum / K,
        'f1': valid_f1_sum / K,
        'one_name': one_name,
        'one_precision': valid_r_pre_sum / K,
        'one_recall': valid_r_recall_sum / K,
        'one_f1': valid_r_f1_sum / K,
        'zero_name': zero_name,
        'zero_precision': valid_nr_pre_sum / K,
        'zero_recall': valid_nr_recall_sum / K,
        'zero_f1': valid_nr_f1_sum / K
    }

    # Save results to file
    save_results(args, results)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()

    main(args)
