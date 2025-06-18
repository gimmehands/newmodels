import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime


def load_all_npy_files(base_path):
    all_features = []
    all_labels = []

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            features_path = os.path.join(folder_path, 'features.npy')
            labels_path = os.path.join(folder_path, 'labels.npy')
            if os.path.exists(features_path) and os.path.exists(labels_path):
                features = np.load(features_path)
                labels = np.load(labels_path)
                all_features.append(features)
                all_labels.append(labels)

    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    return all_features, all_labels


# def plot_tsne(features, labels, title, save_path):
#     tsne = TSNE(n_components=2, random_state=0)
#     tsne_results = tsne.fit_transform(features)
#
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.get_cmap('Dark2', 2), alpha=0.6)
#     plt.colorbar(scatter, ticks=[0, 1])
#     plt.title(title)
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#
#     # Save the plot
#     os.makedirs(save_path, exist_ok=True)
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(os.path.join(save_path, f"{title}_{current_time}.png"))
#     plt.close()

def plot_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.get_cmap('Dark2', 2), alpha=0.6)
    plt.axis('off')

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_path, f"tsne_{current_time}.eps"), format='eps', bbox_inches='tight', pad_inches=0)

    plt.close()


def main(base_path, key):
    features, labels = load_all_npy_files(base_path)
    save_path = f"./tsne_result/{key}"
    # plot_tsne(features, labels, key, save_path)
    plot_tsne(features, labels, save_path)

if __name__ == "__main__":
    base_path_dict = {
        "noES": "./result/as_new/seed60/noES/K1/epoch44/",
        "noFUS": "./result/as_new/seed60/noFUS/K0/epoch48/",
        "noCAI": "./result/as_new/seed60/noCAI/K0/epoch48/",
        "noCLL": "./result/as_new/seed60/noCLL/K0/epoch41/",
        # "noCLL": "./result/as_new/seed60/noCLL/K3/epoch45/",
        # "noCLL": "./result/as_new/seed60/noCLL/K3/epoch19/",
        "noTE": "./result/as_new/seed60/noTE/K0/epoch48/",
    }
    for key, base_path in base_path_dict.items():
        print(f"Plotting t-SNE for {key}...")
        main(base_path, key)