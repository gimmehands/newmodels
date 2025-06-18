import pickle


def load_vectors(file_path):
    w2v = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip the first line
        next(f)
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = list(map(float, values[1:]))
            w2v[word] = vector
    return w2v


def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    txt_file_path = './tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
    pickle_file_path = 'w2v.pickle'

    w2v = load_vectors(txt_file_path)
    save_pickle(w2v, pickle_file_path)
    print(f"Word vectors saved to {pickle_file_path}")


if __name__ == '__main__':
    main()