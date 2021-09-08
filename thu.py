import pandas as pd

from underthesea import word_tokenize

# batch_size = 64
# epochs = 50
max_len = 75
embedding = 40

def load_data(filename):
    df = pd.read_excel(filename)
    df = df.fillna(method = 'ffill')
    return df


df = load_data('data/data_train_test.xlsx')

def process_data(df):
    # Xây dựng vocab cho word và tag
    words = list(df['word'].unique())
    tags = list(df['tag'].unique())

    # Tạo dict word to index, thêm 2 từ đặc biệt là Unknow và Padding
    word2idx = {w : i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    # Tạo dict tag to index, thêm 1 tag đặc biệt và Padding
    tag2idx = {t : i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    # Tạo 2 dict index to word và index to tag
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}
    num_tag = df['tag'].nunique()
    # Save data
    return word2idx, tag2idx, idx2word, idx2tag, num_tag, words, tags


word2idx, tag2idx, idx2word, idx2tag, num_tag,words, tags = process_data(df)
s = "Chiều cuối thu, vùng biển Nghi Xuân ảm đạm"
element = word_tokenize(s)
X = []

for i in range(len(element)):
    if element[i] not in word2idx:
        X.append( word2idx['UNK'])
    else:
        X.append(word2idx[element[i]])



