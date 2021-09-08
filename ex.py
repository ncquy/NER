import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Input, Model
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras_contrib.layers import CRF
import numpy as np
from underthesea import word_tokenize
import streamlit as st
from annotated_text import annotated_text

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


def build_model(num_tags,words, hidden_size = 50):
    # Model architecture
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=len(words) + 2, output_dim=embedding, input_length=max_len, mask_zero=False)(input)
    model = Bidirectional(LSTM(units=hidden_size, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(hidden_size, activation="relu"))(model)
    crf = CRF(num_tags + 1)  # CRF layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    model.summary()
    return model


def process():
    word2idx, tag2idx, idx2word, idx2tag, num_tag,words, tags = process_data(df)
    st.title("Demo nhận dạng thực thể chứ tên")

    #Textbox for text user is entering
    st.subheader("Nhập vào đoạn text")
    text = st.text_input('Enter text') #text is stored in this variable

    #Display results of the NLP task

    # s = "Chào em, chị là Minh"
    element = word_tokenize(text)
    X = []

    for i in range(len(element)):
        if element[i] not in word2idx:
            X.append( word2idx['UNK'])
        else:
            X.append(word2idx[element[i]])
    # X = [[word2idx[w] for w in a] for a in s]

    # Padding các câu về max_len
    X +=[0]*(max_len-len(X))
    X_test = []
    X_test.append(X)
    X_test = pad_sequences(maxlen = max_len, sequences = X_test, padding = "post", value = word2idx["PAD"])


    model = build_model(num_tag,words)
    model.load_weights("model.hdf5")

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = [[idx2tag[i] for i in row] for row in y_pred]







