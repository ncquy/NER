import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Input, Model
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras_contrib.layers import CRF
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
# from seqeval.metrics import f1_score, accuracy_score, recall_score


def load_data(filename):
    df = pd.read_excel(filename)
    df = df.fillna(method = 'ffill')
    return df

def build_vocab(df):
    '''
    Function for buliding vovab for dataset include list of words and tags in dataset, convert sentence
    to number vector
    Input:
        df: dataframe contains word, POS, tag 
    Output:
        word2idx = {'word':index} 
        idx2word = {index:'word'}
        tag2idx = {'tag':index}
        idx2word = {index:'tag'}
        num_tag: number of tag appear in dataset
        words: list of words appear in dataset
        tags: list of tags appear in dataset
    '''

    # Build vocab of words and tags for dataset
    words = list(df['word'].unique())
    tags = list(df['tag'].unique())

    # Build dictionary word to index {'word':index} add "UNK" for word haven't in dataset and "PAD" for padding
    word2idx = {w : i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    # Build dictionary tag to index {'tag':index} add "PAD" for padding
    tag2idx = {t : i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    # Build dictionary index to word {index:'word'} and index to tag {index:'tag'}
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}

    #The number of different tags appearing in the dataset
    num_tag = df['tag'].nunique()
    
    return word2idx, tag2idx, idx2word, idx2tag, num_tag, words, tags

def process_data(sentences,word2idx,tag2idx,num_tag): 
    '''
    Function for processing data, convert raw data to numeric data
    Input:
        sentences: list of sentences for traning or testing, each element is ('word','POS','tag')
        word2idx: dictionary word to index {'word':index}
        tag2idx: dictionary tag to index {'tag':index}
        num_tag: the number of different tags appearing in the dataset
    Output:
        X: numeric vector of sentences input after converting to index
        y: label that is numeric vector of tags after converting to index
    '''   
    # Convert sentences to vector of index
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    # Padding sentences to max_len
    X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word2idx["PAD"])
    # Convert tags to vector of index
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    # Padding tags to max_len
    y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = tag2idx["PAD"])
    # Convert y to one-hot
    y = [to_categorical(i, num_classes = num_tag + 1) for i in y]
    
    return X,y

def build_model(num_tags,words, hidden_size = 50):

    input = Input(shape=(max_len,))
    #To embed text sentences, convert the word indexes into fixed n-dimensional vectors
    model = Embedding(input_dim=len(words) + 2, output_dim=embedding, input_length=max_len, mask_zero=False)(input)
    #LSTM
    model = Bidirectional(LSTM(units=hidden_size, return_sequences=True, recurrent_dropout=0.1))(model)
    #TimeDistributed Layer to get the Dense vector for each word at each step
    model = TimeDistributed(Dense(hidden_size, activation="relu"))(model)
    #CRF
    crf = CRF(num_tags + 1) 
    out = crf(model)  

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model

def embedding_sentence(s,word2idx):
    '''
    Function for embedding sentence, convert sentences to vector of index 
        s: list of sentences need to convert 
        word2idx: dictionary word to index {'word':index}
    Output:
        X: numeric vector of sentences input after converting to index
    ''' 
    # Convert sentences to vector of index
    X = [[word2idx[w] for w in a] for a in s]
    # Padding các câu về max_len
    X_test = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word2idx["PAD"])
    return X_test

class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s: [(w, p, t) for w, p, t in zip(s['word'].values.tolist(),
                                                      s['pos'].values.tolist(),
                                                      s['tag'].values.tolist())]
        self.grouped = self.df.groupby("sentence").apply(agg)
        self.sentences = [s for s in self.grouped]


batch_size = 64
epochs = 20
max_len = 75
embedding = 40
num_folds = 5
acc_per_fold = []
loss_per_fold = []
acc_train_per_fold = []
loss_train_per_fold = []
acc_val_per_fold = []
loss_val_per_fold = []

#Build vocab
df = load_data('data/emandai.xlsx')
word2idx, tag2idx, idx2word, idx2tag, num_tag, words, tags = build_vocab(df)

#Process data for traning
getter_train = sentence(df)
sentences_train = getter_train.sentences

X_train, y_train = process_data(sentences_train,word2idx,tag2idx,num_tag)
X_train = np.array(X_train)
y_train = np.array(y_train)

model = build_model(num_tag,words)
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X_train, y_train):

    # Define callbacks
    checkpoint = ModelCheckpoint(filepath = 'model.hdf5',
                                verbose = 0,
                                mode = 'auto',
                                save_best_only = True,
                                monitor='val_loss')

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X_train[train], np.array(y_train[train]), batch_size=batch_size, epochs=epochs,
                        validation_split=0.2, callbacks=[checkpoint])

    acc_train_per_fold.append(history.history['crf_viterbi_accuracy'])
    loss_train_per_fold.append(history.history['loss'])
    acc_val_per_fold.append(history.history['val_crf_viterbi_accuracy'])
    loss_val_per_fold.append(history.history['val_loss'])
    # Generate generalization metrics
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


    # summarize history for accuracy


acc_train = np.mean(acc_train_per_fold,axis=0)
acc_val = np.mean(acc_val_per_fold,axis=0)
loss_train = np.mean(loss_train_per_fold,axis=0)
loss_val = np.mean(loss_val_per_fold,axis=0)



plt.plot(acc_train)
plt.plot(acc_val)
plt.title('model accuracy ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
    # summarize history for loss

plt.plot(loss_train)
plt.plot(loss_val)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.show()