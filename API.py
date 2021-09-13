from typing import List
from numpy.lib.function_base import append
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Input, Model
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras_contrib.layers import CRF
import numpy as np
from underthesea import word_tokenize

# batch_size = 64
# epochs = 50
max_len = 75
embedding = 40

def tokenizer(row):
    return word_tokenize(row, format="text")

def load_data(filename):
    df = pd.read_excel(filename)
    df = df.fillna(method = 'ffill')
    return df

df = load_data('data/emandai.xlsx')

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

word2idx, tag2idx, idx2word, idx2tag, num_tag,words, tags = process_data(df)

class TextResBody(BaseModel):
    a: list = []
    # a: List[List[str]]

app = FastAPI()


@app.post("/NER_model/")
async def NER_model(request: TextResBody):
    str_input = request.a
    s = []
    for i in str_input:
        s.append(tokenizer(i).split(' '))
    X = []
    for element in s:
        temp = []
        for i in range(len(element)):
            if element[i] not in word2idx:
                temp.append( word2idx['UNK'])
            else:
                temp.append(word2idx[element[i]])
        X.append(temp)
    # X = [[word2idx[w] for w in a] for a in s]

    # Padding các câu về max_len
    X_test = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word2idx["PAD"])
    model = build_model(num_tag,words)
    model.load_weights("model.hdf5")

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = [[idx2tag[i] for i in row] for row in y_pred]

    result = []
    y_pred_non_PAD = []
    for i in y_pred:
        temp = []
        for j in i:
            if (j!="PAD"):
                temp.append(j)
        y_pred_non_PAD.append(temp)
        
    for i in range(len(s)) : 
        result_tem = []
        j=0
        while(j<len(s[i])):
            if (y_pred_non_PAD[i][j]!='O'):
                end_index,entity = find_index_entity(j,y_pred_non_PAD[i],y_pred_non_PAD[i][j])
                result_tem.append({'Type':entity, "Entity_value":' '.join(('_'.join(s[i][j:end_index+1])).split('_'))})
                j = end_index+1
            else:
                j+=1
        result.append({"Original":str_input[i], 'Entity':result_tem})
    sen = ['sentenece '+str(i) for i in range(len(s))]

    result_json = dict(zip(sen, result))
    return result_json



def find_index_entity(start_index,source_string,target_entity):
    '''
    Function for processing data, convert raw data to numeric data
    Input:
        start_index: start index of the entity in the string
        source_string: string to find entity
        target_entity: Tag
    Output:
        end_index: end index of the entity in the string
        entity_name: name of entity is contained in input string
    '''
    entity_name = target_entity.split('-')[1]
    find_value = 'I-'+ entity_name
    end_index = start_index
    if (start_index == len(source_string)-1):
        end_index = start_index
    else:
        i = start_index +1
        while(i<len(source_string)):
            if(source_string[i] == find_value):
                end_index = i
                i+=1
            else:
                break
    return end_index, entity_name


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)