import streamlit as st
from annotated_text import annotated_text
from ner import ner
from underthesea import word_tokenize

def tokenizer(row):
    return word_tokenize(row, format="text")

def find_index_entity(start,a,b):
    entity = b.split('-')[1]
    find='I-'+ entity
    result = start
    if (start == len(a)-1):
        result = start
    else:
        i = start +1
        while(i<len(a)):
            if(a[i]==find):
                result = i
                i+=1
            else:
                break
    return result, entity

dict_color = {'SYS.STREET_NUMBER': "#8ef", 'SYS.STREET': "#faa",
'SYS.DISTRICT': "#afa",'SYS.CITY': 'ef8',  'SYS.WARD': "#fea",
'SYS.ADDRESS_LAND': "#8ef", 'SYS.ADDRESS_HAMLET': "#aaf",
'SYS.ADDRESS_VILLAGE': '#e8f', 'SYS.QUARTER': '#abc',
'SYS.ADDRESS_GROUP': '#bca', 'SYS.ADDRESS_LANE': '#cba', 'SYS.TOWN': 'Kteam', 
'SYS.ADDRESS_APARTMENT': 'Kteam',  'SYS.URBAN_AREA': 'Kteam',
'SYS.FLOOR': 'Kteam', 'SYS.ADDRESS_ROOM': 'Kteam',  'SYS.ADDRESS_ALLEY': 'Kteam',
'SYS.LICENSE_PLATE_NUMBER': 'Kteam','SYS.FULL_NAME': 'Kteam',
}
text_ws = []
text = "gia đình tôi mới chuyển đến địa chỉ 469 lâm nhĩ, cẩm lệ"
text_ws.append(tokenizer(text).split(' '))
tag = ner(text_ws)
#Display results of the NLP task


print(text_ws)
print(tag)


