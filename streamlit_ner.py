import streamlit as st
from annotated_text import annotated_text
from ner import ner
from underthesea import word_tokenize

def tokenizer(row):
    return word_tokenize(row, format="text")

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

dict_color = {'SYS.STREET_NUMBER': "#8ef", 'SYS.STREET': "#faa",
'SYS.DISTRICT': "#afa",'SYS.CITY': '#bca',  'SYS.WARD': "#fea",
'SYS.ADDRESS_LAND': "#8ef", 'SYS.ADDRESS_HAMLET': "#aaf",
'SYS.ADDRESS_VILLAGE': '#e8f', 'SYS.QUARTER': '#abc',
'SYS.ADDRESS_GROUP': '#bca', 'SYS.ADDRESS_LANE': '#cba', 'SYS.TOWN': 'Kteam', 
'SYS.ADDRESS_APARTMENT': 'Kteam',  'SYS.URBAN_AREA': 'Kteam',
'SYS.FLOOR': 'Kteam', 'SYS.ADDRESS_ROOM': 'Kteam',  'SYS.ADDRESS_ALLEY': 'Kteam',
'SYS.LICENSE_PLATE_NUMBER': 'Kteam','SYS.FULL_NAME': 'Kteam',
}
st.title("DEMO NAME ENTITY RECOGNITION")
#Textbox for text user is entering
st.subheader("Please enter your text")
text = st.text_input('Enter text') #text is stored in this variable
st.header("Results")
text_ws =[]
text_ws.append(tokenizer(text).split(' '))
tag = ner(text_ws)
#Display results of the NLP task

display = []
for i in range(len(tag)):
    j = 0
    while(j <len(tag[i])):
        if (tag[i][j]=='O'):
            display.append(' '.join(text_ws[i][j].split('_'))+" ")
            j+=1
        else:
            end_index,entity = find_index_entity(j,tag[i],tag[i][j])
            value_tag = tag[i][j].split('-')[1]
            display.append((' '.join(('_'.join(text_ws[i][j:end_index+1])).split('_'))+" ",value_tag,dict_color[value_tag]))
            j = end_index+1

annotated_text(*display)



